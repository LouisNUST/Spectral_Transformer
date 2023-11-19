import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from functools import partial
from torch import nn, einsum
from FFO import *
# from .FFO import *

import math

class LinearConv(nn.Module):
    def __init__(self, c_in,c_out,bias):
        super(LinearConv, self).__init__()
        """
        x [B,C,H,W] ->conv1*1 [B,C,H,W]
        """
        self.linear=nn.Linear(c_in,c_out,bias=bias)
    def forward(self,x):
        B,C,H,W=x.shape
        x=x.reshape(B,C,H*W).permute(0,2,1)  # B N C
        x=self.linear(x)
        x=x.permute(0,2,1).reshape(B,-1,H,W)
        return x
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class Mlp(nn.Module):
    def __init__(self, in_features,hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.norm1=nn.LayerNorm(hidden_features)
      
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.norm2=nn.LayerNorm(out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # print(x.shape)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)
  
        x = self.dwconv(x, H, W)
        
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


class FNORelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """
    def __init__(self, Ch, h,size):
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
        """
        super().__init__()
        # self.FNO=SpectralgeluConv2d(h*Ch,h*Ch,1,1)
        self.FNO=FFO(h*Ch,h*Ch,7,4,size)  #2.23
        self.sigmoid=nn.Sigmoid()
    def forward(self, q_img, v_img, size):
        B, h, N, Ch = q_img.shape
        H, W = size
        # Convolutional relative position encoding.
        v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)               # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        FNO_v_img=self.FNO(v_img)
        conv_v_img = rearrange(FNO_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)          # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        
        conv_v_img=self.sigmoid(conv_v_img)
        EV_hat_img = q_img * conv_v_img
        return EV_hat_img
class FactorAtt_ConvRelPosEnc(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """
    def __init__(self, dim,size, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., shared_crpe=None,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.FFO=FFO(dim,dim,7,4,size)
        self.qkv = LinearConv(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)                                           # Note: attn_drop is actually not used.
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

    def forward(self, x, size):
        B, N, C = x.shape
        H,W=size
        x=x.permute(0,2,1).view(B,C,H,W)
        # Generate Q, K, V.
        # Shape: [B, h, N, Ch]

        qkv=self.FFO(x)
        q,k,v=self.qkv(qkv).chunk(3,1)   # B,H,W,C
        q=q.view(B,self.num_heads, C // self.num_heads,H*W).permute(0,1,3,2)
        k=k.view(B,self.num_heads, C // self.num_heads,H*W).permute(0,1,3,2)
        v=v.view(B,self.num_heads, C // self.num_heads,H*W).permute(0,1,3,2)

        # Factorized attention.
        k_softmax = k.softmax(dim=2)                                                     # Softmax on dim N.
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)          # Shape: [B, h, Ch, Ch].
        factor_att        = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)  # Shape: [B, h, N, Ch].
        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)                                                # Shape: [B, h, N, Ch].

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)                                           # Shape: [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C].

        # Output projection.
        x = self.proj(x)
        x = self.proj_drop(x)

        return x  

class FoatAttention(nn.Module):
    """ Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module. """
    def __init__(self, dim, size,num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ):#add drop 0.2 3.30
        super().__init__()
        
        self.size=size
        self.crpe1=FNORelPosEnc(Ch=dim // num_heads, h=num_heads,size=self.size)
        self.norm1 = norm_layer(dim)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(
            dim, size=self.size,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            shared_crpe=self.crpe1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # MLP.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        B,C,H,W=x.shape
        size=(H,W)
        x=x.permute(0,2,3,1).view(B,H*W,C)
        
        x=self.norm1(x) #2.6 add
        
        cur = self.factoratt_crpe(x, size) 
        # Apply factorized attention and convolutional relative position encoding.
        x = x + self.drop_path(cur) 

        # MLP. 
        cur = self.norm2(x)
        cur = self.mlp(cur,H,W)
        x = x + self.drop_path(cur)
        x=x.permute(0,2,1).view(B,C,H,W)
        return x

if __name__ == '__main__':
    model=FoatAttention(512)
    inp=torch.randn(1,512,64,64)
    out=model(inp)
    print(out.shape)
    import numpy as np
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)