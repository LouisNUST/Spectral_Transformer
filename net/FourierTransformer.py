from json.tool import main
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from einops import rearrange

from FoatAttention import *
# from .FoatAttention import *

from FFO import *
# from .FFO import *

def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, 1, padding=0)
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor=2):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        x=x.permute(0,3,1,2)
        return x
class PatchExpanding(nn.Module):
    def __init__(self, in_channels, out_channels, upscaling_factor=2):
        super().__init__()
        self.upscaling_factor = upscaling_factor
        self.linear = nn.Linear(in_channels // (upscaling_factor ** 2), out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h * self.upscaling_factor, w * self.upscaling_factor
        new_c = int(c // (self.upscaling_factor ** 2))
        x = torch.reshape(x, (b, new_c, self.upscaling_factor, self.upscaling_factor, h, w))
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = torch.reshape(x, (b, new_c, new_h, new_w)).permute(0, 2, 3, 1)
        x = self.linear(x)
        x=x.permute(0,3,1,2)
        return x
class Fuse_parallel(nn.Module):
    def __init__(self,in_c,out_c,size):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.h,self.w=size
        #se
        self.fc1 = nn.Conv2d(in_c, in_c // 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_c // 4, in_c, kernel_size=1)
    
        #spatial
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, 7,padding=3,  bias=False)


        #Fourier spatial
        # self.fourier=nn.Parameter(torch.randn(self.h,self.w//2+1,2,dtype=torch.float32))
        # self.fourier_conv=nn.Conv2d(2,1,1,1)

        #conv
        self.conv=nn.Sequential(
            nn.Conv2d(in_c,in_c,1,1),
        )

        
        self.fuse=nn.Sequential(
            nn.Conv2d(in_c*3,out_c,1,1),
        )

        if in_c == out_c:
            self.need_skip = nn.Identity()
        else:
            self.need_skip = nn.Conv2d(in_c,out_c,1,1,bias=False)
    def forward(self,up_inp,down_inp):
        up_inp=up_inp
        down_inp=down_inp
        inp=up_inp*down_inp
        #se
        x_se = inp.mean((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.relu(x_se)
        x_se = self.fc2(x_se)
        x_se = self.sigmoid(x_se) * inp
        #spatial
        x_sp = inp
        x_sp = self.compress(x_sp)
        x_sp = self.spatial(x_sp)
        x_sp = self.sigmoid(x_sp) * inp

        #fourier spatial
        # x_sp = inp
        # x_sp = self.compress(x_sp)
        # x_sp=torch.fft.rfft2(x_sp,dim=(2,3),norm='ortho')
        # weight=torch.view_as_complex(self.fourier)
        # x_sp=x_sp*weight
        # x_sp=torch.fft.irfft2(x_sp,s=(self.h,self.w),dim=(2,3),norm='ortho')
        # x_sp=self.fourier_conv(x_sp)
        # x_sp = self.sigmoid(x_sp) * inp

        #conv
        x_conv=self.conv(inp)

        inp_fuse=torch.cat([x_conv,x_se,x_sp],dim=1)
        # inp_fuse=torch.cat([x_se,x_sp],dim=1)
        
        out=self.need_skip(inp)+self.fuse(inp_fuse)  #residual
        return out
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi
class Fuse_Serial(nn.Module):
    def __init__(self,in_c1,in_c2,out_c,attn=False):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv=nn.Sequential(
            nn.Conv2d(in_c1+in_c2,out_c,3,1,1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        if attn:
            self.attn_block = Attention_block(in_c1, in_c2, out_c)
        else:
            self.attn_block = nn.Identity()
    def forward(self,x1,x2,attn=False):
        x1=self.up(x1)
        if attn:
            x2=self.attn_block(x1,x2)
        x1=torch.cat([x1,x2],dim=1)
        x=self.conv(x1)
        return x
class Trans_EB(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        # self.conv = Bottleneck(in_, out)
        # self.conv=ConvRelu(in_,out)
        # self.conv=CoatAttention(in_)
        # self.conv=FFO(in_,out,7,5)
        self.conv=FoatAttention(in_)
        self.activation=torch.nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Down1(nn.Module):
    
    def __init__(self,size):
        super(Down1, self).__init__()
        self.nn1 = PatchMerging(3, 64)
        self.nn2 = FoatAttention(64,size)
        self.nn3 = FoatAttention(64,size)
    def forward(self, inputs):
        scale1_1 = self.nn1(inputs) 
        scale1_2 = self.nn2(scale1_1)
        scale1_3=self.nn3(scale1_2)
        return  scale1_3
class Down2(nn.Module):

    def __init__(self,size):
        super(Down2, self).__init__()
        self.nn1 = FoatAttention(128,size)
        self.nn2 = FoatAttention(128,size)
        self.embed=PatchMerging(64,128)

    def forward(self, inputs):
        emb=self.embed(inputs)
        scale2_1 = self.nn1(emb)
        scale2_2 = self.nn2(scale2_1)
        return scale2_2
class Down3(nn.Module):

    def __init__(self,size):
        super(Down3, self).__init__()

        self.nn1 = FoatAttention(256,size)
        self.nn2 = FoatAttention(256,size)
        self.nn3 = FoatAttention(256,size)
        self.embed = PatchMerging(128,256)

    def forward(self, inputs):
        emb=self.embed(inputs)
        scale3_1 = self.nn1(emb)
        scale3_2 = self.nn2(scale3_1)
        scale3_3 = self.nn2(scale3_2)
        
        return scale3_3
class Down4(nn.Module):

    def __init__(self,size):
        super(Down4, self).__init__()
        self.embed=PatchMerging(256,512)
        self.nn1 = FoatAttention(512,size)
        self.nn2 = FoatAttention(512,size)
        self.nn3 = FoatAttention(512,size)

    def forward(self, inputs):
        emb=self.embed(inputs)
        scale4_1 = self.nn1(emb)
        scale4_2 = self.nn2(scale4_1)
        scale4_3 = self.nn2(scale4_2)
        
        return scale4_3

class Up1(nn.Module):

    def __init__(self,size):
        super().__init__()
        self.nn1 = FoatAttention(64,size)
        self.emb = PatchExpanding(128,64)

    def forward(self, inputs):
        emb=self.emb(inputs)
        scale1_3 = self.nn1(emb)
        return scale1_3
class Up2(nn.Module):

    def __init__(self,size):
        super().__init__()
        self.nn1 = FoatAttention(128,size)
        self.emb=PatchExpanding(256,128)

    def forward(self, inputs):
        emb=self.emb(inputs)
        scale2_3 = self.nn1(emb)
        return  scale2_3
class Up3(nn.Module):

    def __init__(self,size):
        super().__init__()
        self.nn1 = FoatAttention(256,size)
        self.emb=PatchExpanding(512,256)

    def forward(self, inputs):
        emb=self.emb(inputs)
        scale3_4 = self.nn1(emb)
        return scale3_4
class Up4(nn.Module):

    def __init__(self,size):
        super().__init__()
        self.nn1 = FoatAttention(512,size)
    def forward(self, inputs ):
        scale4_4 = self.nn1(inputs)
        return  scale4_4

# @BACKBONES.register_module()
class FourierTransformer(nn.Module):

    def __init__(self,H,W):
        super(FourierTransformer, self).__init__()

        self.size=(H,W)
        self.down1 = Down1(H//2)   
        self.down2 = Down2(H//4)   
        self.down3 = Down3(H//8)   
        self.down4 = Down4(H//16)  


        self.up1 = Up1(H//2)       
        self.up2 = Up2(H//4)       
        self.up3 = Up3(H//8)        
        self.up4 = Up4(H//16)


        self.fuse5=Fuse_parallel(512,512,size=(self.size[0]//16,self.size[1]//16))
        self.fuse4=Fuse_parallel(512,256,size=(self.size[0]//8,self.size[1]//8))  
        self.fuse3=Fuse_parallel(256,128,size=(self.size[0]//4,self.size[1]//4))
        self.fuse2=Fuse_parallel(128,64,size=(self.size[0]//2,self.size[1]//2))   
        self.fuse1=Fuse_parallel(64,64,size=(self.size[0],self.size[1]))
        
        self.attn=True
        self.serial4=Fuse_Serial(512,256,256,self.attn)    
        self.serial3=Fuse_Serial(256,128,128,self.attn)    
        self.serial2=Fuse_Serial(128,64,64,self.attn)
        self.serial1=Fuse_Serial(64,64,64,self.attn)       


        self.head=Conv1X1(64,1)
        self.aux_head=Conv1X1(64,1)
    def forward(self, inputs):
      
        # encoder part
        down1 = self.down1(inputs)
        down2 = self.down2(down1)
        down3= self.down3(down2)
        down4= self.down4(down3)

        # decoder part
       
        up4 = self.up4(down4)
        up3 = self.up3(up4)
        up2 = self.up2(up3)
        up1 = self.up1(up2)
        # print(down1.shape,down2.shape,down3.shape,down4.shape)
        # print(up1.shape,up2.shape,up3.shape,up4.shape)

    
        fuse4=self.fuse4(down4,up4)
        fuse3=self.fuse3(down3,up3)
        fuse2=self.fuse2(down2,up2)
        fuse1=self.fuse1(down1,up1) 
      
        serial3=self.serial3(fuse4,fuse3,self.attn)
        serial2=self.serial2(serial3,fuse2,self.attn)
        serial1=self.serial1(serial2,fuse1,self.attn)
        serial1=F.upsample(serial1,scale_factor=2)
        up1=F.upsample(up1,scale_factor=2)
        out=self.head(serial1)
        aux_out=self.aux_head(up1)
        
        aux_out=aux_out.squeeze(1)
        out=out.squeeze(1)
        return aux_out,out

if __name__ == '__main__':
    model=FourierTransformer(512,512)
    inp=torch.randn(1,3,512,512)
    out=model(inp)
    print(out[0].shape,out[1].shape)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)