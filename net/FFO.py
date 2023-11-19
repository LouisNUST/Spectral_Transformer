#Fourier Filter Operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class FFOConv2d(nn.Module):
    def __init__(self, in_channels,out_channels, modes1, modes2,size):
        super(FFOConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT. /   
        """

        self.in_channels = in_channels
        self.out_channels=out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels ))
        self.size=size
        self.full=False if self.size>16 else True  

         # 3d 平均池化
        self.ap = torch.nn.AvgPool3d(kernel_size=(1, 32, 32))
        # Fc
        self.fc = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        # sigmod
        self.act = nn.Sigmoid()




        if self.full==False:
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, self.modes1, self.modes2, dtype=torch.cfloat))

            # self.linear1_real=nn.Conv2d(self.in_channels,self.out_channels,1)    #share linear weight
            # self.linear1_imag=nn.Conv2d(self.in_channels,self.out_channels,1)
            self.linear1=nn.Parameter(torch.randn(in_channels,out_channels,dtype=torch.cfloat))
            self.bias1=nn.Parameter(torch.randn(1,out_channels,1,1,dtype=torch.cfloat))
            self.linear2=nn.Parameter(torch.randn(in_channels,out_channels,dtype=torch.cfloat))
            self.bias2=nn.Parameter(torch.randn(1,out_channels,1,1,dtype=torch.cfloat))
            # self.linear2_real=nn.Conv2d(self.in_channels,self.out_channels,1)
            # self.linear2_imag=nn.Conv2d(self.in_channels,self.out_channels,1)
        else:
            self.weights_FULL = nn.Parameter(self.scale * torch.rand(in_channels, self.size, self.size//2+1, dtype=torch.cfloat))
            # self.linear_FULL_real=nn.Conv2d(self.in_channels,self.out_channels,1)
            # self.linear_FULL_imag=nn.Conv2d(self.in_channels,self.out_channels,1)
            self.linear=nn.Parameter(torch.randn(in_channels,out_channels,dtype=torch.cfloat))
            self.bias=nn.Parameter(torch.randn(1,out_channels,1,1,dtype=torch.cfloat))
    # Complex multiplication
    def compl_mul2d(self, input, weights,pos):
        # (batch, in_channel, x,y ), (in_channel, x,y) -> (batch, channel, x,y)
        # print('input',input.shape,'weight',weights.shape,type(weights))
        # weights=torch.view_as_complex(weights)
        output=torch.einsum("bcxy,cxy->bcxy", input, weights)
        if self.full==True:
            # output.real=self.linear_FULL_real(output.real)
            # output.imag=self.linear_FULL_imag(output.imag)
            output=torch.einsum("bchw,cd->bdhw",output,self.linear)
            output=output+self.bias
            return output
        if pos=='up':
            # output.real=self.linear1_real(output.real)
            # output.imag=self.linear1_imag(output.imag)
            output=torch.einsum("bchw,cd->bdhw",output,self.linear1)
            output=output+self.bias1
        if pos=='down':
            # output.real=self.linear1_real(output.real)
            # output.imag=self.linear1_imag(output.imag)
            output=torch.einsum("bchw,cd->bdhw",output,self.linear2)
            output=output+self.bias2
        return output

    def forward(self, x):
        batchsize = x.shape[0]
        b,c,h,w = x.shape
        if x.shape[-1]!=self.size and x.shape[-2]!=self.size:
            raise Exception('{} {}size do not match'.format(self.size,x.shape[-1]))
        print(f'x shape {x.shape}')
         #TODO 3d平均池化 b c h w 
        x_sp = self.ap(x) #  
        print(f'x_sp shape {x_sp.shape}')
        x_sp = self.fc(x_sp) #  
        print(f'x_sp shape {x_sp.shape}')


        x_low = self.act(x_sp) # 得到低频阈值
        print(f'x_low shape {x_low.shape}')

        # 使用 interpolate 函数进行线性插值
        x_low = F.interpolate(x_low, size=(h, w), mode='bilinear', align_corners=False)

        # 保留小于阈值的
        x = torch.where(x > x_low, torch.tensor(0.0), x)
        print(f'x shape {x.shape}')

        #Compute Fourier coeffcients up to factor of e^(- something constant)

        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.in_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        if self.full==True:
            out_ft[:,:,:,:]=self.compl_mul2d(x_ft[:, :, :, :], self.weights_FULL,'full')
        else:
            out_ft[:, :, :self.modes1, :self.modes2] = \
                self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1,'up')
            out_ft[:, :, -self.modes1:, :self.modes2] = \
                self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2,'down')
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        # print(x.shape)
        return x

class FFO(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2,size):
        super(FFO, self).__init__()
        self.conv=FFOConv2d(in_channels, out_channels, modes1, modes2,size)
        self.w=nn.Conv2d(in_channels,out_channels,1)
    def forward(self,x):
        x1=self.conv(x)
        x2=self.w(x)
        x=x1+x2
        x=F.gelu(x)
        return x

def measure_param(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
if __name__ == '__main__':
  
    model=FFO(512,512,7,5,256)
    inp=torch.randn(1,512,256,256)
    out=model(inp)
    measure_param(model)
    model=nn.Conv2d(512,512,3,1)
    inp=torch.randn(1,512,256,256)
    out=model(inp)
    measure_param(model)
    