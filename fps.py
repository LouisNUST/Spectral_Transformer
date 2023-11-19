import time

import torch
from nets.crackformer2 import crackformer
from nets.Deepcrack import DeepCrack
from nets.HED import HED
from nets.SDDNet import SDDNet
from nets.RCF import RCF
from nets.STRNet import STRNet
from nets.zjd_unet_2plus_ori import * #要改
#from nets.zjd_unet_2plus import *
from nets.Unet_delinear import *
from nets.Unet_delinear_less import *






input=torch.randn(1,3,512,512).cuda()
# input=torch.randn(1,3,544,384).cuda()
#input=torch.randn(1,3,448,448).cuda()
#input=torch.randn(1,3,480,480).cuda()
#nput=torch.randn(1,3,480,320).cuda()
# model=DeepCrack().cuda()
# model=HED().cuda()
# model=SDDNet(3,1).cuda()
#model=STRNet(3,1).cuda()
model=zjd_unet_2plus().cuda()
# model =linDeform_Unetless().cuda()
# model=RCF().cuda()
# model=crackformer().cuda()

# total_time=0
# start=time.time()
# end=time.time()
# single_fps=1/(end-start)
# total_time+=end-start
# # fps=(i+1)/total_time
# out=model(input)
# print(single_fps,total_time)


model.eval()
torch.cuda.synchronize()
start=time.time()
for index,_ in enumerate(range(600)):
    out=model(input)
    end=time.time()
    # if index%100==0:
    #     print(index/(end-start))
torch.cuda.synchronize()
end=time.time()
print(1/((end-start)/600))

# model.eval() # 进入eval模式（即关闭掉droout方法
# total_time = 0
# device="cuda"
# with torch.no_grad():
#     # predict class
#     input = input.to(device)
#     torch.cuda.synchronize()
#     time_start = time.time()
#     output = model(input.to(device)) # 将图片通过model正向传播，得到输出，将输入进行压缩，
#                                                         # 将batch维度压缩掉，得到最终输出（out）
#     torch.cuda.synchronize()
#     time_end = time.time()
#     # predict = torch.softmax(output, dim=0)              # 经过softmax处理后，就变成概率分布的形式了
#     # predict_cla = torch.argmax(predict).numpy()         # 通过argmax方法，得到概率最大的处所对应的索引
#     single_fps = 1 / (time_end - time_start)
#     time_sum = (time_end - time_start) * 1000
# print_res = "time: {: .3f}ms  single_fps: {: .3f}".format(time_sum,single_fps)
# print(print_res)

# predict[predict_cla] 打印类别名称以及他所对应的预测概率


