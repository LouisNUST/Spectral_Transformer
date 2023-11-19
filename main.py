# import imp
import shutil
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable

from nets.FourierTransformer import *
from utils.utils import *
from utils.Validator import *
from utils.Crackloader import *
from utils.lossFunctions import *


import time
import datetime


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
IoUloss=IoULoss()


def calculate_loss(outputs, labels):
    loss = 0
    loss = cross_entropy_loss_RCF(outputs, labels)
    return loss


def calculate_IoU_loss(outputs, labels):
    loss = 0
    loss = IoUloss(outputs, labels)
    return loss


def trainer(timeStr,net, total_epoch, lr_init, batch_size,train_img_dir, valid_img_dir, valid_lab_dir,
            valid_result_dir, valid_log_dir, best_model_dir, image_format, lable_format, datasetName,pretrain_dir=None):


    # 输入训练图片,制作成dataset,dataset其实就是一个List 结构如同：[[img1,gt1],[img2,gt2],[img3,gt3]]

    img_data = Crackloader(txt_path=train_img_dir, normalize=False)

    #  加载数据 用pytorch给我们设计的加载器，可以自动打乱
    img_batch = data.DataLoader(img_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # 这是根据机器有几块GPU进行选择
    # if torch.cuda.device_count() > 1:
    #     crack = nn.DataParallel(net).cuda()
    # else:
    crack = net.cuda()  # 如果没有多块GPU需要用这种方法

    # 可以加载训练好的模型
    if pretrain_dir is not None:
        crack = torch.load(pretrain_dir).cuda()

    # 生成验证器
    validator = Validator(timeStr,valid_img_dir, valid_lab_dir,
                          valid_result_dir, valid_log_dir, best_model_dir, crack,  image_format, lable_format,datasetName)
    # 训练的核心部分
    log_dir=valid_log_dir+'/'+datasetName+'_'+timeStr+'_loss.txt'
    
    net_str=str(net)
    f=open(log_dir,'a')
    f.writelines(net_str)
    f.close()
    for epoch in range(1, total_epoch):

        losses = Averagvalue()
        crack.train()  # 选择训练状态
        count = 0 # 记录在当前epoch下第几个item，即当前epoch下第几次更新参数
        # 关于学习率更新的办法
        new_lr = updateLR(lr_init, epoch, total_epoch)
        print('Learning Rate: {:.9f}'.format(new_lr))

        # optimizer
        lr = new_lr
        optimizer = torch.optim.Adam(crack.parameters(), lr=lr)

        for (images, labels) in img_batch:
            f=open(log_dir,'a')
            count += 1
            loss = 0

            # 前向传播部分
            images = Variable(images).cuda()
            labels = Variable (labels.float()).cuda()
            # labels=labels.unsqueeze(1) 
            output = crack.forward(images)        
            for out in output:
                loss += 0.5 * calculate_loss(out, labels)
            loss += 1.1 * calculate_loss(output[-1], labels)

            #计算损失部分
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #打印输出损失，lr等
            losses.update(loss.item(), images.size(0))
            lr = optimizer.param_groups[0]['lr']
            if count%20==0:
                info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, total_epoch, count, len(img_batch)) + \
                       'Loss {loss.val:f} (avg:{loss.avg:f} lr {lr:.10f}) '.format(
                           loss=losses, lr=lr)
                print(info)
                f.writelines(time.strftime('%m-%d %X')+" "+info+'\n')
            f.close()
      
        if epoch % 4 == 0: ## 注意
            print("test.txt valid")
            validator.validate(epoch)


def check_dir(path):
    if os.path.exists(path)==False:
        os.makedirs(path)


if __name__ == '__main__':
    
    timeStr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # datasetName="cracktree"
    # datasetName="CrackTree600_800"
    # datasetName="CrackLS315"
    # datasetName="Stone"
    # datasetName="DeepCrack"
    # datasetName="CRACK500"
    # datasetName="500"
    datasetName="CFD" # 数据集替换
    # datasetName="CrackBenchmark"
    # dataName= "train"
   
    netName='FourierTransformer' 
    image_format = "jpg"
    lable_format='jpg'
    total_epoch = 500
    lr_init=0.001 
    batch_size = 2 
    net=FourierTransformer(512,512)
    
    # 训练数据集标识文件加载路径
    train_img_dir = "./datasets/new_dataset/train/"+ datasetName +".txt"   
    valid_img_dir = "./datasets/new_dataset/test/images/"+datasetName+'/'
    valid_lab_dir = "./datasets/new_dataset/test/masks/"+datasetName+'/'
    
    # 模型预测结果存储路径
    valid_result_dir = "./valid_result/"+netName+"/"+datasetName+timeStr+"/Valid_result/"
    valid_log_dir = "./log/" + netName 
    best_model_dir = "./model/" + datasetName +"/"+netName+'/'
    model_str=str(net)
    print(model_str)
    
    check_dir(valid_log_dir)
    check_dir(valid_result_dir)
    # pretrain_dir=""
    trainer(timeStr,net, total_epoch, lr_init, batch_size, train_img_dir, valid_img_dir, valid_lab_dir,
            valid_result_dir, valid_log_dir, best_model_dir,  image_format, lable_format,datasetName) #, pretrain_dir=pretrain_dir

    print("训练结束")

