
from json.tool import main
from random import random

from .ops import blur,fliph,flipv,noise,rotate
import cv2
from skimage import transform
import torch
import numpy as np


class DataAug():
    def __init__(self):
        self.process_blur=blur.Blur(2.0)
        self.process_FlipH=fliph.FlipH()
        self.process_FlipV=flipv.FlipV()
        self.process_Noise=noise.Noise(0.02)
        self.process_Rotate=rotate.Rotate(-90)
    def preprocess(self,img,gt=None):
        # if random()<0.1:
        #     img=self.process_blur.process(img)
        # if random()<0.1:
        #     img=self.process_Noise.process(img)
        # if random()>0.5:
        #     img=self.process_Rotate.process(img)
        #     gt=self.process_Rotate.process(gt)
            
            
        if random()<0.5:
            img=self.process_FlipH.process(img)
            gt=self.process_FlipH.process(gt)
        if random()<0.5:
            img=self.process_FlipV.process(img)
            gt=self.process_FlipV.process(gt)
            
            
        # if img[img>1].any()==True:
        #     return img,gt
        # else:
        #     return img*255,gt
        gt[gt==1]=255
        gt=np.array(gt,dtype=np.uint8)
        # print(len(gt[gt>127]))
        # print(np.max(gt))
        img=np.ascontiguousarray(img)
        gt=np.ascontiguousarray(gt)
        return img,gt
if __name__ == '__main__':
    aug=DataAug()
    for i,each in enumerate(range(100)):
        img=cv2.imread('/home/nlg/yj/fouriernet/datasets/CrackTree/valid/Lable_image/6340.bmp')
        img=torch.from_numpy(img)
        print(img.shape)
        img=aug.preprocess(img,img)
        print(type(img))
        # img=np.array(img)
        # cv2.imwrite('./img/{}.png'.format(i),img)
        print(img[0].shape)