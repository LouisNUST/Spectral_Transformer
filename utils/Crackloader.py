import os
import os.path
import torch
from torchvision import transforms
import numpy as np
import scipy.misc as m
import glob
import torch.utils.data as data
import cv2
from torch.utils import data
from .aug.process import DataAug

class Crackloader(data.Dataset):

    def __init__(self, txt_path,normalize=True):
        self.txt_path = txt_path

        if normalize:
            self.img_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.img_transforms = transforms.ToTensor()

        self.train_set_path = self.make_dataset(txt_path)
        self.Aug=DataAug()
    def __len__(self):
        return len(self.train_set_path)

    def __getitem__(self, index):
        img_path, lbl_path = self.train_set_path[index]
        img = cv2.imread(img_path)     
        H,W,_=img.shape
        if H==448 and W==448:
            img=cv2.resize(img,(512,512),cv2.INTER_NEAREST )
        elif H==600 and W==800:
            img=img[:592,::,::]
        elif H==720 and W==960:
            img=img[:592,:800,::]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.uint8)
        lbl = cv2.imread(lbl_path,0) ##self.root 
        if H==448 and W==448:
            lbl=cv2.resize(lbl,(512,512),cv2.INTER_NEAREST)  
        elif H==600 and W==800:
            lbl=lbl[:592,::]
        elif H==720 and W==960:
            lbl=lbl[:592,:800]
        # img,lbl=self.Aug.preprocess(img,lbl)
        img = self.img_transforms(img)
        img=img.type(torch.FloatTensor)
        _, binary = cv2.threshold(lbl,127, 1, cv2.THRESH_BINARY)
        return img, binary

    def make_dataset(self, txt_path):
        dataset = []
        index=0
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                # print(index,line)
                index+=1
                line = ''.join(line).strip()
                line_list = line.split(' ')
                dataset.append([line_list[0], line_list[1]])
        return dataset




