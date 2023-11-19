from utils.utils import *
from utils.Validator import *
from utils.Crackloader import *
from nets.FourierTransformer import FourierTransformer
import os
netName = "Fourier_aug2"
valid_log_dir = "./log/" + netName
best_model_dir = "./model/" + netName + "/"

# 数据集格式替换
image_format = "jpg"
# lable_format = "bmp"
lable_format="png"
# image_format = "jpg"
# lable_format = "jpg"

#数据集替换
datasetName = "CrackLS315"
# datasetName="CrackTree"
# datasetName='Crack537'
# datasetName='CFD'


valid_img_dir = "./datasets/" + datasetName + "/valid/Valid_image/"
valid_lab_dir = "./datasets/" + datasetName + "/valid/Lable_image/"
# valid_img_dir = "./datasets/new_dataset/test/images/"+datasetName+'/'
# valid_lab_dir = "./datasets/new_dataset/test/masks/"+datasetName+'/'
if os.path.exists(valid_img_dir)==False:
    os.makedirs(valid_img_dir)
if os.path.exists(valid_lab_dir)==False:
    os.makedirs(valid_lab_dir)

# 权重加载
pretrain_dir='model/CrackLS315/FourierTransformer/..pth'


valid_result_dir = "./datasets/"+netName+"/"+datasetName+"/Valid_result/"
def Test():
    crack=FourierTransformer().cuda()
    crack.load_state_dict(torch.load(pretrain_dir))
    validator = Validator(valid_img_dir, valid_lab_dir,
                          valid_result_dir, valid_log_dir, best_model_dir, crack, image_format, lable_format,datasetName)
    validator.validate('test_7_8')

if __name__ == '__main__':
    Test()