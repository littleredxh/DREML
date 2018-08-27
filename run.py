from _code.Be_train import RunTrain
from _code.Be_test import RunTest
import os, torch
from glob import glob

Data= 'CAR'
dst = '_result/'

Len = 12
N_M = 12

img_size = 256

## train
src='/pless_nfs/home/datasets/CAR/data_e/'
data_dict = {p:{os.path.basename(d):glob(d+'/*.jpg') for d in glob(src+p+'/*')} for p in ['tra', 'val']}

# data_dict_tra = torch.load('*.pth')
data_dict_tra = data_dict['tra']
RunTrain(Data, dst, data_dict_tra, img_size, [0,Len,Len], N_M, bt=128, lr=0.01, ep=12)

## test
# data_dict_test = torch.load('*.pth')
data_dict_tra = data_dict['val']
RunTest(Data,  dst, data_dict_test, img_size, [0,Len,Len], 'test')
