from _code.Be_train import RunTrain
from _code.Be_test import RunTest
from _code.Be_test2 import RunTest2
from glob import glob
import os, torch, random

Data= 'CAR'
dst = '_result/'

Len = 24
Dim = 12

img_size = 256

## train
data_dict_tra = torch.load('*.pth')
RunTrain(Data, dst, data_dict_tra, img_size, [0,Len,Len], Dim, bt=128, lr=0.01, ep=10)

### test
data_dict_test = torch.load('*.pth')
RunTest(Data,  dst, data_dict_test, img_size, [0,Len,Len], 'test')
