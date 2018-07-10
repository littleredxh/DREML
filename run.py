from _code.Be_train import RunTrain
from _code.Be_test import RunTest
import os, torch

Data= 'CAR'
dst = '_result/'

Len = 12
Dim = 12

img_size = 256

## train
data_dict_tra = torch.load('*.pth')
RunTrain(Data, dst, data_dict_tra, img_size, [0,Len,Len], Dim, bt=128, lr=0.01, ep=10)

### test
data_dict_test = torch.load('*.pth')
RunTest(Data,  dst, data_dict_test, img_size, [0,Len,Len], 'test')
