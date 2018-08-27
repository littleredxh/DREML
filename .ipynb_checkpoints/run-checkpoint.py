from _code.Be_train import RunTrain
from _code.Be_test import RunTest
import os, torch
from glob import glob

Data= 'CAR'
dst = '_result/'

Len = 12 # number of ensemble
N_M = 12 # number of meta-classes

img_size = 256

## train
data_dict_tra = torch.load('*.pth')
RunTrain(Data, dst, data_dict_tra, img_size, [0,Len,Len], N_M, bt=128, lr=0.01, ep=12)

## test
data_dict_test = torch.load('*.pth')
RunTest(Data,  dst, data_dict_test, img_size, [0,Len,Len], 'test')
