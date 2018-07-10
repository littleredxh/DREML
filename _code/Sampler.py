from torch.utils.data.sampler import Sampler
import numpy as np
import random


class BalanceSampler(Sampler):
    def __init__(self, intervals, GSize=2):
        
        class_len = len(intervals)
        list_sp = []
        
        # find the max interval
        interval_list = [np.arange(b[0],b[1]) for b in intervals]
        len_max = max([b[1]-b[0] for b in intervals])

        # exact division
        if len_max%GSize != 0:
            len_max = len_max+(GSize-len_max%GSize)
        
        for l in interval_list:
            if l.shape[0]<len_max:
                l_ext = np.random.choice(l,len_max-l.shape[0])
                l_ext = np.concatenate((l, l_ext), axis=0)
                l_ext = np.random.permutation(l_ext)
            elif l.shape[0]>len_max:
                l_ext = np.random.choice(l,len_max,replace=False)
                l_ext = np.random.permutation(l_ext)
            elif l.shape[0]==len_max:
                l_ext = np.random.permutation(l)
                
            list_sp.append(l_ext)

        self.idx = np.vstack(list_sp).reshape((GSize*class_len,-1)).T.reshape((1,-1)).flatten().tolist()

    def __iter__(self):
        return iter(self.idx)
    
    def __len__(self):
        return len(self.idx)
    
    