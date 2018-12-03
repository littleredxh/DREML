import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

from .Utils import norml2

class ProxyStatic(Module):
    def __init__(self, D_embed, N):
        """one proxy per class"""
        super(ProxyStatic, self).__init__()
        # orginal proxy
        # self.proxy = norml2(torch.randn(N,N)).cuda()
        
        # proxy in DREML paper
        self.proxy = torch.eye(N).cuda()
        
    def forward(self, fvec, fLvec):
        N = fLvec.size(0)
        
        # normalization(original proxy method)
        # fvec = norml2(fvec)

        # distance matrix
        Dist = fvec.mm((self.proxy).t())
        
        # loss
        Dist = -F.log_softmax(Dist, dim=1)
        loss = Dist[torch.arange(N),fLvec].mean()
        print('loss:{:.4f}'.format(loss.item()),end='\r')
        
        return loss




    
    
    
