import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

class ProxyStaticLoss(Module):
    def __init__(self, embed_size, proxy_num):
        """one proxy per class"""
        super(ProxyStaticLoss, self).__init__()
        self.proxy = torch.eye(proxy_num).cuda()
        
    def forward(self, fvec, fLvec):
        N = fLvec.size(0)

        # distance matrix
        Dist = fvec.mm((self.proxy).t())
        pred = Dist.max(1)[1].cpu()
        
        # loss
        Dist = -F.log_softmax(Dist, dim=1)
        loss = Dist[torch.arange(N),fLvec].mean()
        print('loss:{:.4f}'.format(loss.item()),end='\r')
        
        return loss, pred




    
    
    
