import random, torch

def norml2(vec):# input N by F
    F = vec.size(1)
    w = torch.sqrt((torch.t(vec.pow(2).sum(1).repeat(F,1))))
    return vec.div(w)

def createID(num_int,Len,N):
    """uniformly distributed"""
    multiple = N//num_int
    remain = N%num_int
    if remain!=0: multiple+=1
        
    ID = torch.zeros(N,Len)
    for i in range(Len):
        idx_all = []
        for _ in range(multiple):
            idx_base = [j for j in range(num_int)]
            random.shuffle(idx_base)
            idx_all+=idx_base

        idx_all = idx_all[:N]
        random.shuffle(idx_all)
        ID[:,i] = torch.Tensor(idx_all)
        
    return ID