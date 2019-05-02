import random, torch

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
        
    return ID.long()

def recall(Fvec, imgLab,rank=None):
    # Fvec: torch.Tensor. N by dim feature vector
    # imgLab: a list. N related labels list
    # rank: a list. input k(R@k) you want to calcualte 
    N = len(imgLab)
    imgLab = torch.LongTensor([imgLab[i] for i in range(len(imgLab))])
    
    D = Fvec.mm(torch.t(Fvec))
    D[torch.eye(len(imgLab)).byte()] = -1
    
    if rank==None:
        _,idx = D.sort(1, descending=True)
        imgPre = imgLab[idx[:,0]]
        A = (imgPre==imgLab).float()
        return (torch.sum(A)/N).item()
    else:
        _,idx = D.topk(rank[-1])
        acc_list = []
        for r in rank:
            A = 0
            for i in range(r):
                imgPre = imgLab[idx[:,i]]
                A += (imgPre==imgLab).float()
            acc_list.append((torch.sum((A>0).float())/N).item())
        return torch.Tensor(acc_list)

def recall2(Fvec_val, Fvec_gal, imgLab_val, imgLab_gal,rank=None):
    N = len(imgLab_val)
    imgLab_val = torch.LongTensor([imgLab_val[i] for i in range(len(imgLab_val))])
    imgLab_gal = torch.LongTensor([imgLab_gal[i] for i in range(len(imgLab_gal))])
    
    D = Fvec_val.mm(torch.t(Fvec_gal))
    
    if rank==None:
        _,idx = D.sort(1, descending=True)
        imgPre = imgLab_gal[idx[:,0]]
        A = (imgPre==imgLab_val).float()
        return (torch.sum(A)/N).item()
    else:
        _,idx = D.topk(rank[-1])
        acc_list = []
        for r in rank:
            A = 0
            for i in range(r):
                imgPre = imgLab_gal[idx[:,i]]
                A += (imgPre==imgLab_val).float()
            acc_list.append((torch.sum((A>0).float())/N).item())
        return acc_list