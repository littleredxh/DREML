import torch

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
