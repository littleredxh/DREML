import random, torch, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image

######################################################################
# invert a dictionary: value to key
# ^^^^^^^^^^^^^^^^^^^^^^^
def invDict(dict_in):
    """input type dict"""
    values = sorted(set([v for k,v in dict_in.items()]))
    dict_out = {v:[] for v in values}
    for k,v in dict_in.items(): dict_out[v].append(k)
    return dict_out

def norml2(vec):# input N by F
    F = vec.size(1)
    w = torch.sqrt((torch.t(vec.pow(2).sum(1).repeat(F,1))))
    return vec.div(w)

def createID(num_int,num_rep,N):
    idx_base = []
    for j in range(num_int):
        for i in range(num_rep):
            idx_base.append(j)

    ID = torch.zeros(N,num_int*num_rep)
    for i in range(N):
        random.shuffle(idx_base)
        ID[i,:] = torch.Tensor(idx_base)
    return ID

def createID2(num_int,Len,N):
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

def matrixPlot(mat,dst,figname):
    plt.figure()
    img = plt.imshow(mat, cmap='hot', vmin=mat.min(), vmax=mat.max(),aspect='auto')
    plt.colorbar()
    # plt.xticks(np.arange(0, mat.shape[0], 1))
    # plt.xlabel('pre')
    # plt.yticks(np.arange(0, mat.shape[0], 1))
    # plt.ylabel('lab')
    # plt.axes().xaxis.set_ticks_position('top')
    plt.title(figname,y=1.05)
    if not os.path.exists(dst): os.makedirs(dst)
    plt.savefig(dst+figname,dpi=600)
    plt.close("all") 