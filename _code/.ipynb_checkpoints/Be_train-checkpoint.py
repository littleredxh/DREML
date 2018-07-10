from .color_lib import RGBmean,RGBstdv
from .Utils import createID, createID2
from .Train import learn

import os, torch, random

# L = [start, end, Len]

def RunTrain(Data, dst, data_dict, imagesize, L, Dim, bt=32, avgpool=8, lr=0.01, ep=12, core=[0,1]):
    if not os.path.exists(dst): os.makedirs(dst)

    N = len(data_dict['tra'])
    print('# of classes: {}'.format(N))
    
    # ID matrix
    if os.path.isfile(dst+'ID.pth'):
        ID = torch.load(dst+'ID.pth')
        print('Loading ID')
    else:
        ID = createID2(Dim,L[2],N)
        torch.save(ID, dst+'ID.pth')
        print('Creating ID')
    best = []
    for dim in range(L[0], L[1]):
        print(dst,dim)
        acc=torch.zeros(2)
        for t in range(1):
            acc[t] = learn(ID[:,dim].long(), t, dim, dst, core, RGBmean[Data], RGBstdv[Data], data_dict, num_epochs=ep, init_lr=lr, decay=0.3, batch_size=bt, imgsize=imagesize, avg=avgpool).run()
        best.append(acc.max(0)[1])
    print(best)
    torch.save(best,'best.pth')
