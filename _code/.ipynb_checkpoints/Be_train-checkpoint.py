from .color_lib import RGBmean,RGBstdv
from .Utils import createID
from .Train import learn

import os, torch, random

def RunTrain(Data, dst, data_dict, imagesize, L, Dim, bt=32, avgpool=8, lr=0.01, ep=10, core=[0]):
    if not os.path.exists(dst): os.makedirs(dst)

    # class number
    N = len(data_dict['tra'])
    print('# of classes: {}'.format(N))
    
    # ID matrix
    if os.path.isfile(dst+'ID.pth'):
        ID = torch.load(dst+'ID.pth')
        print('Loading ID')
    else:
        ID = createID(Dim,L[2],N)
        torch.save(ID, dst+'ID.pth')
        print('Creating ID')
        
    for dim in range(L[0], L[1]):
        print(dst,dim)
        learn(ID[:,dim].long(), 0, dim, dst, core, RGBmean[Data], RGBstdv[Data], data_dict, num_epochs=ep, init_lr=lr, decay=0.1, batch_size=bt, imgsize=imagesize, avg=avgpool).run()
        