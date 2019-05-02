import os, torch, random
from _code.Utils import createID
from _code.Train import learn 

ensemble_size = 12 # size of ensemble
meta_class_size = 12 # size of meta-classes

## train
Data = 'CAR'
data_dict = torch.load('/pless_nfs/home/datasets/CAR/data_dict_emb.pth')
dst = '_result/'

# ID matrix
print('Creating ID')
ID = createID(meta_class_size, ensemble_size, len(data_dict['tra']))

x = learn(Data, ID, dst, data_dict, num_epochs=12, batch_size=128)
x.run()
torch.save(ID, dst+'ID.pth')
