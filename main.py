import os, torch, random
import argparse
from _code.Utils import createID
from _code.Train import learn 


parser = argparse.ArgumentParser(description='running parameters')
parser.add_argument('--Data', type=str, help='dataset name: CUB, CAR, SOP or ICR')
parser.add_argument('--Esize', type=int, help='ensemble_size')
parser.add_argument('--Msize', type=int, help='meta_class_size')
parser.add_argument('--epochs', type=int, help='epochs')
args = parser.parse_args()

## ensemble setting
ensemble_size = args.Esize # size of ensemble
meta_class_size = args.Msize # size of meta-classes

## dataset
Data = args.Data
data_dict = torch.load('/home/xuanhong/datasets/{}/data_dict_emb.pth'.format(Data))
dst = '_result/E{}_M{}/'.format(ensemble_size, meta_class_size)
if not os.path.exists(dst): os.makedirs(dst)
    
## ID matrix
print('Creating ID')
ID = createID(meta_class_size, ensemble_size, len(data_dict['tra']))
torch.save(ID, dst+'ID.pth')

## train
print('Training Ensemble model')
x = learn(Data, ID, dst, data_dict, num_epochs=12, batch_size=128)
x.run()


