import os, torch, random

from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms

from .Utils import norml2
from .Reader import ImageReader
from .color_lib import RGBmean,RGBstdv

def eva(dsets, model):
    Fvecs = []
    dataLoader = torch.utils.data.DataLoader(dsets, batch_size=200, sampler=SequentialSampler(dsets), num_workers=24)

    torch.set_grad_enabled(False)
    for data in dataLoader:
        inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
        fvec = norml2(model(inputs_bt.to('cuda:0')))
        Fvecs.append(fvec.to('cpu'))
            
    return torch.cat(Fvecs,0)


def RunTest(Data, dst, data_dict, imgsize, L, phase):
    if os.path.isfile(dst + phase + 'dsets.pth'):
        dsets = torch.load(dst+ phase + 'dsets.pth')
        print('Loading dsets')
    else:
        data_transforms = transforms.Compose([transforms.Resize(imgsize),
                                              transforms.CenterCrop(imgsize),
                                              transforms.ToTensor(),
                                              transforms.Normalize(RGBmean[Data], RGBstdv[Data])])
        
        dsets = ImageReader(data_dict, data_transforms) 
        torch.save(dsets, dst + phase + 'dsets.pth')
        print('Creating dsets')
        
    for l in range(L[0], L[1]):
        print(l,end='\r')
        model = torch.load(dst + 'model_{:02}_{}.pth'.format(l,0)).train(False)
        Fvecs = eva(dsets, model)
        torch.save(Fvecs, dst + str(l) + phase + 'Fvecs.pth')

    
    
    
    