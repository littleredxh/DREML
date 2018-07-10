import numpy as np
import os, time, copy, random
from glob import glob

from torchvision import models, transforms, datasets
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.sampler import SequentialSampler
import torch.optim as optim
import torch.nn as nn
import torch

from .Sampler import BalanceSampler
from .Reader import ImageReader
from .Utils import invDict
from .model.Net import resnetX
from scipy.stats import special_ortho_group

PHASE = ['tra','val']

class learn():
    def __init__(self,ID, no, idx, dst, gpuid, RGBmean, RGBstdv, data_dict, num_epochs=10, init_lr=0.01, decay=0.01, batch_size=400, imgsize=128, avg=8, num_workers=16):
        self.ID = ID
        self.no = no
        self.idx = idx
        self.dst = dst
        self.gpuid = gpuid #len(gpuid)=1: 
        
        if len(gpuid)>1: 
            self.mp = True
        else:
            self.mp = False
            
        self.batch_size = batch_size; print('batch size: {}'.format(batch_size))
        self.num_workers = num_workers; print('num workers: {}'.format(num_workers))
        
        self.decay_time = [False,False]
        self.init_lr = init_lr; print('init_lr : {}'.format(init_lr))
        self.decay_rate = decay
        self.num_epochs = num_epochs

        self.avg = avg
        self.data_dict_ori = data_dict
        
        self.imgsize = imgsize; print('image size: {}'.format(imgsize))
        self.RGBmean = RGBmean
        self.RGBstdv = RGBstdv
        
        self.record = []
        if not self.setsys(): print('system error'); return

    def run(self):
        self.loadData()
        self.setModel()
        self.criterion = nn.CrossEntropyLoss()
        self.opt(self.num_epochs)
        return self.best_acc
    

    ##################################################
    # step 0: System check
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        self.device = torch.device('cuda:0')
        return True
    
    ##################################################
    # step 1: Loading Data
    ##################################################
    def loadData(self):
        # truncate data
        TH = 100
        
        # sort classes and fix the class order  
        all_class = sorted([k for k in self.data_dict_ori['tra']], key = lambda x:x)

        # append image
        self.data_dict = {p:{i:[] for i in range(self.ID.max().item()+1)} for p in ['tra','val']}
        for i in range(len(all_class)):
            tra_imgs = self.data_dict_ori['tra'][all_class[i]]
            meta_class = self.ID[i].item()
            if len(tra_imgs)>TH:
                tra_imgs = random.sample(tra_imgs,TH)
                self.data_dict['tra'][meta_class]+=tra_imgs[:int(0.9*len(tra_imgs))]
                self.data_dict['val'][meta_class]+=tra_imgs[int(0.9*len(tra_imgs)):]
            else:
                self.data_dict['tra'][meta_class]+=tra_imgs[:int(0.9*len(tra_imgs))]
                self.data_dict['val'][meta_class]+=tra_imgs[int(0.9*len(tra_imgs)):]
                
        self.data_transforms = {'tra': transforms.Compose([
                                       transforms.Resize(int(self.imgsize*1.1)),
                                       transforms.RandomRotation(10),
                                       transforms.RandomCrop(self.imgsize),
                                       transforms.RandomHorizontalFlip(),
                                       # transforms.ColorJitter(0.1,0.1,0.1,0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize(self.RGBmean, self.RGBstdv)]),
                                'val': transforms.Compose([
                                       transforms.Resize(self.imgsize),
                                       transforms.CenterCrop(self.imgsize),
                                       transforms.ToTensor(),
                                       transforms.Normalize(self.RGBmean, self.RGBstdv)])}
        

        self.dsets = {p: ImageReader(self.data_dict[p], self.data_transforms[p]) for p in PHASE}
        print(len(self.dsets['tra']))
        self.classSize = len(self.data_dict['tra'])
        print('output size: {}'.format(self.classSize))

        return
    
    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self):
        print('Setting model')
#         self.model = resnetX(self.classSize, pretrained=True)
#         self.model.avgpool=nn.AvgPool2d(self.avg)
#         num_ftrs = self.model.fcnew.in_features
#         self.model.fcnew = nn.Linear(num_ftrs, self.classSize)
        
#         self.model = self.model.to('cuda:0')
#         self.optimizer = optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=0.9)
        
        self.model = models.resnet50(pretrained=True)
        self.model.avgpool=nn.AvgPool2d(self.avg)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.classSize)
        
        
#         # oth
#         x = torch.Tensor(special_ortho_group.rvs(num_ftrs))
#         self.model.fc.weight.data = x[:self.classSize,:]
#         self.model.fc.bias.data = torch.zeros(self.classSize)
        
#         for param in self.model.fc.parameters(): param.requires_grad = False
#         self.model = self.model.cuda()
#         Params = iter([p[1] for p in self.model.named_parameters() if p[0].split('.')[0]!='fc'])
#         self.optimizer = optim.SGD(Params, lr=self.init_lr, momentum=0.9)
        
        # parallel computing and opt setting
        if self.mp:
            print('Training on Multi-GPU')
            self.batch_size = self.batch_size*len(self.gpuid)
            self.model = torch.nn.DataParallel(self.model,device_ids=self.gpuid).to('cuda:0')#
            self.optimizer = optim.SGD(self.model.module.parameters(), lr=self.init_lr, momentum=0.9)
        else: 
            print('Training on Single-GPU')
            self.model = self.model.to('cuda:0')
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=0.9)
        return
    
    def lr_scheduler(self, epoch):
        if epoch>=0.6*self.num_epochs and not self.decay_time[0]: 
            self.decay_time[0] = True
            lr = self.init_lr*0.1#self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch>=0.8*self.num_epochs and not self.decay_time[1]: 
            self.decay_time[1] = True
            lr = self.init_lr*0.01#self.decay_rate*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return
            
    ##################################################
    # step 3: Learning
    ##################################################
    def tra(self):
        if self.mp:
            self.model.module.train(True)  # Set model to training mode
        else:
            self.model.train(True)  # Set model to training mode
            
        dataLoader = torch.utils.data.DataLoader(self.dsets['tra'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)#sampler=XSampler(self.classMark['tra'], GSize=1)
        
        L_data, T_data, N_data = 0.0, 0, 0
        # iterate batch
        
        for data in dataLoader:
            self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                fvec = self.model(inputs_bt.to('cuda:0'))
                loss = self.criterion(fvec, labels_bt.to('cuda:0'))

                loss.backward()
                self.optimizer.step()  

            _, preds_bt = torch.max(fvec.to('cpu'), 1)

            L_data += loss.item()
            T_data += torch.sum(preds_bt == labels_bt).item()
            N_data += len(labels_bt)
            
        return L_data/N_data, T_data/N_data 

    def val(self):
        if self.mp:
            self.model.module.eval()  # Set model to eval mode
        else:
            self.model.eval()  # Set model to eval mode
            
        dataLoader = torch.utils.data.DataLoader(self.dsets['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        L_data, T_data, N_data = 0.0, 0, 0
        # iterate batch
        with torch.set_grad_enabled(False):
            for data in dataLoader:
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                fvec = self.model(inputs_bt.to('cuda:0'))
                loss = self.criterion(fvec, labels_bt.to('cuda:0'))

                _, preds_bt = torch.max(fvec.to('cpu'), 1)

                L_data += loss.item()
                T_data += torch.sum(preds_bt == labels_bt).item()
                N_data += len(labels_bt)
            
        return L_data/N_data, T_data/N_data
        
    def opt(self, num_epochs):
        # recording time and epoch acc and best result
        since = time.time()
        self.best_epoch = 0
        self.best_acc = 0
        for epoch in range(num_epochs):
            print('Epoch {}/{} \n '.format(epoch, num_epochs - 1) + '-' * 40)
            self.lr_scheduler(epoch)
            
            tra_loss, tra_acc = self.tra()
            val_loss, val_acc = self.val()
            
            self.record.append((epoch, tra_loss, val_loss, tra_acc, val_acc))
            print('tra - Loss:{:.4f} - Acc:{:.4f}\nval - Loss:{:.4f} - Acc:{:.4f}'.format(tra_loss, tra_acc, val_loss, val_acc))    
    
            # deep copy the model
            if epoch >= 1 and val_acc> self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                torch.save(self.model, self.dst + 'model_{:02}_{}.pth'.format(self.idx,self.no))
        
        torch.save(torch.Tensor(self.record), self.dst + 'record_{:02}_{}.pth'.format(self.idx, self.no))
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        print('Best val acc in epoch: {}'.format(self.best_epoch))
        return
    
    

    