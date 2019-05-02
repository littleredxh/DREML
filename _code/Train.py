import os, time, random, copy

from torchvision import models, transforms, datasets
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

from .Loss import ProxyStaticLoss
from .Reader import ImageReader
from .color_lib import RGBmean,RGBstdv

class learn():
    def __init__(self, Data, ID, dst, data_dict, num_epochs=10, batch_size=128):
        self.Data = Data
        self.ID = ID
        self.dst = dst
        
        self.data_dict_tra = data_dict['tra']
        self.data_dict_val = data_dict['test']
        
        self.batch_size = batch_size; print('batch size: {}'.format(self.batch_size))
        self.num_workers = 8; print('num workers: {}'.format(self.num_workers))
        
        self.init_lr = 0.01; print('init_lr : {}'.format(self.init_lr))
        self.decay_rate = 0.01
        self.num_epochs = num_epochs

        self.imgsize = 256; print('image size: {}'.format(self.imgsize))
        self.RGBmean = RGBmean[Data]
        self.RGBstdv = RGBstdv[Data]
        
        # sort classes and fix the class order  
        all_class = sorted(self.data_dict_tra)
        self.idx_to_ori_class = {i:all_class[i] for i in range(len(all_class))}
        if not self.setsys(): print('system error'); return

    def run(self):
        for i in range(self.ID.size(1)):
            print('Training ensemble #{}'.format(i))
            self.l = i # index of the ensembles
            self.meta_id = self.ID[:,i].tolist()
            self.decay_time = [False,False]
            self.loadData()
            self.setModel()
            self.criterion = ProxyStaticLoss(self.classSize, self.classSize)
            best_model = self.opt(self.num_epochs)
            self.eva(best_model)
        return
    

    ##################################################
    # step 0: System check
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        return True
    
    ##################################################
    # step 1: Loading Data
    ##################################################
    def loadData(self):
        # balance data for each class
        TH = 300
        
        # append image
        self.data_dict_meta = {i:[] for i in range(max(self.meta_id)+1)}
        for i,c in self.idx_to_ori_class.items():
            meta_class_id = self.meta_id[i]
            tra_imgs = self.data_dict_tra[c]
            if len(tra_imgs)>TH: tra_imgs = random.sample(tra_imgs,TH)
            self.data_dict_meta[meta_class_id]+=tra_imgs
        
        self.data_transforms_tra = transforms.Compose([transforms.Resize(int(self.imgsize*1.1)),
                                                       transforms.RandomCrop(self.imgsize),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(self.RGBmean, self.RGBstdv)])
        
        self.data_transforms_val = transforms.Compose([transforms.Resize(self.imgsize),
                                                       transforms.CenterCrop(self.imgsize),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(self.RGBmean, self.RGBstdv)])
        
        self.classSize = len(self.data_dict_meta)
        print('output size: {}'.format(self.classSize))

        return
    
    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self):
        print('Setting model')
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.classSize)
        self.model.avgpool=nn.AdaptiveAvgPool2d(1)
        self.model = self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=0.9)
        return
    
    def lr_scheduler(self, epoch):
        if epoch>=0.5*self.num_epochs and not self.decay_time[0]: 
            self.decay_time[0] = True
            lr = self.init_lr*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch>=0.8*self.num_epochs and not self.decay_time[1]: 
            self.decay_time[1] = True
            lr = self.init_lr*self.decay_rate*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return
            
    ##################################################
    # step 3: Learning
    ##################################################
    def opt(self, num_epochs):
        # recording time and epoch acc and best result
        since = time.time()
        best_epoch = 0
        best_acc = 0
        record = []
        for epoch in range(num_epochs):
            print('Epoch {}/{} \n '.format(epoch, num_epochs - 1) + '-' * 40)
            self.lr_scheduler(epoch)
            
            tra_loss, tra_acc = self.tra()
            
            record.append((epoch, tra_loss, tra_acc))
            print('tra - Loss:{:.4f} - Acc:{:.4f}'.format(tra_loss, tra_acc))
            
            # deep copy the model
            if epoch >= 1 and tra_acc> best_acc:
                best_acc = tra_acc
                best_epoch = epoch
                best_model = copy.deepcopy(self.model)
                torch.save(best_model, self.dst + 'model_{:02}.pth'.format(self.l))
        
        torch.save(torch.Tensor(record), self.dst + 'record_{:02}.pth'.format(self.l))
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        print('Best tra acc: {:.2f}'.format(best_acc))
        print('Best tra acc in epoch: {}'.format(best_epoch))
        return best_model
    
    def tra(self):
        # Set model to training mode
        self.model.train()
        dsets = ImageReader(self.data_dict_meta, self.data_transforms_tra)
        dataLoader = torch.utils.data.DataLoader(dsets, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        L_data, T_data, N_data = 0.0, 0, 0
        
        # iterate batch
        for data in dataLoader:
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                fvec = self.model(inputs_bt.cuda())
                loss = self.criterion(fvec, labels_bt)
                loss.backward()
                self.optimizer.step()  

            _, preds_bt = torch.max(fvec.cpu(), 1)

            L_data += loss.item()
            T_data += torch.sum(preds_bt == labels_bt).item()
            N_data += len(labels_bt)
            
        return L_data/N_data, T_data/N_data 
    
    def eva(self,best_model):
        best_model.eval()
        dsets = ImageReader(self.data_dict_val, self.data_transforms_val)
        dataLoader = torch.utils.data.DataLoader(dsets, self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        Fvecs = []
        with torch.set_grad_enabled(False):
            for data in dataLoader:
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                fvec = F.normalize(best_model(inputs_bt.cuda()), p = 2, dim = 1)
                Fvecs.append(fvec.cpu())

        Fvecs_all = torch.cat(Fvecs,0)
        torch.save(dsets, self.dst+'testdsets.pth')
        torch.save(Fvecs_all, self.dst+ str(self.l) + 'testFvecs.pth')
        return 

    