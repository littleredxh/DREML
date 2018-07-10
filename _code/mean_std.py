import torch,os
from torchvision import datasets, transforms, models
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

src = 'ICR/'

data_transforms = transforms.Compose([
    transforms.Scale([64,64]),
#     transforms.CenterCrop(128),
    transforms.ToTensor()])

dsets = datasets.ImageFolder(src, data_transforms)# os.path.join(src, 'tra')
mean = []
stdd = []

for i in range(len(dsets)):
    print(i,end='\r')
    
    mean.append(dsets[i][0].mean(1).mean(1))

mean_all = torch.stack(mean,0)
m = mean_all.mean(0)
print(m)

for i in range(len(dsets)):
    print(i,end='\r')
    img = dsets[i][0]
    img[0,:,:] = img[0,:,:]-m[0]
    img[1,:,:] = img[1,:,:]-m[1]
    img[2,:,:] = img[2,:,:]-m[2]
    img_std = img.pow(2).sum(1).sum(1)
    stdd.append(img_std)

stdd_all = torch.stack(stdd,0)
print(torch.sqrt(stdd_all.sum(0)/(img.size(1)*img.size(2)*len(dsets)-1)))