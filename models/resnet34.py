# ###
# # Taken from https://github.com/GohVh/resnet34-unet/blob/main/model.py
# ###

# import torch
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F

# class DecoderBlock(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, conv_in_channels, conv_out_channels, up_in_channels=None, up_out_channels=None):
#         super().__init__()
#         """
#         eg:
#         decoder1:
#         up_in_channels      : 1024,     up_out_channels     : 512
#         conv_in_channels    : 1024,     conv_out_channels   : 512

#         decoder5:
#         up_in_channels      : 64,       up_out_channels     : 64
#         conv_in_channels    : 128,      conv_out_channels   : 64
#         """
#         if up_in_channels==None:
#             up_in_channels=conv_in_channels
#         if up_out_channels==None:
#             up_out_channels=conv_out_channels

#         self.up = nn.ConvTranspose2d(up_in_channels, up_out_channels, kernel_size=2, stride=2)
#         self.conv = nn.Sequential(
#             nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(conv_out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(conv_out_channels),
#             nn.ReLU(inplace=True)
#         )

#     # x1-upconv , x2-downconv
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat([x1, x2], dim=1)
#         return self.conv(x)

# class UnetResnet34(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()
#         resnet34 = torchvision.models.resnet34(pretrained=True)
#         filters = [64, 128, 256, 512]

#         self.firstlayer = nn.Sequential(*list(resnet34.children())[:3])
#         self.maxpool = list(resnet34.children())[3]
#         self.encoder1 = resnet34.layer1
#         self.encoder2 = resnet34.layer2
#         self.encoder3 = resnet34.layer3
#         self.encoder4 = resnet34.layer4

#         self.bridge = nn.Sequential(
#             nn.Conv2d(filters[3], filters[3]*2, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(filters[3]*2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
            
#         )

#         self.decoder1 = DecoderBlock(conv_in_channels=filters[3]*2, conv_out_channels=filters[3])
#         self.decoder2 = DecoderBlock(conv_in_channels=filters[3], conv_out_channels=filters[2])
#         self.decoder3 = DecoderBlock(conv_in_channels=filters[2], conv_out_channels=filters[1])
#         self.decoder4 = DecoderBlock(conv_in_channels=filters[1], conv_out_channels=filters[0])
#         self.decoder5 = DecoderBlock(
#             conv_in_channels=filters[1], conv_out_channels=filters[0], up_in_channels=filters[0], up_out_channels=filters[0]
#         )

#         self.lastlayer = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=filters[0], out_channels=filters[0], kernel_size=2, stride=2),
#             nn.Conv2d(filters[0], num_classes, kernel_size=3, padding=1, bias=False)
#         )
    
#     def forward(self, x):
#         e1 = self.firstlayer(x)
#         maxe1 = self.maxpool(e1)
#         e2 = self.encoder1(maxe1)
#         e3 = self.encoder2(e2)
#         e4 = self.encoder3(e3)
#         e5 = self.encoder4(e4)
        
#         c = self.bridge(e5)
        
#         d1 = self.decoder1(c, e5)
#         d2 = self.decoder2(d1, e4)
#         d3 = self.decoder3(d2, e3)
#         d4 = self.decoder4(d3, e2)
#         d5 = self.decoder5(d4, e1)

#         out = self.lastlayer(d5)

#         return out


from multiprocessing import freeze_support
import torchvision

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import random
import cv2
import matplotlib.pyplot as plt

class CUB(Dataset):
    def __init__(self, files_path, labels, train_test, image_name, train=True, 
                 transform=False):
      
        self.files_path = files_path
        self.labels = labels
        self.transform = transform
        self.train_test = train_test
        self.image_name = image_name
        
        if train:
          mask = self.train_test.is_train.values == 1
          
        else:
          mask = self.train_test.is_train.values == 0
        
        
        self.filenames = self.image_name.iloc[mask]
        self.labels = self.labels[mask]
        self.num_files = self.labels.shape[0]
       
      
        
    def __len__(self):
        return self.num_files
    
    def __getitem__(self, index):
        y = self.labels.iloc[index,1] - 1
        
        file_name = self.filenames.iloc[index, 1]
        path = self.files_path/'images'/file_name
        x = read_image(path)
        if self.transform:
            x = apply_transforms(x)
        else:
            x = cv2.resize(x, (224,224))
        x = normalize(x)
        x =  np.rollaxis(x, 2) # To meet torch's input specification(c*H*W) 
        return x,y
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        resnet = models.resnet34(pretrained=True)
        # freezing parameters
        for param in resnet.parameters():
            param.requires_grad = False
        # convolutional layers of resnet34
        layers = list(resnet.children())[:8]
        self.top_model = nn.Sequential(*layers).cuda()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 200)
    
    def forward(self, x):
        x = F.relu(self.top_model(x))
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.shape[0], -1) # flattening 
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.fc2(x)
        return x

def read_image(path):
    im = cv2.imread(str(path))
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Data Augmentation
import math
def center_crop(im, min_sz=None):
    """ Returns a center crop of an image"""
    r,c,*_ = im.shape
    if min_sz is None: min_sz = min(r,c)
    start_r = math.ceil((r-min_sz)/2)
    start_c = math.ceil((c-min_sz)/2)
    return crop(im, start_r, start_c, min_sz, min_sz)

def crop(im, r, c, target_r, target_c): return im[r:r+target_r, c:c+target_c]

def random_crop(x, target_r, target_c):
    """ Returns a random crop"""
    r,c,*_ = x.shape
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(rand_r*(r - target_r)).astype(int)
    start_c = np.floor(rand_c*(c - target_c)).astype(int)
    return crop(x, start_r, start_c, target_r, target_c)

def rotate_cv(im, deg, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, 
                          flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def normalize(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im/255.0 - imagenet_stats[0])/imagenet_stats[1]

def apply_transforms(x, sz=(224, 224), zoom=1.05):
    """ Applies a random crop, rotation"""
    sz1 = int(zoom*sz[0])
    sz2 = int(zoom*sz[1])
    x = cv2.resize(x, (sz1, sz2))
    x = rotate_cv(x, np.random.uniform(-10,10))
    x = random_crop(x, sz[1], sz[0])
    if np.random.rand() >= .5:
                x = np.fliplr(x).copy()
    return x

def denormalize(img):
  imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
  return img*imagenet_stats[1] + imagenet_stats[0]

def show_image(img):
  img = img.transpose(1,2,0)
  img= denormalize(img)
  plt.imshow(img)

def visualize(dataloader, categories):
    """Imshow for Tensor."""
    x,y = next(iter(dataloader))
    
    fig = plt.figure(figsize=(10, 10))
    for i in range(8):
      inp = x[i]
      inp = inp.numpy().transpose(1,2,0)
      inp = denormalize(inp)
      
      ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[])
      plt.imshow(inp)
      plt.title(str(categories[y[i]]))

def get_optimizer(model, lr = 0.01, wd = 0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim

def save_model(m, p): torch.save(m.state_dict(), p)

def load_model(m, p): m.load_state_dict(torch.load(p))

def LR_range_finder(model, train_dl, lr_low=1e-5, lr_high=1, epochs=1, beta=0.9):
  losses = []
  # Model save path
  p = PATH/"mode_tmp.pth"
  save_model(model, str(p))
  num = len(train_dl)-1
  mult = (lr_high / lr_low) ** (1.0/num)
  lr = lr_low
  avg_loss = 0.
  best_loss = 0.
  batch_num = 0
  log_lrs = []

  model.train()
  for i in range(epochs):
    for x,y in train_dl:
      batch_num +=1
      optim = get_optimizer(model, lr=lr)
      x = x.cuda().float()
      y = y.cuda().long()   
      out = model(x)
      criterion = nn.CrossEntropyLoss()
      loss = criterion(out, y)
      print("made it to 336")
      #Compute the smoothed loss
      avg_loss = beta * avg_loss + (1-beta) *loss.item()
      smoothed_loss = avg_loss / (1 - beta**batch_num)

      #Stop if the loss is exploding
      if batch_num > 1 and smoothed_loss > 4 * best_loss:
        return log_lrs, losses

      #Record the best loss
      if smoothed_loss < best_loss or batch_num==1:
        best_loss = smoothed_loss
      #Store the values
      losses.append(smoothed_loss)
      log_lrs.append(math.log10(lr))
      print("made it to 352")
      optim.zero_grad()
      loss.backward()
      optim.step()
      #Update the lr for the next step
      lr *= mult
  print("made it to 357")
  load_model(model, str(p))
  return log_lrs, losses

def get_triangular_lr(lr_low, lr_high, iterations):
    iter1 = int(0.35*iterations)
    iter2 = int(0.85*iter1)
    iter3 = iterations - iter1 - iter2
    delta1 = (lr_high - lr_low)/iter1
    delta2 = (lr_high - lr_low)/(iter1 -1)
    lrs1 = [lr_low + i*delta1 for i in range(iter1)]
    lrs2 = [lr_high - i*(delta1) for i in range(0, iter2)]
    delta2 = (lrs2[-1] - lr_low)/(iter3)
    lrs3 = [lrs2[-1] - i*(delta2) for i in range(1, iter3+1)]
    return lrs1+lrs2+lrs3

def train_triangular_policy(model, train_dl, valid_dl, lr_low=1e-5, 
                            lr_high=0.01, epochs = 4):
    idx = 0
    iterations = epochs*len(train_dl)
    lrs = get_triangular_lr(lr_low, lr_high, iterations)
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for i, (x, y) in enumerate(train_dl):
            optim = get_optimizer(model, lr = lrs[idx], wd =0)
            batch = y.shape[0]
            x = x.cuda().float()
            y = y.cuda().long()
            out = model(x)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            idx += 1
            total += batch
            sum_loss += batch*(loss.item())
        print("train loss", sum_loss/total)
        val_metrics(model, valid_dl)
    return sum_loss/total

def val_metrics(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0 
    for x, y in valid_dl:
        batch = y.shape[0]
        x = x.cuda().float()
        y = y.cuda().long()
        out = model(x)
        _, pred = torch.max(out, 1)
        correct += pred.eq(y.data).sum().item()
        y = y.long()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, y)
        sum_loss += batch*(loss.item())
        total += batch
    print("val loss and accuracy", sum_loss/total, correct/total)

from datetime import datetime
def training_loop(model, train_dl, valid_dl, steps=3, lr_low=1e-6, lr_high=0.01, epochs = 4):
    for i in range(steps):
        start = datetime.now() 
        loss = train_triangular_policy(model, train_dl, valid_dl, lr_low, lr_high, epochs)
        end = datetime.now()
        t = 'Time elapsed {}'.format(end - start)
        print("----End of step", t)

def set_trainable_attr(m, b=True):
    for p in m.parameters(): p.requires_grad = b

def unfreeze(model, l):
    top_model = model.top_model
    set_trainable_attr(top_model[l])





#New
lrs, losses = LR_range_finder(model, train_loader, lr_low=1e-7, lr_high=0.01)
plt.plot(lrs, losses)
plt.show()

training_loop(model, train_loader, valid_loader, steps=1, lr_low= 1e-5, lr_high=1*1e-4, epochs = 30)

p = PATH/"model2_tmp.pth"
save_model(model, str(p))
load_model(model, str(p))



load_model(model, str(p))
unfreeze(model, 4)
unfreeze(model, 3)
unfreeze(model,2)



lrs, losses = LR_range_finder(model, train_loader, lr_low=1e-7, lr_high=0.01)
plt.plot(lrs, losses)
plt.show()




training_loop(model, train_loader, valid_loader, steps=1, lr_low= 1e-7, lr_high=1*1e-5, epochs = 10)



training_loop(model, train_loader, valid_loader, steps=1, lr_low= 1e-7, lr_high=5*1e-7, epochs = 10)

if __name__ == '__main__':
    PATH = Path('./archive/CUB_200_2011')
    labels = pd.read_csv(PATH/"image_class_labels.txt", header=None, sep=" ")
    labels.columns = ["id", "label"]
    labels.head(2)

    print("Made it to 275")
    labels.describe()


    print("Made it to 279")
    train_test = pd.read_csv(PATH/"train_test_split.txt", header=None, sep=" ")
    train_test.columns = ["id", "is_train"]
    train_test.head(2)

    print("Made it to 284")
    images = pd.read_csv(PATH/"images.txt", header=None, sep=" ")
    images.columns = ["id", "name"]
    images.head(2)

    print("Made it to 289")
    classes = pd.read_csv(PATH/"classes.txt", header=None, sep=" ")
    classes.columns = ["id", "class"]
    classes.head(2)

    print("Made it to 294")
    categories = [x for x in classes["class"]]

    print("Made it to 297")
    train_dataset = CUB(PATH, labels, train_test, images, train= True, transform= True)
    valid_dataset = CUB(PATH, labels, train_test, images, train= False, transform= False)
    print("Made it to 300")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=1)
    p = PATH/"model1_tmp.pth"
    save_model(model, str(p))
    load_model(model, str(p))
    
    model = Net().cuda()


print("Made it to 432")
lrs, losses = LR_range_finder(model, train_loader, lr_low=1e-6, lr_high=0.1)


print("Made it to 436")
plt.plot(lrs, losses)
plt.show()



val_metrics(model, valid_loader)




training_loop(model, train_loader, valid_loader, steps=1, lr_low= 1e-3, lr_high=1*1e-2, epochs = 16)



    freeze_support()
    classes.columns = ["id", "class"]
    classes.head(2)

    print("Made it to 294")
    categories = [x for x in classes["class"]]

    print("Made it to 297")
    train_dataset = CUB(PATH, labels, train_test, images, train= True, transform= True)
    valid_dataset = CUB(PATH, labels, train_test, images, train= False, transform= False)
    print("Made it to 300")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=1)

    print("Made it to 304")
    def get_optimizer(model, lr = 0.01, wd = 0.0):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optim = torch.optim.Adam(parameters, lr=lr, weight_decay=wd)
        return optim

    print("Made it to 310")
    def save_model(m, p): torch.save(m.state_dict(), p)
    print("Made it to 312")
