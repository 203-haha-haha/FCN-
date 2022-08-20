import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision import models
from torchvision.models.vgg import VGG

import cv2
import numpy as np
from datetime import datetime

import torch.optim as optim
import matplotlib.pyplot as plt

# 将标记图（每个像素值代该位置像素点的类别）转换为onehot编码
def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf

# 利用torchvision提供的transform，定义原始图片的预处理步骤（转换为tensor和标准化处理） 
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# 利用torch提供的Dataset类，定义我们自己的数据集
class BagDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir('./bag_data'))

    def __getitem__(self, idx):
        img_name = os.listdir('./bag_data')[idx]
        imgA = cv2.imread('./bag_data/'+img_name)
        imgA = cv2.resize(imgA, (160, 160))

        imgB = cv2.imread('./bag_data_mask/'+img_name, 0)
        imgB = cv2.resize(imgB, (160, 160))
       # imgB = imgB/255               #将图片转换为灰度0-1之间
        #imgB = imgB.astype('uint8')
        #imgB = onehot(imgB, 2)
        #imgB = imgB.transpose(2,0,1)
        imgB = torch.LongTensor(imgB)
        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)
        return imgA, imgB

# <-------------------------------------------------------->#
# 下面开始定义网络模型
# 先定义VGG结构
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# Vgg网络结构配置（数字代表经过卷积后的channel数，‘M’代表卷积层）
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 由cfg构建vgg-Net的卷积层和池化层(block1-block5)
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# 下面开始构建VGGnet
class VGGNet(VGG):
    def __init__(self, pretrained = True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]
        
        # 获取VGG模型训练好的参数，并加载（第一次执行需要下载一段时间）
        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        # 去掉vgg最后的全连接层(classifier)
        if remove_fc:  
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # 利用之前定义的ranges获取每个maxpooling层输出的特征图
        for idx, (begin, end) in enumerate(self.ranges):
        #self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16 examples)
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x
        # output 为一个字典键x1d对应第一个maxpooling输出的特征图，x2...x5类推
        return output

# 下面由VGG构建FCN8s
class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.conv6 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']    # maxpooling5的feature map (1/32)
        x4 = output['x4']    # maxpooling4的feature map (1/16)
        x3 = output['x3']    # maxpooling3的feature map (1/8)
    
        score = self.relu(self.conv6(x5))    # conv6  size不变 (1/32)
        score = self.relu(self.conv7(score)) # conv7  size不变 (1/32)
        score = self.relu(self.deconv1(x5))   # out_size = 2*in_size (1/16)       
        score = self.bn1(score + x4)                      
        score = self.relu(self.deconv2(score)) # out_size = 2*in_size (1/8)           
        score = self.bn2(score + x3)                      
        score = self.bn3(self.relu(self.deconv3(score)))  # out_size = 2*in_size (1/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # out_size = 2*in_size (1/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # out_size = 2*in_size (1)
        score = self.classifier(score)                    # size不变，使输出的channel等于类别数
        
        return score  


class FCN32s(nn.Module):
    
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']

        score = self.bn1(self.relu(self.deconv1(x5)))     
        score = self.bn2(self.relu(self.deconv2(score)))  
        score = self.bn3(self.relu(self.deconv3(score)))  
        score = self.bn4(self.relu(self.deconv4(score)))  
        score = self.bn5(self.relu(self.deconv5(score)))  
        score = self.classifier(score)                    
        #print(score.shape)
        return score


def get_boundary(pic,is_mask):
    if not is_mask:
        pic = torch.argmax(pic,1).cpu().numpy().astype('float64')
    else:
        pic = pic.cpu().numpy()
    batch = pic.shape[0]
    width = pic.shape[1]
    height=pic.shape[2]
    new_pic = np.zeros([batch, width + 2, height + 2])
    mask_erode = np.zeros([batch, width, height])
    dil = int(round(0.02*np.sqrt(width ** 2 + height ** 2)))
    if dil < 1:
        dil = 1
    for i in range(batch):
        new_pic[i] = cv2.copyMakeBorder(pic[i], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for j in range(batch):
        pic_erode = cv2.erode(new_pic[j],kernel,iterations=dil)
        mask_erode[j] = pic_erode[1: width + 1, 1: height + 1]
    return torch.from_numpy(pic-mask_erode)

def get_biou(pre_pic ,real_pic):
    inter = 0
    union = 0
    pre_pic = get_boundary(pre_pic, is_mask=False)
    real_pic = get_boundary(real_pic, is_mask=True)
    batch, width, height = pre_pic.shape
    for i in range(batch):
        predict = pre_pic[i]
        mask = real_pic[i]
        inter += ((predict * mask) > 0).sum()
        union += ((predict + mask) > 0).sum()
    if union < 1:
        return 0
    biou = (inter/union)
    return biou
def get_miou(pre_pic, real_pic):
    miou = 0
    pre_pic = torch.argmax(pre_pic,1)
    batch, width, height = pre_pic.shape
    for i in range(batch):
        predict = pre_pic[i]
        mask = real_pic[i]
        union = torch.logical_or(predict,mask).sum()
        inter = ((predict + mask)==2).sum()
        if union < 1e-5:
            return 0
        miou += inter / union
    return miou/batch

# 实例化数据集
bag = BagDataset(transform)

train_size = int(0.85 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

# 利用DataLoader生成一个分batch获取数据的可迭代对象
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

epochs=30
lr=1e-5
vgg_model = VGGNet(requires_grad=True, show_params=False)
model= FCN32s(pretrained_net=vgg_model, n_class=2)
loss=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters(),lr=lr)
test_Acc = []
test_mIou = []
test_biou=[]

for epoch in range(epochs):
    mIoU=[]
    BIoU=[]
    for X,y in train_loader:
        total_loss=[]
        opt.zero_grad()
        y_pre=model(X)
        l=loss(y_pre, y)
        total_loss.append(l)
        l.backward()
        opt.step()
    total_loss=sum(total_loss)
    print('epoch:',epoch+1,'loss:',total_loss)


    with torch.no_grad():
        ttotal_loss=[]
        for tx,ty in test_loader:
            ty_pre=model(tx)
            loss1=loss(ty_pre,ty)
            ttotal_loss.append(loss1)
        ttotal_loss=sum(ttotal_loss)
        print('test_loss:',ttotal_loss)
        mIoU.append(get_miou(ty_pre, ty))
        BIoU.append(get_biou(ty_pre, ty))
    test_mIou.append(sum(mIoU) / len(mIoU))
    test_biou.append(sum(BIoU) / len(BIoU))
    print("mIoU是:")
    print(test_mIou)
    print("BIoU是:")
    print(test_biou)

    if epoch>=1:
        torch.save(model.state_dict(), 'checkpoints/fcn_model_{}.pth'.format(epoch))
