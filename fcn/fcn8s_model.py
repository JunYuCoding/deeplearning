import os
# https://github.com/bat67/pytorch-FCN-easiest-demo/blob/master/FCN.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision import models
from torchvision.models.vgg import VGG

import cv2
import numpy as np


# # 将标记图（每个像素值代该位置像素点的类别）转换为onehot编码
# def onehot(data, n):
#     buf = np.zeros(data.shape + (n,))
#     nmsk = np.arange(data.size) * n + data.ravel()
#     buf.ravel()[nmsk - 1] = 1
#     return buf
#
#
# # 利用torchvision提供的transform，定义原始图片的预处理步骤（转换为tensor和标准化处理）
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
#
# # 利用torch提供的Dataset类，定义我们自己的数据集
# class BagDataset(Dataset):
#
#     def __init__(self, transform=None):
#         self.transform = transform
#
#     def __len__(self):
#         return len(os.listdir('./bag_data'))
#
#     def __getitem__(self, idx):
#         img_name = os.listdir('./bag_data')[idx]
#         imgA = cv2.imread('./bag_data/' + img_name)
#         imgA = cv2.resize(imgA, (160, 160))
#         imgB = cv2.imread('./bag_data_msk/' + img_name, 0)
#         imgB = cv2.resize(imgB, (160, 160))
#         imgB = imgB / 255
#         imgB = imgB.astype('uint8')
#         imgB = onehot(imgB, 2)
#         imgB = imgB.transpose(2, 0, 1)
#         imgB = torch.FloatTensor(imgB)
#         # print(imgB.shape)
#         if self.transform:
#             imgA = self.transform(imgA)
#
#         return imgA, imgB
#
#
# # 实例化数据集
# bag = BagDataset(transform)
#
# train_size = int(0.9 * len(bag))
# test_size = len(bag) - train_size
# train_dataset, test_dataset = random_split(bag, [train_size, test_size])
#
# # 利用DataLoader生成一个分batch获取数据的可迭代对象
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
# test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

# <-------------------------------------------------------->#
# 下面开始定义网络模型
# 先定义VGG结构

# ranges 是用于方便获取和记录每个池化层得到的特征图
# 例如vgg16，需要(0, 5)的原因是为方便记录第一个pooling层得到的输出(详见下午、稳VGG定义)
ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
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
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
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
            # self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16 examples)
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x
        # output 为一个字典键x1d对应第一个maxpooling输出的特征图，x2...x5类推
        return output


# 下面由VGG构建FCN8s
class FCN8s(nn.Module):

    def __init__(self, pretrained_net, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # self.pretrained_net = pretrained_net
        net = models.vgg16(pretrained=True)   # 从预训练模型加载VGG16网络参数
        self.pretrained_net = net.features  # 只使用Vgg16的五层卷积层（特征提取层）（3，224，224）----->（512，7，7）
        # backbone采用VGG16，把VGG的fully-connect层用卷积来表示，即conv6-7（一个大小和feature_map同样size的卷积核，就相当于全连接
        self.conv6 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        #（512，7，7）
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        # (512, 14, 14)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        # (256, 28, 28)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # (128,56,56)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # (64, 112, 112)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        # (32, 224, 224)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
        # (num_classes, 224, 224)
    def forward(self, x):
        # output = self.pretrained_net(x)
        # x5 = output['x5']  # maxpooling5的feature map (1/32)
        # x4 = output['x4']  # maxpooling4的feature map (1/16)
        # x3 = output['x3']  # maxpooling3的feature map (1/8)
        for i in range(len(self.pretrained_net)):
            x = self.pretrained_net[i](x)
            if i == 16:
                x3 = x  # maxpooling3的feature map (1/8)
            if i == 23:
                x4 = x  # maxpooling4的feature map (1/16)
            if i == 30:
                x5 = x  # maxpooling5的feature map (1/32)
        score = self.relu(self.conv6(x5))  # conv6  size不变 (1/32)
        score = self.relu(self.conv7(score))  # conv7  size不变 (1/32)

        score = self.relu(self.deconv1(x5))  # out_size = 2*in_size (1/16)
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))  # out_size = 2*in_size (1/8)
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))  # out_size = 2*in_size (1/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # out_size = 2*in_size (1/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # out_size = 2*in_size (1)
        score = self.classifier(score)  # size不变，使输出的channel等于类别数

        return score
if __name__ == "__main__":
    vgg_model = VGGNet(requires_grad=True, show_params=False)
    model = FCN8s(pretrained_net=vgg_model,num_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(model)
#
# vgg_model = VGGNet(requires_grad=True, show_params=False)
# print(vgg_model)
# fcn_model = FCN8s(pretrained_net=vgg_model, num_classes=2)
# print(fcn_model)
'''
FCN的代码实现上总体都是卷积+转置卷积+跳链接的结构。实际上只要实现特征提取（提取抽象特征）——转置卷积（恢复原图大小）——给每一个像素分类的过程就够了。
采用vgg16的五层卷积层作为特征提取网络，然后接五个转置卷积（2x）恢复到原图大小，然后再接一个卷积层把feature map的通道调整为类别个数（21）。最后再softmax分类就行了。

'''