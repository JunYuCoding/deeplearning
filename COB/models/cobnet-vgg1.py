import torch
import numpy as np
from .cobnet_orientation import CobNetOrientationModule
from .cobnet_fuse import CobNetFuseModule
from torch import nn

# import utils as utls
from torchvision import transforms as trfms
import torchvision.models as models
from torchvision.models.resnet import Bottleneck
import math
from torch.nn import functional as F


def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1:y1 + th, x1:x1 + tw]


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 -
                                                 abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


class CobNet(nn.Module):
    def __init__(self, n_orientations=8):

        super(CobNet, self).__init__()

        # self.base_model = models.resnet50(pretrained=True)
        # replace the model with vgg16
        self.base_model =  models.vgg16(pretrained=True)

        # use the following self.reducers for resnet50
        # self.reducers = nn.ModuleList([
        #     nn.Conv2d(self.base_model.conv1.out_channels,#64
        #               out_channels=1, kernel_size=1),
        #     nn.Conv2d(self.base_model.layer1[-1].conv3.out_channels,  # 256
        #               out_channels=1,
        #               kernel_size=1),
        #     nn.Conv2d(self.base_model.layer2[-1].conv3.out_channels, #512
        #               out_channels=1,
        #               kernel_size=1),
        #     nn.Conv2d(self.base_model.layer3[-1].conv3.out_channels, #1024
        #               out_channels=1,
        #               kernel_size=1),
        #     nn.Conv2d(self.base_model.layer4[-1].conv3.out_channels, #2048
        #               out_channels=1,
        #               kernel_size=1),
        # ])

        # use the following self.reducers for vgg16 referring to RCF architecture
        self.reducers = nn.ModuleList([
            nn.Conv2d(self.base_model.features[2].out_channels, #64
                       out_channels=1, kernel_size=1),
            nn.Conv2d(self.base_model.features[7].out_channels,  #128
                       out_channels=1,
                       kernel_size=1),
            nn.Conv2d(self.base_model.features[14].out_channels,  #256
                       out_channels=1,
                       kernel_size=1),
            nn.Conv2d(self.base_model.features[21].out_channels, #512
                       out_channels=1,
                       kernel_size=1),
            nn.Conv2d(self.base_model.features[28].out_channels, #512
                       out_channels=1,
                       kernel_size=1),
        ])

        for m in self.reducers:
            # 按正态分布对tensor随机赋值
            nn.init.normal_(m.weight, std=0.01)
            #使用 nn.init.constant_对参数赋值
            nn.init.constant_(m.bias, 0)

        self.fuse = CobNetFuseModule()

        self.n_orientations = n_orientations
        # use the following self.orientations for resnet50
        # self.orientations = nn.ModuleList(
        #     [CobNetOrientationModule(in_channels=[64, 256, 512, 1024, 2048]) for _ in range(n_orientations)])

        # use the following self.orientations for vgg 1&2
        self.orientations = nn.ModuleList(
        [CobNetOrientationModule(in_channels=[64, 128, 256, 512, 512]) for _ in range(n_orientations)])
    def forward_sides(self, im):
        in_shape = im.shape[2:]
        # pass through base_model and store intermediate activations (sides)
        # using the following part for resnet50
        # pre_sides = []
        # x = self.base_model.conv1(im)
        # x = self.base_model.bn1(x)
        # x = self.base_model.relu(x)  # 64
        # pre_sides.append(x)
        # x = self.base_model.maxpool(x)
        # x = self.base_model.layer1(x)  #256
        # pre_sides.append(x)
        # x = self.base_model.layer2(x) # 512
        # pre_sides.append(x)
        # x = self.base_model.layer3(x)  #1024
        # pre_sides.append(x)
        # x = self.base_model.layer4(x)  #2048
        # pre_sides.append(x)

        # using the following for VGG16-RCF (vgg1)
        pre_sides = []
        index_list = [0,4,9,16,23]
        for i, index in enumerate(index_list):
            if i==0:
                x = self.base_model.features[:index_list[i + 1]](im)
            elif i==4:
                x = self.base_model.features[index_list[i]:29](x)
            else:
                x = self.base_model.features[index_list[i]:index_list[i+1]](x)
            pre_sides.append(x) # 64, 128, 256, 512, 512

        # using the following for VGG16-RCF, sum of all conv layer (vgg2)
        # pre_sides = []
        # x1 = self.base_model.features[:2](im)
        # x2 = self.base_model.features[2:4](x1)
        # pre_sides.append(x1 + x2)
        # x1 = self.base_model.features[4:7](x2)
        # x2 = self.base_model.features[7:9](x1)
        # pre_sides.append(x1 + x2)
        # x1 = self.base_model.features[9:12](x2)
        # x2 = self.base_model.features[12:14](x1)
        # x3 = self.base_model.features[14:16](x2)
        # pre_sides.append(x1 + x2 + x3)
        # x1 = self.base_model.features[16:19](x3)
        # x2 = self.base_model.features[19:21](x1)
        # x3 = self.base_model.features[21:23](x2)
        # pre_sides.append(x1 + x2 + x3)
        # x1 = self.base_model.features[23:26](x3)
        # x2 = self.base_model.features[26:28](x1)
        # x3 = self.base_model.features[28:30](x2)
        # pre_sides.append(x1 + x2 + x3)
        ''' vgg model
        features Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace=True)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace=True)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace=True)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace=True)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace=True)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace=True)
  (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): ReLU(inplace=True)
  (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace=True)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace=True)
  (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (25): ReLU(inplace=True)
  (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (27): ReLU(inplace=True)
  (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (29): ReLU(inplace=True)
  (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
        '''

        late_sides = []
        for s, m in zip(pre_sides, self.reducers):# [(_,_)()()] m => conv,  s => x

            late_sides.append(m(s))
        upsamp = nn.UpsamplingBilinear2d(in_shape)
        so0 = upsamp(late_sides[0])
        so1 = upsamp(late_sides[1])
        so2 = upsamp(late_sides[2])
        so3 = upsamp(late_sides[3])
        so4 = upsamp(late_sides[4])

        return pre_sides, [so0, so1, so2, so3, so4]

    def forward_orient(self, sides, shape=512):

        upsamp = nn.UpsamplingBilinear2d((shape, shape))
        orientations = []

        for m in self.orientations:
            or_ = upsamp(m(sides))
            orientations.append(or_)

        return orientations

    def forward_fuse(self, sides):

        y_fine, y_coarse = self.fuse(sides)

        return y_fine, y_coarse

    def forward(self, im):
        pre_sides, late_sides = self.forward_sides(im)

        orientations = self.forward_orient(pre_sides)
        y_fine, y_coarse = self.forward_fuse(late_sides)

        return {
            'pre_sides': pre_sides,
            'late_sides': late_sides,
            'orientations': orientations,
            'y_fine': y_fine,
            'y_coarse': y_coarse
        }
if __name__ =="__main__" :
    cob = CobNet()
    print(cob.base_model.layer1)
    #
    # vgg= models.vgg16(pretrained=True)
    # print(vgg)

