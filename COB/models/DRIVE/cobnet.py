import torch
import numpy as np
from cobnet_orientation import CobNetOrientationModule
from cobnet_fuse import CobNetFuseModule
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
        # self.base_model = models.segmentation.deeplabv3_resnet50(pretrained=True)

        # 然后替换一下就可以了
        self.base_model = models.resnet50(pretrained=True)
        # self.base_model =  models.vgg16(pretrained=True)

        # self.base_model.maxpool = nn.Sequential()
        ''' reducers:
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
            Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
            Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
            Conv2d(1024, 1, kernel_size=(1, 1), stride=(1, 1))
            Conv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1))
        '''
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
        self.reducers = nn.ModuleList([
            nn.Conv2d(64,
                      out_channels=1, kernel_size=1),
            nn.Conv2d(256,  # 256
                      out_channels=1,
                      kernel_size=1),
            nn.Conv2d(512,  # 512
                      out_channels=1,
                      kernel_size=1),
            nn.Conv2d(1024,  # 1024
                      out_channels=1,
                      kernel_size=1),
            nn.Conv2d(2048,  # 2048
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
        self.orientations = nn.ModuleList(
            [CobNetOrientationModule() for _ in range(n_orientations)])

    def forward_sides(self, im):
        in_shape = im.shape[2:]
        # pass through base_model and store intermediate activations (sides)
        pre_sides = [] # 通过resnet50 然后添加到pre_sides
        x = self.base_model.conv1(im)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        pre_sides.append(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        pre_sides.append(x)
        x = self.base_model.layer2(x)
        pre_sides.append(x)
        x = self.base_model.layer3(x)
        pre_sides.append(x)
        x = self.base_model.layer4(x)
        pre_sides.append(x)
        '''
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

        # x = self.base_model.features[0](im)
        # x = self.base_model.features[1](x)
        # pre_sides.append(x)
        # x = self.base_model.features[2](x)
        # x = self.base_model.features[3](x)
        # pre_sides.append(x)
        # x = self.base_model.features[4](x)
        # x = self.base_model.features[5](x)
        # x = self.base_model.features[6](x)
        # pre_sides.append(x)
        # x = self.base_model.features[7](x)
        # x = self.base_model.features[8](x)
        # pre_sides.append(x)
        # x = self.base_model.features[9](x)
        # x = self.base_model.features[10](x)
        # x = self.base_model.features[11](x)
        # pre_sides.append(x)

        # x = self.base_model.features[12](x)
        # x = self.base_model.features[13](x)
        # pre_sides.append(x)
        # x = self.base_model.features[14](x)
        # x = self.base_model.features[15](x)
        # pre_sides.append(x)
        # x = self.base_model.features[16](x)
        # x = self.base_model.features[17](x)
        # x = self.base_model.features[18](x)
        # pre_sides.append(x)
        # x = self.base_model.features[19](x)
        # x = self.base_model.features[20](x)
        # pre_sides.append(x)
        # x = self.base_model.features[21](x)
        # x = self.base_model.features[22](x)
        # pre_sides.append(x)
        # x = self.base_model.features[23](x)
        # x = self.base_model.features[24](x)
        # x = self.base_model.features[25](x)
        # pre_sides.append(x)
        # x = self.base_model.features[26](x)
        # x = self.base_model.features[27](x)
        # pre_sides.append(x)
        # x = self.base_model.features[28](x)
        # x = self.base_model.features[29](x)
        # pre_sides.append(x)

        late_sides = []
        for s, m in zip(pre_sides, self.reducers):# [(_,_)()()] m => conv,  s => x

            late_sides.append(m(s))
        
        # img_H, img_W = in_shape[0], in_shape[1]
        # weight_deconv0 = make_bilinear_weights(2, 1).cuda()
        # weight_deconv1 = make_bilinear_weights(4, 1).cuda()
        # weight_deconv2 = make_bilinear_weights(8, 1).cuda()
        # weight_deconv3 = make_bilinear_weights(16, 1).cuda()
        # weight_deconv4 = make_bilinear_weights(32, 1).cuda()

        # upsample0 = F.conv_transpose2d(late_sides[0], weight_deconv0, stride=2)
        # upsample1 = F.conv_transpose2d(late_sides[1], weight_deconv1, stride=4)
        # upsample2 = F.conv_transpose2d(late_sides[2], weight_deconv2, stride=8)
        # upsample3 = F.conv_transpose2d(late_sides[3],
        #                                weight_deconv3,
        #                                stride=16)
        # upsample4 = F.conv_transpose2d(late_sides[4],
        #                                weight_deconv4,
        #                                stride=32)

        # so0 = crop(upsample0, img_H, img_W)
        # so1 = crop(upsample1, img_H, img_W)
        # so2 = crop(upsample2, img_H, img_W)
        # so3 = crop(upsample3, img_H, img_W)
        # so4 = crop(upsample4, img_H, img_W)
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

