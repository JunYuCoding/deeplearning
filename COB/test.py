import torch
from torch import nn
import torchvision.models as models

from models.cobnet_fuse import CobNetFuseModule
from models.cobnet_orientation import CobNetOrientationModule
#
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir='./log/')
class Net(nn.Module):
    def __init__(self, n_orientations=8):

        super(Net, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        ''' reducers:
            Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
            Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
            Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
            Conv2d(1024, 1, kernel_size=(1, 1), stride=(1, 1))
            Conv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1))
        '''
        self.reducers = nn.ModuleList([
            nn.Conv2d(self.base_model.conv1.out_channels,#64
                      out_channels=1, kernel_size=1),
            nn.Conv2d(self.base_model.layer1[-1].conv3.out_channels,  # 256
                      out_channels=1,
                      kernel_size=1),
            nn.Conv2d(self.base_model.layer2[-1].conv3.out_channels, #512
                      out_channels=1,
                      kernel_size=1),
            nn.Conv2d(self.base_model.layer3[-1].conv3.out_channels, #1024
                      out_channels=1,
                      kernel_size=1),
            nn.Conv2d(self.base_model.layer4[-1].conv3.out_channels, #2048
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
            pre_sides = []
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

            late_sides = []
            for s, m in zip(pre_sides, self.reducers):
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


net = Net()
# print(net.base_model)
# # 参数model就是你的实例化好的模型，参数input_to_model就是输入到模型的数据。
net.base_model.conv1.kernel_size=3
#然后替换一下就可以了
net.base_model.maxpool = nn.Sequential()
print(net.base_model)
# base = models.resnet50(pretrained=True)
# print(base)