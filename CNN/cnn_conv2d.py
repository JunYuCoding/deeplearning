import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataLoader = DataLoader(dataset, batch_size=64)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)  # 卷积层

    def forward(self, x):
        x = self.conv1(x)
        return x


writer = SummaryWriter("./logs")
cnn = CNN()
step = 0
for data in dataLoader:
    imgs, targets = data
    output = cnn(imgs)
    print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    print(output.shape)  # torch.Size([64, 6, 30, 30]) ->变成【xxx,3,30,30】 要变成3个channel
    output = torch.reshape(output, (-1, 3, 30, 30))
    print(output)
    break
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1
writer.close()
