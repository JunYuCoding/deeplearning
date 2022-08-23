import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataLoader = DataLoader(dataset, batch_size=64)
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)
    def forward(self,input):
        output = self.maxpool1(input)
        return output
writer = SummaryWriter("./logs")
cnn = CNN()
step = 0
for data in dataLoader:
    imgs, targets = data
    output = cnn(imgs)
    writer.add_images("input_pool", imgs, step)
    writer.add_images("output_pool", output, step)
    step += 1
writer.close()