import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


# VGG
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        return self.linear1(input)


vgg = CNN()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    input = torch.reshape(imgs, (1, 1, 1, -1))  # 展开成1，1，1，xxxx
    # input = torch.flatten(imgs) # 或这种
    output = vgg(input)
    print(output.shape)
