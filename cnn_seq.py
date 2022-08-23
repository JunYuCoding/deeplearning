import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)  ## 不知道上面的Linear(1024,64)中的1024 可以先删除此处两行，看输出
        x = self.linear2(x)
        return x
        # return self.model1(x)

cnn = CNN()
print(cnn)
input = torch.ones((64, 3, 32, 32))
output = cnn(input)
print(output.shape)
