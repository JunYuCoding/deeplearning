import time

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=1)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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

        return self.model1(x)

cnn = CNN()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(cnn.parameters(),lr=0.01)

for epoch in range(20):
    start = time.time()
    epoch_loss = 0.0 # 每个epoch的损失总和
    for imgs, targets in dataloader:
        outputs = cnn(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        # print(result_loss)
        epoch_loss += result_loss
    print(time.time()-start,"回合总损失",epoch_loss)
