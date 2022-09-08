import time

import torchvision
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import *
#
# from cnn_model import *

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
print("train_data_length:{},test_data_length:{}".format(len(train_data), len(test_data)))

train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
cnn.to(device)
net = cnn
for pname, p in net.named_parameters():
    print(pname,"==",p)
print(net.named_parameters())
# 损失函数
loss_function = nn.CrossEntropyLoss()
loss_function.to(device)
# 学习率
learning_rate = 1e-2  # 1* 10^-2 = 0.01

# 优化器
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 记录训练的轮数
epoch = 10
# 添加tensorboard
writer = SummaryWriter("./logs")
for i in range(epoch):
    print("第 {} 轮训练开始".format(i + 1))
    # 训练步骤开始
    start_time = time.time()
    cnn.train()  # if net hava dropout and batchNorm要这行
    for imgs, targets in train_data_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = cnn(imgs)
        loss = loss_function(outputs, targets)
        # optimizer 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("时间",end_time-start_time)
            print("当前第{}轮,训练第{}次，Loss:{}".format(i + 1, total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 测试步骤开始
    cnn.eval()  # if net hava dropout and batchNorm要这行
    total_test_loss = 0.0
    # 整体正确率
    total_accuracy = 0.0
    with torch.no_grad():
        for imgs, targets in test_data_loader:
            outputs = cnn(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("第{}轮，整体测试集上的Loss:{}".format(i + 1, total_test_loss))
    print("第{}轮，整体测试集上的accuracy:{}".format(i + 1, total_accuracy / len(test_data)))
    writer.add_scalar("total_test_loss", total_test_loss, total_test_step)
    writer.add_scalar("total_accuracy", total_accuracy / (len(test_data)), total_test_step)
    total_test_step += 1
    # 每轮可保存模型
    torch.save(cnn, "./model/cnn_{}epoch.pth".format(i + 1))
    print("模型已保存")
