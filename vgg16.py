import torchvision
import torch
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
print(vgg16)
# apply to cifar10 dataset
vgg16.classifier.add_module("add_linear",nn.Linear(1000,10
                                                   ))
print(vgg16)
# or modify
vgg16.classifier[6] = nn.Linear(4096,10)
print(vgg16)
#
# torch.save(vgg16,"./model/vgg16saved.pth")
# torch.save(vgg16.state_dict(),"vgg16saved2.pth")


