import numpy
import torch
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from fcn8s_model import VGGNet, FCN8s
from cnn_model import *
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("../../dataset/VOC2012/JPEGImages/2007_000032.jpg")  # airplane
# image.show()
image = image.convert('RGB')  # png 四通道，还有一个透明通道
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.ToTensor()]
)
image = transform(image)
print(image.shape)

vgg_model = VGGNet(requires_grad=True, show_params=False)
# fcn_model = FCN8s(pretrained_net=vgg_model, num_classes=2)
model = FCN8s(vgg_model, 21)
model.load_state_dict(torch.load("./checkpoints/best.pt"))
# model = torch.load("./checkpoints/best.pt")
print(model)  # dict
# torch.Size([32, 3, 224, 224])
image = torch.reshape(image, (1, 3, 224, 224))
print(image.shape)
model.eval()
writer = SummaryWriter("./fcn_logs")

with torch.no_grad():
    output = model(image)

print(output.shape)
classes = output.argmax(dim=1)
print(classes.shape)
print(classes)
classes[classes != 1] = 0
print(classes.shape)
classes_one = np.array(classes, dtype=np.bool)
numpy.save("n",classes_one)
classes_one.astype(np.bool)
print(classes_one.dtype)
classes_one.resize(224, 224)
plt.imshow(classes_one)
plt.show()

# # 将tensor转化成numpy
# img_numpy = output[0].numpy()
# cv2.imshow("imgshow",img_numpy)
# cv2.waitKey(0)

# writer.add_image("output",output,1,dataformats="HWC")
# print(output.argmax(1))  # 1表示列 | | | 横着比 ，0 表示行  从上往下纵向着比
'''
这里通过一个反卷积层，将输入来的heatMap的尺寸变为32倍，于是得到了一个与原图尺寸一样特征图，
再通过一个卷积层(不改变图像尺寸)将图像通道数改变为分类的类别数量，
最后通过softmax，则每个像素点的位置处是一个深度为类别数量的向量，每个值分别代表属于对应类别的概率，
最后再通过np.argmax()就可以得到一张二维数据图，每个像素值就代表对应的类别

'''