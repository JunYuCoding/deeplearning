import torch
import torchvision.transforms

from cnn_model import *
import cv2
from PIL import Image

# image = cv2.imread("./imgs/dog.png")
# image = Image.open("./imgs/dog.png") # 类别5
# image = Image.open("./imgs/airplane.png") # 类别0
image = Image.open("./imgs/bird.png") # 类别2
print(image)
image = image.convert('RGB')  # png 四通道，还有一个透明通道
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)),
     torchvision.transforms.ToTensor()]
)
image = transform(image)
print(image.shape)
model = torch.load("./model/cnn_10epoch.pth")
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))  # 1表示列 | | | 横着比 ，0 表示行  从上往下纵向着比
