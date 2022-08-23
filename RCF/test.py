import os
import numpy as np
from PIL import Image

import torchvision

import matplotlib
from torch.utils.data import DataLoader

from RCF.data_loader import BSDS_RCFLoader
from RCF.models import RCF

matplotlib.use('Agg')

from os.path import join, isdir, splitext

import torch

import cv2
import torchvision.transforms
from torchvision import transforms
#
# lb = cv.imread('D:\\work\pytorch_learn\\RCF\\data\\HED-BSDS\\train\\aug_gt\\0.0_1_0\\2092.png')
# w,h = lb.shape[0],lb.shape[1]
# count = 0
#
# for i in range(w):
#     for j in range(h):
#         if(lb[i][j][2]==1):
#             count+=1
# count
# cur = torch.tensor(lb)
# res = torch.eq(cur, 1).float()
# res
#
# dataset
test_dataset = BSDS_RCFLoader(root="data/HED-BSDS", split="test")
test_loader = DataLoader(
    test_dataset, batch_size=1,
    num_workers=8, drop_last=True, shuffle=False)
import  argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--testimg', help='test img name', default=None)
args = parser.parse_args()
name = "5096"
def test(model, test_loader, epoch, save_dir):
    model.eval()

    if not isdir(save_dir):
        os.makedirs(save_dir)

    image = cv2.imread("./data/HED-BSDS/test/{}.jpg".format(name))
    h,w = image.shape[0],image.shape[1]
    # cv2.imshow("img", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    trans = transforms.ToTensor()
    image = trans(image)
    image = torch.reshape(image, (1, 3, h, w))
    print(image.shape)
    _,_, H, W = image.shape
    results = model(image)
    print(results)
    result = torch.squeeze(results[-1].detach()).cpu().numpy()
    print(result)
    results_all = torch.zeros((len(results), 1, H, W))
    for i in range(len(results)):
        results_all[i, 0, :, :] = results[i]
    # filename = splitext(test_list[idx])[0]
    filename = name
    # 可直接将tensor保存为图片，若tensor在cuda上也会移到CPU中进行保存。
    aa = 1 - results_all
    torchvision.utils.save_image(1 - results_all, join(save_dir, "%s.jpg" % filename))
    result = Image.fromarray((result * 255).astype(np.uint8))
    result.save(join(save_dir, "%s.png" % filename))


if __name__ == "__main__":
    model = RCF()
    checkpoint = torch.load("./bsds500_pascal_model.pth",map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(checkpoint)
    # print(model)
    test(model,test_loader,3,'test_dir')
