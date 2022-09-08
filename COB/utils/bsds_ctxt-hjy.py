import torch
import cv2
import numpy as np
import os.path as osp


class BSDS_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_imgs='data/HED-BSDS',root_segs=None,split='train', transform=False):
        super(BSDS_Dataset, self).__init__()
        self.root = root_imgs
        self.split = split
        self.splits = ['train', 'test']
        self.transform = transform
        if self.split == 'train':
            self.file_list = osp.join(self.root, 'bsds_pascal_train_pair.lst')
        elif self.split == 'val':
            self.file_list = osp.join(self.root, 'bsds_pascal_train_pair.lst') # val在lst文件内
        elif self.split == 'test':
            self.file_list = osp.join(self.root, 'test.lst')
        else :
            raise ValueError('Invalid split type!')
        with open(self.file_list, 'r') as f:
            self.file_list = f.readlines()
        self.mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        # if self.split == 'train':
        #     img_file, label_file = self.file_list[index].split()
        #     # label_file  train/aug_gt/0.0_1_0/100075.png
        #     # print(label_file.split('/')[3].split('.')[0],"label_file_base_name")
        #     base_name = label_file.split('/')[3].split('.')[0]
        #     label = cv2.imread(osp.join(self.root, label_file), 0)
        #     label = np.array(label, dtype=np.float32)
        #     label = label[np.newaxis, :, :]
        #     label[label == 0] = 0
        #     label[np.logical_and(label > 0, label < 127.5)] = 2
        #     label[label >= 127.5] = 1
        #     label = label[0] # 只取w h维度,原来是1 w h
        #     # print(label.shape,'shape')
        # else:
        #     img_file = self.file_list[index].rstrip() # test/100007.jpg
        img_file, label_file = self.file_list[index].split()
        # label_file  train/aug_gt/0.0_1_0/100075.png
        # print(label_file.split('/')[3].split('.')[0],"label_file_base_name")
        base_name = label_file.split('/')[3].split('.')[0]
        label = cv2.imread(osp.join(self.root, label_file), 0)
        label = np.array(label, dtype=np.float32)
        label = label[np.newaxis, :, :]
        label[label == 0] = 0
        label[np.logical_and(label > 0, label < 127.5)] = 2
        label[label >= 127.5] = 1
        label = label[0] # 只取w h维度,原来是1 w h
        img = cv2.imread(osp.join(self.root, img_file))
        img = np.array(img, dtype=np.float32)
        # img = (img - self.mean).transpose((2, 0, 1)) # 不用转，Pascal 是 0 ，1 2 w h c
        img = (img - self.mean)
        # if self.split == 'train':
        #     # return {'image': img, 'labels': label, 'base_name': base_name}
        #     # return img, label
        # else:
        #     return img
        return {'image': img, 'labels': label, 'base_name': base_name}
if __name__ == "__main__":
    bsds = BSDS_Dataset(root_imgs='/root/hjy_pro/RCF/data/HED-BSDS/',split='val')
    bsds.__getitem__(0)
