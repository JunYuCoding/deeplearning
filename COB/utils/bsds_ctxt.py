import torch
import cv2
import numpy as np
import os.path as osp

import os
from os.path import join as pjoin
from skimage.io import imsave, imread
import scipy.io as io
import glob

class BSDS500Loader:
    """Data loader for the BSDS500 Berkeley Segmentation dataset.

    A total of three data splits are provided for working with the BSDS500 data:
        train: The original BSDS500 training data - 200 images
        val: The original BSDS500 validation data - 100 images
        test: The original BSDS500 validation data - 200 images
    """
    def __init__(self, root_imgs, root_segs, split='train'):
        self.root_imgs = root_imgs  # BSDS500/data/images
        self.root_segs = root_segs  # BSDS500/data/groundTruth
        self.splits = ['train', 'val', 'test']
        self.split = split

        # read BSDS train, val and test  sets
        self.base_names = dict()
        for dt in self.splits:
            self.base_names[dt] = [os.path.splitext(os.path.basename(f))[0]
                                   for f in glob.glob(pjoin(self.root_imgs, dt, '*.jpg'))]
        # print(self.base_names)

    def __len__(self):
        return len(self.base_names[self.split])

    def __getitem__(self, index):
        base_name = self.base_names[self.split][index]
        im_path = pjoin(self.root_imgs, self.split, base_name + '.jpg')
        lbl_path = pjoin(self.root_segs, self.split, base_name + '.mat')

        im = imread(im_path)
        data = io.loadmat(lbl_path)
        lbl = data['groundTruth']

        # add the Boundaries together for BSDS500 data
        # find the maximum class segmentation
        nlbl = np.argmax([np.max(lbl[0,i]['Segmentation'][0,0])
                for i in np.arange(lbl.size)])
        bdy = lbl[0, nlbl]['Segmentation'][0, 0] # nlbl = 3
        lbl = bdy

        return {'image': im, 'labels': lbl, 'base_name': base_name}

if __name__ == "__main__":
    root_path = '/root/hjy_pro/COB'
    dl = BSDS500Loader(root_imgs=pjoin(root_path, 'BSDS500', 'data/images'),
                       root_segs=pjoin(root_path, 'BSDS500', 'data/groundTruth'))
    dl.__getitem__(1)
