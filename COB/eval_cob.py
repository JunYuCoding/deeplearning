#!/usr/bin/env python3
import scipy.io
import glob
import os
from os.path import join as pjoin
import cv2
import configargparse
import numpy as np
from models.cobnet import CobNet
import torch
from skimage.io import imsave, imread
from imgaug import augmenters as iaa
from utils.augmenters import Normalize, rescale_augmenter
import matplotlib.pyplot as plt
if __name__ == "__main__":

    p = configargparse.ArgParser()
    # python eval_cob.py --model-path  runs/cob/checkpoints/cp_fs.pth.tar --in-path test/HED-BSDS/test/ --out-path runs/result

    # 0824 python eval_cob.py --model-path  runs/cob_bsds/checkpoints/cp_or.pth.tar --in-path test/HED-BSDS/test/ --out-path runs/result_bsds_user_or

    '''
    the first line is origin author
    # python eval_cob.py --model-path  runs/cob/checkpoints/cp_or.pth.tar --in-path BSDS500/data/images/test/ --out-path runs/evalresult_resnet_pascal  
    # python eval_cob.py --model-path  runs/cob_backbone_resnet_bsds500/checkpoints/cp_or.pth.tar --in-path BSDS500/data/images/test/ --out-path runs/evalresult_resnet_bsds500  ok
    # python eval_cob.py --model-path  runs/cob_backbone_vgg1_pascal/checkpoints/cp_or.pth.tar --in-path BSDS500/data/images/test/ --out-path runs/evalresult_vgg1_pascal       ok
    # python eval_cob.py --model-path  runs/cob_backbone_vgg1_bsds500/checkpoints/cp_or.pth.tar --in-path BSDS500/data/images/test/ --out-path runs/evalresult_vgg1_bsds500    ok
    # python eval_cob.py --model-path  runs/cob_backbone_vgg2_pascal/checkpoints/cp_or.pth.tar --in-path BSDS500/data/images/test/ --out-path runs/evalresult_vgg2_pascal  ok
    # python eval_cob.py --model-path  runs/cob_backbone_vgg2_bsds500/checkpoints/cp_or.pth.tar --in-path BSDS500/data/images/test/ --out-path runs/evalresult_vgg2_bsds500  ok

    '''


    p.add('--model-path', required=True,default='runs/cob/checkpoints/cp_fs.pth.tar')
    p.add('--in-path', required=True,default='test/HED-BSDS/test/')
    p.add('--out-path', required=True,default='runs/result')
    cfg = p.parse_args()

    exts = ['jpg', 'jpeg', 'png']
    if os.path.isdir(cfg.in_path):
        im_paths = []
        for ext in exts:
            im_paths.extend(glob.glob(pjoin(cfg.in_path, '*.' + ext)))
        print('found {} images to process'.format(len(im_paths)))
        if (not os.path.exists(cfg.out_path)):
            os.makedirs(cfg.out_path)
        out_paths = [
            pjoin(cfg.out_path,
                  os.path.split(imp)[-1]) for imp in im_paths
        ]

    elif os.path.isfile(cfg.in_path):
        assert (not os.path.isdir(
            cfg.out)), 'when in is file, give a file name for out'

    model = CobNet()
    state_dict = torch.load(cfg.model_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    # print(model.orientations)
    normalize = iaa.Sequential([
        rescale_augmenter,
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for i, (imp, outp) in enumerate(zip(im_paths, out_paths)):
        print('[{}/{}] {} -> {}'.format(i + 1, len(im_paths), imp, outp))
        im_orig = imread(imp)
        base_name = outp.split('/')[2].split('.')[0]
        im = normalize(image=im_orig)
        im = torch.from_numpy(np.moveaxis(im, -1, 0)).unsqueeze(0).float()
        res = model(im)

        dicts = {}
        plt.subplot(241)
        plt.imshow(im_orig)
        plt.title('origin')
        dicts['im'] = im_orig
        plt.xticks([]), plt.yticks([])
        plt.subplot(242)
        plt.imshow(res['y_fine'].sigmoid().squeeze().cpu().detach().numpy())
        dicts['y_fine'] = res['y_fine'].sigmoid().squeeze().cpu().detach().numpy()
        plt.title('y_fine')
        plt.xticks([]), plt.yticks([])
        plt.subplot(243)
        plt.imshow(res['y_coarse'].sigmoid().squeeze().cpu().detach().numpy())
        dicts['y_coarse'] = res['y_coarse'].sigmoid().squeeze().cpu().detach().numpy()
        plt.title('y_coarse')
        plt.xticks([]), plt.yticks([])
        for i in range(4,9):
            plt.subplot(240+i)
            plt.imshow(res['late_sides'][i-4].sigmoid().squeeze().cpu().detach().numpy())
            plt.title('late_side{}'.format(i-4))
            plt.xticks([]), plt.yticks([])

        # late_sides = np.concatenate([res['late_sides'][i].sigmoid().squeeze().cpu().detach().numpy() for i in range(4)])
        shape = res['late_sides'][3].sigmoid().squeeze().cpu().detach().numpy().shape
        # print(res['late_sides'][3].sigmoid().squeeze().cpu().detach().numpy().shape)
        # late_sides = np.zeros((5, 512, 768))
        late_sides = np.zeros((5, shape[0], shape[1]))
        for i in range(5):
            late_sides[i, :, :] = res['late_sides'][i].sigmoid().squeeze().cpu().detach().numpy()
        dicts['late_sides'] = late_sides
        plt.savefig(pjoin(cfg.out_path, base_name +"_origin_fine_coarse_lateside0-4.jpg"))
        plt.figure()
        for i in range(1,9):
            plt.subplot(240 + i)
            plt.imshow(res['orientations'][i - 1].sigmoid().squeeze().cpu().detach().numpy())
            plt.title('orientation{}'.format(i-1))
            plt.xticks([]), plt.yticks([])
        plt.savefig(pjoin(cfg.out_path,base_name +"_orientations0-7.jpg"))
        plt.show()
        # orientations = np.concatenate([res['orientations'][i].sigmoid().squeeze().cpu().detach().numpy() for i in range(8)])
        shape_or = res['orientations'][3].sigmoid().squeeze().cpu().detach().numpy().shape
        orientations = np.zeros((8, shape_or[0], shape_or[1]))
        for i in range(8):
            orientations[i, :, :] = res['orientations'][i].sigmoid().squeeze().cpu().detach().numpy()
        # print("orientations",orientations.shape)
        dicts['orientations'] = orientations
        scipy.io.savemat(pjoin(cfg.out_path,base_name +"_result.mat"),dicts)
        # print(base_name +"_done ")
        # break
        # draw origin y_fine y_coarse
        # plt.subplot(131)
        # plt.imshow(im_orig)
        # plt.subplot(132)
        # plt.imshow(res['y_fine'].sigmoid().squeeze().cpu().detach().numpy())
        # plt.subplot(133)
        # plt.imshow(res['y_coarse'].sigmoid().squeeze().cpu().detach().numpy())
        # plt.savefig(outp)
        # plt.show()



