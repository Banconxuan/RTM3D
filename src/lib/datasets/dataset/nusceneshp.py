from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import torch.utils.data as data
class NUSCENESHP(data.Dataset):
    num_classes = 10
    num_joints = 9
    default_resolution = [896, 1600]

    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    flip_idx = [[0, 1], [2, 3], [4, 5], [6, 7]]

    def __init__(self, opt, split):
        super(KITTIHP, self).__init__()
        self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [4, 6], [3, 5], [5, 6],
                      [5, 7]]

        self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8]
        self.data_dir = os.path.join(opt.data_dir, 'kitti')
        self.img_dir= os.path.join(self.data_dir,'image')
        self.calib_dir = os.path.join(self.data_dir,'calib')
        if split == 'test':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'image_info_test-dev2017.json').format(split)
        else:
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                'kitti_{}_nuscenes.json').format(split)
        self.max_objs = 32
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt
        self.alpha_in_degree = False
        print('==> initializing kitti{} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        image_ids = self.coco.getImgIds()
        if split == 'train':
            self.images = []
            for img_id in image_ids:
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if len(idxs) > 0:
                    self.images.append(img_id)
        else:
            self.images = image_ids
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def __len__(self):
        return self.num_samples

