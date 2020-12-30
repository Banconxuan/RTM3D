from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import cv2
nuscenes = False

DATA_PATH = './kitti_format/data/kitti/'
if nuscenes:
    DATA_PATH = './kitti_format/data/nuscenes/'
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os
import math
SPLITS = ['train1']
import _init_paths
from utils.ddd_utils import compute_box_3d, project_to_image, project_to_image3,alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''


def _bbox_to_coco_bbox(bbox):
    return [(bbox[0]), (bbox[1]),
            (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]


def read_clib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 2:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib
def read_clib3(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 3:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib
def read_clib0(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 0:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib
cats = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Person_sitting',
        'Tram', 'Misc', 'DontCare']
det_cats=['Car', 'Pedestrian', 'Cyclist']
if nuscenes:
    cats = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
            'traffic_cone', 'barrier']
    det_cats = cats
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
# cat_info = [{"name": "pedestrian", "id": 1}, {"name": "vehicle", "id": 2}]
F = 721
H = 384  # 375
W = 1248  # 1242
EXT = [45.75, -0.34, 0.005]
CALIB = np.array([[F, 0, W / 2, EXT[0]], [0, F, H / 2, EXT[1]],
                  [0, 0, 1, EXT[2]]], dtype=np.float32)

cat_info = []
for i, cat in enumerate(cats):
    cat_info.append({'name': cat, 'id': i + 1})

for SPLIT in SPLITS:
    image_set_path = os.path.join(DATA_PATH,"image/")
    ann_dir = os.path.join(DATA_PATH,"label/")
    calib_dir = os.path.join(DATA_PATH,"calib/")
    splits = ['train','val']
    if nuscenes:
        splits = ['train_nuscenes', 'val_nuscenes']
    # splits = ['trainval', 'test']
    calib_type = {'train': 'training', 'val': 'training', 'trainval': 'training',
                  'test': 'testing', 'train_stereo': 'training'}
    for split in splits:
        ret = {'images': [], 'annotations': [], "categories": cat_info}
        image_set = open(DATA_PATH + '{}.txt'.format(split), 'r')
        image_to_id = {}
        for line in image_set:
            if line[-1] == '\n':
                line = line[:-1]
            image_id = int(line)
            calib_path = calib_dir  + '{}.txt'.format(line)

            calib0 = read_clib0(calib_path)
            if image_id>7480 and image_id<14962:
                calib = read_clib3(calib_path)
            else:
                calib = read_clib(calib_path)

            image_info = {'file_name': '{}.png'.format(line),
                          'id': int(image_id),
                          'calib': calib.tolist()}
            ret['images'].append(image_info)
            if split == 'test':
                continue
            ann_path = ann_dir + '{}.txt'.format(line)
            # if split == 'val':
            #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
            anns = open(ann_path, 'r')
            for ann_ind, txt in enumerate(anns):
                tmp = txt[:-1].split(' ')
                cat_id = cat_ids[tmp[0]]
                truncated = int(float(tmp[1]))
                occluded = int(tmp[2])
                alpha = float(tmp[3])
                dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
                location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
                rotation_y = float(tmp[14])
                num_keypoints = 0
                box_2d_as_point=[0]*27
                bbox=[0.,0.,0.,0.]
                calib_list = np.reshape(calib, (12)).tolist()
                if tmp[0] in det_cats:
                    image = cv2.imread(os.path.join(image_set_path, image_info['file_name']))
                    bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
                    box_3d = compute_box_3d(dim, location, rotation_y)
                    box_2d_as_point,vis_num,pts_center = project_to_image(box_3d, calib,image.shape)
                    box_2d_as_point=np.reshape(box_2d_as_point,(1,27))
                    #box_2d_as_point=box_2d_as_point.astype(np.int)
                    box_2d_as_point=box_2d_as_point.tolist()[0]
                    num_keypoints=vis_num

                    off_set=(calib[0,3]-calib0[0,3])/calib[0,0]
                    location[0] += off_set###################################################confuse
                    alpha = rotation_y - math.atan2(pts_center[0, 0] - calib[0, 2], calib[0, 0])
                    ann = {'segmentation': [[0,0,0,0,0,0]],
                           'num_keypoints':num_keypoints,
                           'area':1,
                           'iscrowd': 0,
                           'keypoints': box_2d_as_point,
                           'image_id': image_id,
                           'bbox': _bbox_to_coco_bbox(bbox),
                           'category_id': cat_id,
                           'id': int(len(ret['annotations']) + 1),
                           'dim': dim,
                           'rotation_y': rotation_y,
                           'alpha': alpha,
                           'location':location,
                           'calib':calib_list,
                            }
                    ret['annotations'].append(ann)
        print("# images: ", len(ret['images']))
        print("# annotations: ", len(ret['annotations']))
        # import pdb; pdb.set_trace()
        out_path = '{}annotations/kitti_{}.json'.format(DATA_PATH, split)
        json.dump(ret, open(out_path, 'w'))

