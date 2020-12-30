import numpy as np
import math
import os

import cv2
import torch
import random

import shutil
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)
def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    if rot_y > np.pi:
      rot_y -= 2 * np.pi
    if rot_y < -np.pi:
      rot_y += 2 * np.pi
    return rot_y
def unproject_2d_to_3d1(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth - P[2, 3]
  x = (pt_2d[0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32)
  return pt_3d
def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth + P[2, 3]
  x=(pt_2d[0]*z-P[0, 3]-depth*P[0,2])/P[0,0]
  y = (pt_2d[1] * z - P[1, 3] - depth * P[1, 2]) / P[1, 1]

  pt_3d = np.array([x, y, z], dtype=np.float32)
  return pt_3d
if __name__ == '__main__':
    img2_path='/home/lipeixuan/3ddetectionkitti/data_object_image_2/training/image_2/'
    img3_path = '/home/lipeixuan/3ddetectionkitti/data_object_image_3/training/image_3/'
    calib_path = '/home/lipeixuan/3ddetectionkitti/data_object_calib/training/calib/'
    label_path='/home/lipeixuan/3ddetectionkitti/label/training/label_2/'
    outp='/home/lipeixuan/3ddetectionkitti/kitti_format/'
    test_path='/home/lipeixuan/3ddetectionkitti/data_object_image_2/testing/image_2/'
    image_target_path = outp + 'data/kitti/image/'
    if not os.path.exists(image_target_path):
        os.makedirs(image_target_path)
    test_target_path = outp + 'data/kitti/test/'
    if not os.path.exists(test_target_path):
        os.makedirs(test_target_path)
    calib_target_path = outp + 'data/kitti/calib/'
    if not os.path.exists(calib_target_path):
        os.makedirs(calib_target_path)
    label_target_path = outp + 'data/kitti/label/'
    if not os.path.exists(label_target_path):
        os.makedirs(label_target_path)
    images = os.listdir(test_path)
    for idx in images:
        img_name = os.path.join(test_path, idx)
        shutil.copyfile(img_name, test_target_path + idx)
    images=os.listdir(img2_path)
    for idx in images:
        img_name=os.path.join(img2_path,idx)
        shutil.copyfile(img_name,image_target_path+idx)
    images = os.listdir(img3_path)
    for idx in images:
        img_name = os.path.join(img3_path, idx)
        idx_tar="{:06d}".format( int(float(idx[:6])+7481) )+'.png'
        shutil.copyfile(img_name, image_target_path + idx_tar)
    calibes = os.listdir(calib_path)
    for idx in calibes:
        img_name = os.path.join(calib_path, idx)
        shutil.copyfile(img_name, calib_target_path + idx)
        calibes = os.listdir(calib_path)
    calibes = os.listdir(calib_path)
    for idx in calibes:
        img_name = os.path.join(calib_path, idx)
        idx_tar = "{:06d}".format( int(float(idx[:6])+7481) ) + '.txt'
        shutil.copyfile(img_name, calib_target_path + idx_tar)
    labeles = os.listdir(label_path)
    for idx in labeles:
        img_name = os.path.join(label_path, idx)
        shutil.copyfile(img_name, label_target_path + idx)
        calibes = os.listdir(calib_path)
    labeles = os.listdir(label_path)
    for idx in labeles:
        img_name = os.path.join(label_path, idx)
        idx_tar = "{:06d}".format( int(float(idx[:6])+7481) ) + '.txt'
        shutil.copyfile(img_name, label_target_path + idx_tar)
        # src=os.path.join(data_path,idx)
        # num=int(float(idx[:-4]))
        # num+=num_f
        # frame_id = get_image_index_str(num)+'.png'
        # dis=os.path.join(dis_image,frame_id)
        # os.rename(src,dis)
    # data_path = path + 'velodyne_points/data/'
    # images = os.listdir(data_path)
    # for idx in images:
    #     # shutil.copyfile(calib,calib_dis+idx[:-4]+'.txt')