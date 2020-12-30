"""Detection visualizing.
This file helps visualize 3D detection result in 2D image format
"""

import csv
import time
import argparse
import os
import sys
import numpy as np
import os.path
import cv2
import math as m
import utils.kitti_read as kitti_utils
import matplotlib
import math


def Space2Image(P0, pts3):
    pts2_norm = P0.dot(pts3)
    pts2 = np.array([int(pts2_norm[0] / pts2_norm[2]), int(pts2_norm[1] / pts2_norm[2])])
    return pts2


def Space2Bev(P0, side_range=(-20, 20),
              fwd_range=(0, 70),
              res=0.1):
    x_img = (P0[0] / res).astype(np.int32)
    y_img = (-P0[2] / res).astype(np.int32)

    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.floor(fwd_range[1] / res)) - 1

    return np.array([x_img, y_img])


def vis_lidar_in_bev(pointcloud, width=750, side_range=(-20, 20), fwd_range=(0, 70),
                     min_height=-2.5, max_height=1.5):
    ''' Project pointcloud to bev image for simply visualization

        Inputs:
            pointcloud:     3 x N in camera 2 frame
        Return:
            cv color image

    '''
    res = float(fwd_range[1] - fwd_range[0]) / width
    x_lidar = pointcloud[0, :]
    y_lidar = pointcloud[1, :]
    z_lidar = pointcloud[2, :]

    ff = np.logical_and((z_lidar > fwd_range[0]), (z_lidar < fwd_range[1]))
    ss = np.logical_and((x_lidar > side_range[0]), (x_lidar < side_range[1]))
    indices = np.argwhere(np.logical_and(ff, ss)).flatten()

    x_img = (x_lidar[indices] / res).astype(np.int32)
    y_img = (-z_lidar[indices] / res).astype(np.int32)

    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.floor(fwd_range[1] / res)) - 1

    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    x_img[x_img > x_max - 1] = x_max - 1
    y_img[y_img > y_max - 1] = y_max - 1

    im[:, :] = 255
    im[y_img, x_img] = 100
    im_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    return im_rgb


def vis_create_bev(width=750, side_range=(-20, 20), fwd_range=(0, 70),
                   min_height=-2.5, max_height=1.5):
    ''' Project pointcloud to bev image for simply visualization

        Inputs:
            pointcloud:     3 x N in camera 2 frame
        Return:
            cv color image

    '''
    res = float(fwd_range[1] - fwd_range[0]) / width
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[:, :] = 255
    im_rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    return im_rgb


def vis_box_in_bev(im_bev, pos, dims, orien, width=750, gt=False,score=None,
                   side_range=(-20, 20), fwd_range=(0, 70),
                   min_height=-2.73, max_height=1.27):
    ''' Project 3D bounding box to bev image for simply visualization
        It should use consistent width and side/fwd range input with
        the function: vis_lidar_in_bev

        Inputs:
            im_bev:         cv image
            pos, dim, orien: params of the 3D bounding box
        Return:
            cv color image

    '''
    dim = dims.copy()
    buf = dim.copy()
    # dim[0]=buf[2]
    # dim[2]=buf[0]
    # dim[0]=buf[1]
    # dim[1]=buf[2]
    # dim[2]=buf[0]
    res = float(fwd_range[1] - fwd_range[0]) / width

    R = kitti_utils.E2R(orien, 0, 0)
    pts3_c_o = []
    pts2_c_o = []
    # pts3_c_o.append(pos + R.dot([-dim[0], 0, -dim[2]])/2.0)
    # pts3_c_o.append(pos + R.dot([-dim[0], 0, dim[2]])/2.0) #-x z
    # pts3_c_o.append(pos + R.dot([dim[0], 0, dim[2]])/2.0) # x, z
    # pts3_c_o.append(pos + R.dot([dim[0], 0, -dim[2]])/2.0)
    #
    # pts3_c_o.append(pos + R.dot([0, 0, dim[2]*2/3]))

    pts3_c_o.append(pos + R.dot(np.array([dim[0] / 2., 0, dim[2] / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([dim[0] / 2, 0, -dim[2] / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-dim[0] / 2, 0, -dim[2] / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-dim[0] / 2, 0, dim[2] / 2.0]).T))

    pts3_c_o.append(pos + R.dot([dim[0] / 1.5, 0, 0]))
    pts2_bev = []
    for index in range(5):
        pts2_bev.append(Space2Bev(pts3_c_o[index], side_range=side_range,
                                  fwd_range=fwd_range, res=res))

    if gt is False:
        lineColor3d = (100, 100, 0)
    else:
        lineColor3d = (0, 0, 255)
    if gt == 'next':
        lineColor3d = (255, 0, 0)
    if gt == 'g':
        lineColor3d = (0, 255, 0)
    if gt == 'b':
        lineColor3d = (255, 0, 0)
    if gt == 'n':
        lineColor3d = (5, 100, 100)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[1][0], pts2_bev[1][1]), lineColor3d, 1)
    cv2.line(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[2][0], pts2_bev[2][1]), lineColor3d, 1)
    cv2.line(im_bev, (pts2_bev[2][0], pts2_bev[2][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 1)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 1)

    cv2.line(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, 1)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[4][0], pts2_bev[4][1]), lineColor3d, 1)

    if score is not None:
        show_text(im_bev,pts2_bev[4],score)
    return im_bev


def show_text(img, cor, score):
    txt = '{:.2f}'.format(score)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, txt, (cor[0], cor[1]),
                font, 0.3, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return img
def vis_single_box_in_img(img, calib, pos, dim, theta):
    ''' Project 3D bounding box to rgb frontview for simply visualization

        Inputs:
            img:         cv image
            calib:       FrameCalibrationData
            pos, dim, orien: params of the 3D bounding box
        Return:
            cv color image

    '''

    pts3_c_o = []
    pts2_c_o = []
    # 2D box
    R = kitti_utils.E2R(theta, 0, 0)
    pts3_c_o.append(pos + R.dot([-dim[0], 0, -dim[2]]) / 2.0)
    pts3_c_o.append(pos + R.dot([-dim[0], 0, dim[2]]) / 2.0)  # -x z
    pts3_c_o.append(pos + R.dot([dim[0], 0, dim[2]]) / 2.0)  # x, z
    pts3_c_o.append(pos + R.dot([dim[0], 0, -dim[2]]) / 2.0)

    pts3_c_o.append(pos + R.dot([-dim[0], -dim[1] * 2, -dim[2]]) / 2.0)
    pts3_c_o.append(pos + R.dot([-dim[0], -dim[1] * 2, dim[2]]) / 2.0)  # -x z
    pts3_c_o.append(pos + R.dot([dim[0], -dim[1] * 2, dim[2]]) / 2.0)  # x, z
    pts3_c_o.append(pos + R.dot([dim[0], -dim[1] * 2, -dim[2]]) / 2.0)

    for i in range(8):
        pts2_c_o.append(Space2Image(calib.p2[:, 0:3], pts3_c_o[i]))
        if (pts3_c_o[i][2] < 0):
            return img

    lineColor3d = (0, 200, 0)
    cv2.line(img, (pts2_c_o[0][0], pts2_c_o[0][1]), (pts2_c_o[1][0], pts2_c_o[1][1]), lineColor3d, 1)
    cv2.line(img, (pts2_c_o[1][0], pts2_c_o[1][1]), (pts2_c_o[2][0], pts2_c_o[2][1]), lineColor3d, 1)
    cv2.line(img, (pts2_c_o[2][0], pts2_c_o[2][1]), (pts2_c_o[3][0], pts2_c_o[3][1]), lineColor3d, 1)
    cv2.line(img, (pts2_c_o[0][0], pts2_c_o[0][1]), (pts2_c_o[3][0], pts2_c_o[3][1]), lineColor3d, 1)

    cv2.line(img, (pts2_c_o[4][0], pts2_c_o[4][1]), (pts2_c_o[5][0], pts2_c_o[5][1]), lineColor3d, 1)
    cv2.line(img, (pts2_c_o[5][0], pts2_c_o[5][1]), (pts2_c_o[6][0], pts2_c_o[6][1]), lineColor3d, 1)
    cv2.line(img, (pts2_c_o[6][0], pts2_c_o[6][1]), (pts2_c_o[7][0], pts2_c_o[7][1]), lineColor3d, 1)
    cv2.line(img, (pts2_c_o[7][0], pts2_c_o[7][1]), (pts2_c_o[4][0], pts2_c_o[4][1]), lineColor3d, 1)

    cv2.line(img, (pts2_c_o[4][0], pts2_c_o[4][1]), (pts2_c_o[0][0], pts2_c_o[0][1]), lineColor3d, 1)
    cv2.line(img, (pts2_c_o[5][0], pts2_c_o[5][1]), (pts2_c_o[1][0], pts2_c_o[1][1]), lineColor3d, 1)
    cv2.line(img, (pts2_c_o[6][0], pts2_c_o[6][1]), (pts2_c_o[2][0], pts2_c_o[2][1]), lineColor3d, 1)
    cv2.line(img, (pts2_c_o[7][0], pts2_c_o[7][1]), (pts2_c_o[3][0], pts2_c_o[3][1]), lineColor3d, 1)

    return img


def draw_ellipsoid3D_fasle(ellipsodi_param, ax, color_set='r'):
    xr = abs(ellipsodi_param[1])
    yr = abs(ellipsodi_param[2])
    zr = abs(ellipsodi_param[0])
    xc = ellipsodi_param[3]
    yc = ellipsodi_param[4]
    zc = ellipsodi_param[5]

    # xr=zrr
    # yr=xrr
    # zr=yrr
    #
    # xc=zcr
    # yc=xcr
    # zc=ycr

    rotation = ellipsodi_param[6:9]

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    cor_3d = np.zeros((3, 100, 100))
    cor_3d[0, :, :] = x

    cor_3d[1, :, :] = y

    cor_3d[2, :, :] = z
    R = utiles.eulerAnglesToRotationMatrix([1.2, 1.2, 1.2])  # rotation)
    cor_3dd = cor_3d.copy()
    for i in range(100):
        buf = cor_3dd[:, :, i]
        buf = R.dot(buf)
        cor_3d[:, :, i] = buf

    x = cor_3d[0, :]
    y = cor_3d[1, :]
    z = cor_3d[2, :]

    x = xr * x + xc
    y = yr * y + yc
    z = zr * z + zc

    x_s = z
    y_s = x
    z_s = y
    ax.plot_wireframe(x_s, y_s, z_s, rstride=1, cstride=1, color=color_set)


def draw_ellipsoid3D(ellipsodi_param, ax, color_set='r'):
    xr = abs(ellipsodi_param[1])
    yr = abs(ellipsodi_param[2])
    zr = abs(ellipsodi_param[0])
    xc = ellipsodi_param[3]
    yc = ellipsodi_param[4]
    zc = ellipsodi_param[5]

    rotation = ellipsodi_param[6:9]

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    rotation[1] += math.pi / 2
    R = utiles.eulerAnglesToRotationMatrix(rotation)

    x = xr * x
    y = yr * y
    z = zr * z

    cor_3d = np.zeros((3, 100, 100))
    cor_3d[0, :, :] = x

    cor_3d[1, :, :] = y

    cor_3d[2, :, :] = z
    # cor_3dd = cor_3d.copy()
    # for i in range(100):
    #     for j in range(100):
    #         buf=np.array([x[i,j],y[i,j],z[i,j]])
    #         buf=R.dot(buf)
    #         cor_3d[:,i,j]=buf
    cor_3dd = cor_3d.copy()
    for i in range(100):
        buf = cor_3dd[:, :, i]
        buf = R.dot(buf)
        cor_3d[:, :, i] = buf

    x = cor_3d[0, :, :]

    y = cor_3d[1, :, :]

    z = cor_3d[2, :, :]
    x = x + xc
    y = y + yc
    z = z + zc

    x_s = z
    y_s = x
    z_s = y
    ax.plot_wireframe(x_s, y_s, z_s, rstride=1, cstride=1, color=color_set)


def draw_Box3D(box3d, ax, color_set='b'):
    for box in box3d:
        bbox = box.copy()
        box[0] = bbox[2]
        box[1] = bbox[0]
        box[2] = bbox[1]

    line_order = ([0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                  [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3])

    for k in line_order:
        ax.plot3D(*zip(box3d[k[0]].T, box3d[k[1]].T), lw=1.5, color=color_set)


def draw_Box3DinImage(box2d, img, color_set=(255, 0, 0)):
    line_order = ([0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                  [6, 7], [7, 4], [4, 0], [5, 1], [6, 2], [7, 3])
    for k in line_order:
        cv2.line(img, (int(box2d[k[0]][0]), int(box2d[k[0]][1])), (int(box2d[k[1]][0]), int(box2d[k[1]][1])), color_set,
                 1)

    return img


def Box3D2ellipsoid_param(dim, pos, rotation_y, method='in'):  # lhw

    # x= dim[2]/2
    # y = dim[1]/2
    # z = dim[0]/2
    if method == 'norm':
        x = dim[0] / 2
        y = dim[1] / 2
        z = dim[2] / 2
    if method == 'out':
        x = math.sqrt(2) * dim[0] / 2
        y = math.sqrt(2) * dim[1] / 2
        z = math.sqrt(2) * dim[2] / 2
    if method == 'in':
        x = math.sqrt(dim[0] ** 2 + dim[2] ** 2) / 2
        y = dim[1] / 2
        z = 0
    if method == 'middle':
        x = 0
        y = dim[1] / 2
        z = dim[2] / 2
    if method == 'middle_back':
        x = dim[1] / 4
        y = dim[1] / 2
        z = dim[2] / 2
    if method == 'front' or method == 'back':
        x = 0
        y = dim[1] / 2
        z = dim[2] / 2
    if method == 'left' or method == 'right':
        x = dim[0] / 2
        y = dim[1] / 2
        z = 0

    # lhw
    dif_Ry = math.atan2(dim[2], dim[0])

    cx = pos[0]
    cy = pos[1] - dim[1] / 2
    cz = pos[2]
    R = kitti_utils.E2R(rotation_y, 0, 0)
    if method == 'front':

        if rotation_y > 0:
            cent = np.array(pos) + R.dot(np.array([dim[0] / 2., -dim[1] / 2, 0]).T)
        else:
            cent = np.array(pos) + R.dot(np.array([-dim[0] / 2., -dim[1] / 2, 0]).T)
        cx = cent[0]
        cy = cent[1]
        cz = cent[2]
    if method == 'back':

        if rotation_y > 0:
            cent = np.array(pos) + R.dot(np.array([-dim[0] / 2., -dim[1] / 2, 0]).T)
        else:
            cent = np.array(pos) + R.dot(np.array([dim[0] / 2., -dim[1] / 2, 0]).T)
        cx = cent[0]
        cy = cent[1]
        cz = cent[2]
    if method == 'left':
        if rotation_y > 0:
            cent = np.array(pos) + R.dot(np.array([0, -dim[1] / 2, -dim[2] / 2]).T)
        else:
            cent = np.array(pos) + R.dot(np.array([0, -dim[1] / 2, dim[2] / 2]).T)
        cx = cent[0]
        cy = cent[1]
        cz = cent[2]
    if method == 'right':
        if rotation_y > 0:
            cent = np.array(pos) + R.dot(np.array([0, -dim[1] / 2, dim[2] / 2]).T)
        else:
            cent = np.array(pos) + R.dot(np.array([0, -dim[1] / 2, -dim[2] / 2]).T)
        cx = cent[0]
        cy = cent[1]
        cz = cent[2]
    if method == 'middle_back':
        if rotation_y > 0:
            cent = np.array(pos) + R.dot(np.array([dim[0] / 4., -dim[1] / 2, 0]).T)
        else:
            cent = np.array(pos) + R.dot(np.array([-dim[0] / 4., -dim[1] / 2, 0]).T)
        cx = cent[0]
        cy = cent[1]
        cz = cent[2]
    rotation = [0, rotation_y, 0]
    param = [x, y, z, cx, cy, cz] + rotation
    return param, dif_Ry


def Box3D2ellipsoid_param_front(dim, pos, rotation_y, method='in'):  # lhw

    # x= dim[2]/2
    # y = dim[1]/2
    # z = dim[0]/2
    if method == 'out':
        x = math.sqrt(2) * dim[0] / 2
        y = math.sqrt(2) * dim[1] / 2
        z = math.sqrt(2) * dim[2] / 2
    if method == 'in':
        x = math.sqrt(dim[0] ** 2 + dim[2] ** 2) / 2
        y = dim[1] / 2
        z = 0
    if method == 'middle':
        x = 0
        y = dim[1] / 2
        z = dim[2] / 2
    # lhw
    dif_Ry = math.atan2(dim[2], dim[0])

    cx = pos[0]
    cy = pos[1] - dim[1] / 2
    cz = pos[2]

    rotation = [0, rotation_y, 0]
    param = [x, y, z, cx, cy, cz] + rotation
    return param, dif_Ry


def draw_ellipse2D(param, img, color_set=(100, 0, 100)):
    img = img.copy()
    # param=[100,100,2,2,np.array([[0,1],[1,0]])]
    N = 200.
    theta = np.linspace(0, 2 * np.pi, N)

    x = param[0] * np.cos(theta)
    y = param[1] * np.sin(theta)

    cor_2d = np.zeros((2, 200))
    cor_2d[0, :] = x
    cor_2d[1, :] = y
    cor_2d = param[4].dot(cor_2d)
    cor_2d[0, :] = cor_2d[0, :] + param[2]
    cor_2d[1, :] = cor_2d[1, :] + param[3]
    cor_2d = cor_2d.T.astype(int)
    cv2.drawContours(img, [cor_2d], 0, color_set, 1)
    return img


def draw_box2D(box, image, color_set=(255, 0, 0)):
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_set, 1)
    return image


def vis_points_in_bev(im_bev, points, width=750, gt=False,
                      side_range=(-20, 20), fwd_range=(0, 70),
                      min_height=-2.73, max_height=1.27):
    res = float(fwd_range[1] - fwd_range[0]) / width
    pts3_c_o = points[:, :4]
    pts3_c_o = pts3_c_o[:3, :]
    pts3_c_o = pts3_c_o.T
    pts2_bev = []
    for index in range(4):
        pts2_bev.append(Space2Bev(pts3_c_o[index], side_range=side_range,
                                  fwd_range=fwd_range, res=res))

    if gt is False:
        lineColor3d = (100, 100, 0)
    else:
        lineColor3d = (0, 0, 255)
    if gt == 'next':
        lineColor3d = (255, 0, 0)
    if gt == 'g':
        lineColor3d = (0, 255, 0)
    if gt == 'b':
        lineColor3d = (255, 0, 0)
    if gt == 'n':
        lineColor3d = (5, 100, 100)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[1][0], pts2_bev[1][1]), lineColor3d, 2)
    cv2.line(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[2][0], pts2_bev[2][1]), lineColor3d, 2)
    cv2.line(im_bev, (pts2_bev[2][0], pts2_bev[2][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 2)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 2)

    return im_bev


def vis_points_circle_in_bev(im_bev, points, width=750, gt=False,
                             side_range=(-20, 20), fwd_range=(0, 70),
                             min_height=-2.73, max_height=1.27):
    res = float(fwd_range[1] - fwd_range[0]) / width
    pts3_c_o = points[:, :4]
    pts3_c_o = pts3_c_o[:3, :]
    pts3_c_o = pts3_c_o.T
    pts2_bev = []
    for index in range(4):
        pts2_bev.append(Space2Bev(pts3_c_o[index], side_range=side_range,
                                  fwd_range=fwd_range, res=res))

    if gt is False:
        lineColor3d = (100, 100, 0)
    else:
        lineColor3d = (0, 0, 255)
    if gt == 'next':
        lineColor3d = (255, 0, 0)
    if gt == 'g':
        lineColor3d = (0, 255, 0)
    if gt == 'b':
        lineColor3d = (255, 0, 0)
    if gt == 'n':
        lineColor3d = (5, 100, 100)
    cv2.circle(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), 2, (255, 0, 0))
    cv2.circle(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), 2, (0, 255, 0))
    cv2.circle(im_bev, (pts2_bev[2][0], pts2_bev[2][1]), 2, (0, 0, 255))
    cv2.circle(im_bev, (pts2_bev[3][0], pts2_bev[3][1]), 2, (255, 255, 0))
    return im_bev


def vis_all_points_circle_in_bev(im_bev, points, width=750, gt=False,
                                 side_range=(-20, 20), fwd_range=(0, 70),
                                 min_height=-2.73, max_height=1.27):
    res = float(fwd_range[1] - fwd_range[0]) / width
    pts3_c_o = points
    pts3_c_o = pts3_c_o.T
    pts2_bev = []
    for index in range(pts3_c_o.shape[0]):
        pts2_bev.append(Space2Bev(pts3_c_o[index], side_range=side_range,
                                  fwd_range=fwd_range, res=res))

    if gt == 'c':
        lineColor3d = (100, 100, 0)
    if gt == 'r':
        lineColor3d = (0, 0, 255)
    if gt == 'g':
        lineColor3d = (0, 255, 0)
    if gt == 'b':
        lineColor3d = (255, 0, 0)
    if gt == 'n':
        lineColor3d = (5, 100, 100)
    for i in range(len(pts2_bev)):
        cv2.circle(im_bev, (pts2_bev[i][0], pts2_bev[i][1]), 2, lineColor3d)
    # cv2.circle(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), 2, (0,255,0))
    # cv2.circle(im_bev, (pts2_bev[2][0], pts2_bev[2][1]), 2, (0,0,255))
    # cv2.circle(im_bev, (pts2_bev[3][0], pts2_bev[3][1]), 2, (255,255,0))
    return im_bev


def vis_points_in_image(points, img,score=None):
    colors_hp = [(255, 0, 0), (0, 255, 255), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 255), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0),
                 (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                 (255, 0, 0), (0, 255, 0), (0, 0, 255),
                 (255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)
                 ]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                  [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    points=points.astype(np.int)
    for i in range(points.shape[1]):
        cv2.circle(img, (int(points[0, i]), int(points[1, i])), 3, colors_hp[i],-1)
        show_text(img,(int(points[0, i])+3, int(points[1, i])),score[i])
    for j, e in enumerate(edges):
      #if points[e].min() > 0:
        cv2.line(img, (points[0,e[0]], points[1,e[0]]),
                      (points[0,e[1]], points[1,e[1]]), (0,255,0), 1,lineType=cv2.LINE_AA)
    # for i in range(points.shape[1]):
    #     cv2.circle(img, (int(points[0, i]), int(points[1, i])), 3, colors_hp[i],-1)
    return img


def vis_pointcloudin_bev(im_bev, points, width=750, gt=False,
                         side_range=(-20, 20), fwd_range=(0, 70),
                         min_height=-2.73, max_height=1.27):
    res = float(fwd_range[1] - fwd_range[0]) / width
    pts3_c_o = points[:, :4]
    pts3_c_o = pts3_c_o[:3, :]
    pts3_c_o = pts3_c_o.T
    pts2_bev = []
    for index in range(4):
        pts2_bev.append(Space2Bev(pts3_c_o[index], side_range=side_range,
                                  fwd_range=fwd_range, res=res))

    if gt is False:
        lineColor3d = (100, 100, 0)
    else:
        lineColor3d = (0, 0, 255)
    if gt == 'next':
        lineColor3d = (255, 0, 0)
    if gt == 'g':
        lineColor3d = (0, 255, 0)
    if gt == 'b':
        lineColor3d = (255, 0, 0)
    if gt == 'n':
        lineColor3d = (5, 100, 100)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[1][0], pts2_bev[1][1]), lineColor3d, 2)
    cv2.line(im_bev, (pts2_bev[1][0], pts2_bev[1][1]), (pts2_bev[2][0], pts2_bev[2][1]), lineColor3d, 2)
    cv2.line(im_bev, (pts2_bev[2][0], pts2_bev[2][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 2)
    cv2.line(im_bev, (pts2_bev[0][0], pts2_bev[0][1]), (pts2_bev[3][0], pts2_bev[3][1]), lineColor3d, 2)

    return im_bev