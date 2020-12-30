import numpy as np
import csv
import time
import os
import sys
import os.path
import math as m
import shutil
import cv2


# process all the data in camera 2 frame, the kitti raw data is on camera 0 frame
# ------------------------------------------------------ Class define ------------------------------------------------------#
class Box2d:
    def __init__(self):
        self.box = []  # left, top, right bottom in 2D image
        self.keypoints = []  # holds the u coordinates of 4 keypoints, -1 denotes the invisible one
        self.visible_left = 0  # The left side is visible (not occluded) by other object
        self.visible_right = 0  # The right side is visible (not occluded) by other object


class KittiObject:
    def __init__(self):
        self.cls = ''  # Car, Van, Truck
        self.truncate = 0  # float 0(non-truncated) - 1(totally truncated)
        self.occlusion = 0  # integer 0, 1, 2, 3
        self.alpha = 0  # viewpoint angle -pi - pi
        self.Box2D = []  # Box2d list, default order: box_left, box_right, box_merge
        self.Box3D_in_image2 = []
        self.Box3D_in_image3 = []
        self.pos = []  # x, y, z in cam2 frame
        self.dim = []  # width(x), height(y), length(z)
        self.orientation = 0  # [-pi - pi]
        self.R = []  # rotation matrix in cam2 frame
        self.Box3D = []
        self.theta = 0
        self.data_line_num = 0  # create data
        self.score = 0
        self.baseline_23 = 0


class FrameCalibrationData:
    '''Frame Calibration Holder
        p0-p3      Camera P matrix. Contains extrinsic 3x4
                   and intrinsic parameters.
        r0_rect    Rectification matrix, required to transform points 3x3
                   from velodyne to camera coordinate frame.
        tr_velodyne_to_cam0     Used to transform from velodyne to cam 3x4
                                coordinate frame according to:
                                Point_Camera = P_cam * R0_rect *
                                                Tr_velo_to_cam *
                                                Point_Velodyne.
    '''

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p2_3 = []
        self.r0_rect = []
        self.t_cam2_cam0 = []
        self.tr_velodyne_to_cam0 = []
        self.p2_0 = []
        self.p3_0 = []


# ------------------------------------------------------ Math opreation ------------------------------------------------------#
def E2R(Ry, Rx, Rz):
    '''Combine Euler angles to the rotation matrix (right-hand)

        Inputs:
            Ry, Rx, Rz : rotation angles along  y, x, z axis
                         only has Ry in the KITTI dataset
        Returns:
            3 x 3 rotation matrix

    '''
    R_yaw = np.array([[m.cos(Ry), 0, m.sin(Ry)],
                      [0, 1, 0],
                      [-m.sin(Ry), 0, m.cos(Ry)]])
    R_pitch = np.array([[1, 0, 0],
                        [0, m.cos(Rx), -m.sin(Rx)],
                        [0, m.sin(Rx), m.cos(Rx)]])
    # R_roll = np.array([[[m.cos(Rz), -m.sin(Rz), 0],
    #                    [m.sin(Rz), m.cos(Rz), 0],
    #                    [ 0,         0 ,     1]])
    return (R_pitch.dot(R_yaw))


def Space2Image(P0, pts3):
    ''' Project a 3D point to the image

        Inputs:
            P0 : Camera intrinsic matrix 3 x 4
            pts3 : 4-d homogeneous coordinates
        Returns:
            image uv coordinates

    '''
    pts2_norm = P0.dot(pts3)
    pts2 = np.array([(pts2_norm[0] / pts2_norm[2]), (pts2_norm[1] / pts2_norm[2])])
    return pts2


def pixel2cam(point2d, P):
    new_p = point2d.copy()
    new_p[0, :] = (point2d[0, :] - P[0, 2]) / P[0, 0]
    new_p[1, :] = (point2d[1, :] - P[1, 2]) / P[1, 1]
    return new_p


def NormalizeVector(P):
    return np.append(P, [1])


# ------------------------------------------------------ Data reading ------------------------------------------------------#

def read_obj_calibration(CALIB_PATH):
    ''' Reads in Calibration file from Kitti Dataset.

        Inputs:
        CALIB_PATH : Str PATH of the calibration file.

        Returns:
        frame_calibration_info : FrameCalibrationData
                                Contains a frame's full calibration data.
        ^ z        ^ z                                      ^ z         ^ z
        | cam2     | cam0                                   | cam3      | cam1
        |-----> x  |-----> x                                |-----> x   |-----> x

    '''
    frame_calibration_info = FrameCalibrationData()

    data_file = open(CALIB_PATH, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    # based on camera 0
    frame_calibration_info.p0 = p_all[0]
    frame_calibration_info.p1 = p_all[1]
    frame_calibration_info.p2 = p_all[2]
    frame_calibration_info.p3 = p_all[3]

    frame_calibration_info.p2_0 = np.copy(p_all[2])
    E = np.identity(3)
    frame_calibration_info.p2_0[:3, :3] = E
    frame_calibration_info.p3_0 = np.copy(p_all[3])
    E = np.identity(3)
    frame_calibration_info.p3_0[:3, :3] = E

    frame_calibration_info.p0_2 = np.copy(frame_calibration_info.p2_0)
    frame_calibration_info.p0_2[:, 3] = - frame_calibration_info.p0_2[:, 3]
    frame_calibration_info.p0_3 = np.copy(frame_calibration_info.p3_0)
    frame_calibration_info.p0_3[:, 3] = - frame_calibration_info.p0_3[:, 3]
    # based on camera 2
    frame_calibration_info.p2_2 = np.copy(p_all[2])
    frame_calibration_info.p2_2[0, 3] = frame_calibration_info.p2_2[0, 3] - frame_calibration_info.p2[0, 3]

    frame_calibration_info.p2_3 = np.copy(p_all[3])
    frame_calibration_info.p2_3[0, 3] = frame_calibration_info.p2_3[0, 3] - frame_calibration_info.p2[0, 3]

    frame_calibration_info.t_cam2_cam0 = np.zeros(3)
    frame_calibration_info.t_cam2_cam0[0] = (frame_calibration_info.p2[0, 3] - frame_calibration_info.p0[0, 3]) / \
                                            frame_calibration_info.p2[0, 0]
    frame_calibration_info.baseline_23 = abs(frame_calibration_info.p2[0, 3]) + abs(frame_calibration_info.p3[0, 3])

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calibration_info.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calibration_info.tr_velodyne_to_cam0 = np.reshape(tr_v2c, (3, 4))

    return frame_calibration_info


def read_obj_data(LABEL_PATH, calib=None, im_shape=None):
    '''Reads in object label file from Kitti Object Dataset.

        Inputs:
            LABEL_PATH : Str PATH of the label file.
        Returns:
            List of KittiObject : Contains all the labeled data

    '''
    used_cls = ['Car', 'Van', 'Truck', 'Misc']
    used_cls = ['Pedestrian', 'Car', 'Cyclist']
    # used_cls = ['car', 'van', 'truck']
    objects = []

    detection_data = open(LABEL_PATH, 'r')
    detections = detection_data.readlines()

    for object_index in range(len(detections)):

        data_str = detections[object_index]
        data_list = data_str.split()

        if data_list[0] not in used_cls:
            continue

        object_it = KittiObject()
        object_it.data_line_num = object_index
        object_it.cls = data_list[0]
        object_it.truncate = float(data_list[1])
        object_it.occlusion = int(data_list[2])
        object_it.alpha = float(data_list[3])
        object_it.Box2D = [float(data_list[4]), float(data_list[5]), float(data_list[6]), float(data_list[7])]

        #                            width   x       height  y       lenth z
        object_it.dim = np.array([data_list[10], data_list[8], data_list[9]]).astype(float)

        # hwl
        # lhw
        # The KITTI GT 3D box is on cam0 frame, while we deal with cam2 frame
        object_it.pos = np.array(data_list[11:14]).astype(float)  # 0.062
        # The orientation definition is inconsitent with right-hand coordinates in kitti
        object_it.orientation = float(data_list[14])
        if len(data_list) > 15:
            object_it.score = float(data_list[15])
        object_it.R = E2R(object_it.orientation, 0, 0)

        object_it.theta = object_it.alpha + m.pi / 2 - m.atan2(-object_it.pos[0], object_it.pos[2])
        pts3_c_o = []  # 3D location of 3D bounding box corners

        pts3_c_o.append(object_it.pos + object_it.R.dot(np.array([object_it.dim[0] / 2., 0, object_it.dim[2] / 2.0]).T))
        pts3_c_o.append(object_it.pos + object_it.R.dot(np.array([object_it.dim[0] / 2, 0, -object_it.dim[2] / 2.0]).T))
        pts3_c_o.append(
            object_it.pos + object_it.R.dot(np.array([-object_it.dim[0] / 2, 0, -object_it.dim[2] / 2.0]).T))
        pts3_c_o.append(object_it.pos + object_it.R.dot(np.array([-object_it.dim[0] / 2, 0, object_it.dim[2] / 2.0]).T))

        pts3_c_o.append(
            object_it.pos + object_it.R.dot(
                np.array([object_it.dim[0] / 2, -object_it.dim[1], object_it.dim[2] / 2.0]).T))
        pts3_c_o.append(
            object_it.pos + object_it.R.dot(
                np.array([object_it.dim[0] / 2, -object_it.dim[1], -object_it.dim[2] / 2.0]).T))
        pts3_c_o.append(
            object_it.pos + object_it.R.dot(
                np.array([-object_it.dim[0] / 2, -object_it.dim[1], -object_it.dim[2] / 2.0]).T))
        pts3_c_o.append(
            object_it.pos + object_it.R.dot(
                np.array([-object_it.dim[0] / 2, -object_it.dim[1], object_it.dim[2] / 2.0]).T))

        #
        object_it.Box3D = pts3_c_o
        box2d = []
        # for i in range(8):
        #     # if pts3_c_o[i][2] < 0:
        #     #     continue
        #     pt2 = Space2Image(calib.p2, NormalizeVector(pts3_c_o[i]))
        #     box2d.append(pt2)
        # object_it.Box3D_in_image2 = box2d
        # box2d = []
        # for i in range(8):
        #     # if pts3_c_o[i][2] < 0:
        #     #     continue
        #     pt2 = Space2Image(calib.p3, NormalizeVector(pts3_c_o[i]))
        #     box2d.append(pt2)
        # object_it.Box3D_in_image3 = box2d
        objects.append(object_it)

    return objects

def read_obj_data1(LABEL_PATH, calib=None, im_shape=None):
    '''Reads in object label file from Kitti Object Dataset.

        Inputs:
            LABEL_PATH : Str PATH of the label file.
        Returns:
            List of KittiObject : Contains all the labeled data

    '''

    # used_cls = ['car', 'van', 'truck']
    objects = []

    detection_data = open(LABEL_PATH, 'r')
    detections = detection_data.readlines()

    for object_index in range(len(detections)):

        data_str = detections[object_index]
        data_list = data_str.split()

        if data_list[0] =='DontCare':
            continue

        object_it = KittiObject()
        object_it.data_line_num = object_index
        object_it.cls = data_list[0]
        object_it.truncate = float(data_list[1])
        object_it.occlusion = int(data_list[2])
        object_it.alpha = float(data_list[3])
        object_it.Box2D = [float(data_list[4]), float(data_list[5]), float(data_list[6]), float(data_list[7])]

        #                            width   x       height  y       lenth z
        object_it.dim = np.array([data_list[10], data_list[8], data_list[9]]).astype(float)

        # hwl
        # lhw
        # The KITTI GT 3D box is on cam0 frame, while we deal with cam2 frame
        object_it.pos = np.array(data_list[11:14]).astype(float)  # 0.062
        # The orientation definition is inconsitent with right-hand coordinates in kitti
        object_it.orientation = float(data_list[14])
        if len(data_list) > 15:
            object_it.score = float(data_list[15])
        object_it.R = E2R(object_it.orientation, 0, 0)

        object_it.theta = object_it.alpha + m.pi / 2 - m.atan2(-object_it.pos[0], object_it.pos[2])
        pts3_c_o = []  # 3D location of 3D bounding box corners

        pts3_c_o.append(object_it.pos + object_it.R.dot(np.array([object_it.dim[0] / 2., 0, object_it.dim[2] / 2.0]).T))
        pts3_c_o.append(object_it.pos + object_it.R.dot(np.array([object_it.dim[0] / 2, 0, -object_it.dim[2] / 2.0]).T))
        pts3_c_o.append(
            object_it.pos + object_it.R.dot(np.array([-object_it.dim[0] / 2, 0, -object_it.dim[2] / 2.0]).T))
        pts3_c_o.append(object_it.pos + object_it.R.dot(np.array([-object_it.dim[0] / 2, 0, object_it.dim[2] / 2.0]).T))

        pts3_c_o.append(
            object_it.pos + object_it.R.dot(
                np.array([object_it.dim[0] / 2, -object_it.dim[1], object_it.dim[2] / 2.0]).T))
        pts3_c_o.append(
            object_it.pos + object_it.R.dot(
                np.array([object_it.dim[0] / 2, -object_it.dim[1], -object_it.dim[2] / 2.0]).T))
        pts3_c_o.append(
            object_it.pos + object_it.R.dot(
                np.array([-object_it.dim[0] / 2, -object_it.dim[1], -object_it.dim[2] / 2.0]).T))
        pts3_c_o.append(
            object_it.pos + object_it.R.dot(
                np.array([-object_it.dim[0] / 2, -object_it.dim[1], object_it.dim[2] / 2.0]).T))

        #
        object_it.Box3D = pts3_c_o
        box2d = []
        for i in range(8):
            # if pts3_c_o[i][2] < 0:
            #     continue
            pt2 = Space2Image(calib.p2, NormalizeVector(pts3_c_o[i]))
            box2d.append(pt2)
        object_it.Box3D_in_image2 = box2d
        box2d = []
        for i in range(8):
            # if pts3_c_o[i][2] < 0:
            #     continue
            pt2 = Space2Image(calib.p3, NormalizeVector(pts3_c_o[i]))
            box2d.append(pt2)
        object_it.Box3D_in_image3 = box2d
        objects.append(object_it)

    return objects
def right_box(data_list,calib,im_shape):
    object_it = KittiObject()
    object_it.dim = np.array([data_list[10], data_list[8], data_list[9]]).astype(float)

    # hwl
    # lhw
    # The KITTI GT 3D box is on cam0 frame, while we deal with cam2 frame
    object_it.pos = np.array(data_list[11:14]).astype(float)  # 0.062
    # The orientation definition is inconsitent with right-hand coordinates in kitti
    object_it.orientation = float(data_list[14])
    if len(data_list) > 15:
        object_it.score = float(data_list[15])
    object_it.R = E2R(object_it.orientation, 0, 0)

    object_it.theta = object_it.alpha + m.pi / 2 - m.atan2(-object_it.pos[0], object_it.pos[2])
    pts3_c_o = []  # 3D location of 3D bounding box corners

    pts3_c_o.append(object_it.pos + object_it.R.dot(np.array([object_it.dim[0] / 2., 0, object_it.dim[2] / 2.0]).T))
    pts3_c_o.append(object_it.pos + object_it.R.dot(np.array([object_it.dim[0] / 2, 0, -object_it.dim[2] / 2.0]).T))
    pts3_c_o.append(
        object_it.pos + object_it.R.dot(np.array([-object_it.dim[0] / 2, 0, -object_it.dim[2] / 2.0]).T))
    pts3_c_o.append(object_it.pos + object_it.R.dot(np.array([-object_it.dim[0] / 2, 0, object_it.dim[2] / 2.0]).T))

    pts3_c_o.append(
        object_it.pos + object_it.R.dot(
            np.array([object_it.dim[0] / 2, -object_it.dim[1], object_it.dim[2] / 2.0]).T))
    pts3_c_o.append(
        object_it.pos + object_it.R.dot(
            np.array([object_it.dim[0] / 2, -object_it.dim[1], -object_it.dim[2] / 2.0]).T))
    pts3_c_o.append(
        object_it.pos + object_it.R.dot(
            np.array([-object_it.dim[0] / 2, -object_it.dim[1], -object_it.dim[2] / 2.0]).T))
    pts3_c_o.append(
        object_it.pos + object_it.R.dot(
            np.array([-object_it.dim[0] / 2, -object_it.dim[1], object_it.dim[2] / 2.0]).T))

    #
    object_it.Box3D = pts3_c_o
    box2d = []
    for i in range(8):
        # if pts3_c_o[i][2] < 0:
        #     continue
        pt2 = Space2Image(calib.p3, NormalizeVector(pts3_c_o[i]))
        box2d.append(pt2)
    box2d=np.array(box2d)
    box2d[box2d<0]=0
    box2d[:,0][box2d[:,0]>im_shape[1]] = im_shape[1]
    box2d[:, 1][box2d[:, 1] > im_shape[0]] = im_shape[0]
    box=np.min(box2d,axis=0).tolist()+np.max(box2d,axis=0).tolist()
    return box
def parm_to_3DBox(parm, P,im_shape=None):
    ry = parm[0]
    l = parm[1]
    h = parm[2]
    w = parm[3]
    px = parm[4]
    py = parm[5]
    pz = parm[6]

    R = E2R(ry, 0, 0)

    pos = np.array(parm[4:]).astype(float)
    pts3_c_o = []  # 3D location of 3D bounding box corners

    pts3_c_o.append(pos + R.dot(np.array([l / 2., 0, w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([l / 2, 0, -w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-l / 2, 0, -w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-l / 2, 0, w / 2.0]).T))

    pts3_c_o.append(pos + R.dot(np.array([l / 2, -h, w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([l / 2, -h, -w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-l / 2, -h, -w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-l / 2, -h, w / 2.0]).T))

    #
    box2d = []
    states=True
    for i in range(8):
        pt2 = Space2Image(P, NormalizeVector(pts3_c_o[i]))
        if np.min(pt2)<0:
            states=False
        #pt2[pt2<0]=0
        if im_shape:
            im_h = im_shape[0]
            im_w = im_shape[1]
            if pt2[0]>im_w:
                states = False
                #pt2[0]=im_w
            if pt2[1]>im_h:
                states = False
                #pt2[1]=im_h

        box2d.append(pt2)

    return box2d,states
def parm_to_3DBox1(parm, P,im_shape=None):
    ry = parm[0]
    l = parm[1]
    h = parm[2]
    w = parm[3]
    px = parm[4]
    py = parm[5]
    pz = parm[6]

    R = E2R(ry, 0, 0)

    pos = np.array(parm[4:]).astype(float)
    pts3_c_o = []  # 3D location of 3D bounding box corners

    pts3_c_o.append(pos + R.dot(np.array([l / 2., 0, w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([l / 2, 0, -w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-l / 2, 0, -w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-l / 2, 0, w / 2.0]).T))

    pts3_c_o.append(pos + R.dot(np.array([l / 2, -h, w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([l / 2, -h, -w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-l / 2, -h, -w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-l / 2, -h, w / 2.0]).T))

    #
    box2d = []
    states=True
    for i in range(8):
        pt2 = Space2Image(P, NormalizeVector(pts3_c_o[i]))

        box2d.append(pt2)

    return box2d,states
def parm_to_3DBox_in3Dspace(parm, P,im_shape=None):
    ry = parm[0]
    l = parm[1]
    h = parm[2]
    w = parm[3]
    px = parm[4]
    py = parm[5]
    pz = parm[6]

    R = E2R(ry, 0, 0)

    pos = np.array(parm[4:]).astype(float)
    pts3_c_o = []  # 3D location of 3D bounding box corners

    pts3_c_o.append(pos + R.dot(np.array([l / 2., 0, w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([l / 2, 0, -w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-l / 2, 0, -w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-l / 2, 0, w / 2.0]).T))

    pts3_c_o.append(pos + R.dot(np.array([l / 2, -h, w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([l / 2, -h, -w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-l / 2, -h, -w / 2.0]).T))
    pts3_c_o.append(pos + R.dot(np.array([-l / 2, -h, w / 2.0]).T))




    return pts3_c_o
def project_to_image(point_cloud, p):
    ''' Projects a 3D point cloud to 2D points for plotting

        Inputs:
            point_cloud: 3D point cloud (3, N)
            p: Camera matrix (3, 4)
        Return:
            pts_2d: the image coordinates of the 3D points in the shape (2, N)

    '''

    pts_2d = np.dot(p, np.append(point_cloud,
                                 np.ones((1, point_cloud.shape[1])),
                                 axis=0))

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, 0)
    return pts_2d


def point_in_2Dbox(points_im, obj):
    '''Select points contained in object 2D box

        Inputs:
            points_im: N x 2 numpy array in image
            obj: KittiObject
        Return:
            pointcloud indexes

    '''
    point_filter = (points_im[:, 0] > obj.box[0]) & \
                   (points_im[:, 0] < obj.box[2]) & \
                   (points_im[:, 1] > obj.box[1]) & \
                   (points_im[:, 1] < obj.box[3])
    return point_filter


def box2D_sourround_box3dimimage(box3d_in_image):
    box3d_in_image = np.array(box3d_in_image)
    min_cord= np.min(box3d_in_image, axis=0)
    min_x=min_cord[0]
    min_y=min_cord[1]
    max_cord = np.max(box3d_in_image, axis=0)
    max_x = max_cord[0]
    max_y = max_cord[1]
    box2d = [min_x, min_y, max_x, max_y]
    area=(max_x-min_x)*(max_y-min_y)
    return np.array(box2d),area

def box2D_sourround_box3dimimage_mat(box3d_in_image):
    min_cord= np.min(box3d_in_image, axis=1)
    max_cord = np.max(box3d_in_image, axis=1)
    box2d=np.append(min_cord,max_cord,axis=1)
    return box2d
def lidar_to_cam_frame(xyz_lidar, frame_calib):
    '''Transforms the pointclouds to the camera 2 frame.

        Inputs:
            xyz_lidar : N x 3  x,y,z coordinates of the pointcloud in lidar frame
            frame_calib : FrameCalibrationData
        Returns:
            ret_xyz : N x 3  x,y,z coordinates of the pointcloud in cam2 frame

    '''
    # Pad the r0_rect matrix to a 4x4
    r0_rect_mat = frame_calib.r0_rect
    r0_rect_mat = np.pad(r0_rect_mat, ((0, 1), (0, 1)),
                         'constant', constant_values=0)
    r0_rect_mat[3, 3] = 1

    # Pad the tr_vel_to_cam matrix to a 4x4
    tf_mat = frame_calib.tr_velodyne_to_cam0
    tf_mat = np.pad(tf_mat, ((0, 1), (0, 0)),
                    'constant', constant_values=0)
    tf_mat[3, 3] = 1

    # Pad the t_cam2_cam0 matrix to a 4x4
    t_cam2_cam0 = np.identity(4)
    t_cam2_cam0[0:3, 3] = frame_calib.t_cam2_cam0

    # Pad the pointcloud with 1's for the transformation matrix multiplication
    one_pad = np.ones(xyz_lidar.shape[0]).reshape(-1, 1)
    xyz_lidar = np.append(xyz_lidar, one_pad, axis=1)

    # p_cam = P2 * R0_rect * Tr_velo_to_cam * p_velo
    rectified = np.dot(r0_rect_mat, tf_mat)

    to_cam2 = np.dot(t_cam2_cam0, rectified)
    ret_xyz = np.dot(to_cam2, xyz_lidar.T)

    # Change to N x 3 array for consistency.
    return ret_xyz[0:3].T


def get_point_cloud(LIDAR_PATH, frame_calib, image_shape=None, objects=None):
    ''' Calculates the lidar point cloud, and optionally returns only the
        points that are projected to the image.

        Inputs:
            LIDAR_PATH: string specify the lidar file path
            frame_calib: FrameCalibrationData
        Return:
            (3, N) point_cloud in the form [[x,...][y,...][z,...]]

    '''
    if not os.path.isfile(LIDAR_PATH):
        return np.array([[0], [0], [0]])
    if image_shape is not None:
        im_size = [image_shape[1], image_shape[0]]
    else:
        im_size = [1242, 375]

    with open(LIDAR_PATH, 'rb') as fid:
        data_array = np.fromfile(fid, np.single)
    xyzi = data_array.reshape(-1, 4)
    x = xyzi[:, 0]
    y = xyzi[:, 1]
    z = xyzi[:, 2]
    i = xyzi[:, 3]

    # Calculate the point cloud
    point_cloud = np.vstack((x, y, z))
    point_cloud = lidar_to_cam_frame(point_cloud.T, frame_calib)  # N x 3

    # Only keep points in front of camera (positive z)
    point_cloud = point_cloud[point_cloud[:, 2] > 0].T  # camera frame 3 x N
    # point_cloud=point_cloud.T
    # Project to image frame
    point_in_im = project_to_image(point_cloud, p=frame_calib.p2).T

    # Filter based on the given image size

    image_filter = (point_in_im[:, 0] > 0) & \
                   (point_in_im[:, 0] < im_size[0]) & \
                   (point_in_im[:, 1] > 0) & \
                   (point_in_im[:, 1] < im_size[1])

    object_filter = np.zeros(point_in_im.shape[0], dtype=bool)
    if objects is not None:
        for i in range(len(objects)):
            object_filter = np.logical_or(point_in_2Dbox(point_in_im, objects[i]), object_filter)

        object_filter = np.logical_and(image_filter, object_filter)
    else:
        object_filter = image_filter

    point_cloud = point_cloud.T[object_filter].T

    return point_cloud


def infer_boundary(im_shape, boxes_left):
    ''' Approximately infer the occlusion border for all objects
        accoording to the 2D bounding box

        Inputs:
            im_shape: H x W x 3
            boxes_left: rois x 4
        Return:
            left_right: left and right borderline for each object rois x 2
    '''
    left_right = np.zeros((boxes_left.shape[0], 2), dtype=np.float32)
    depth_line = np.zeros(im_shape[1] + 1, dtype=float)
    for i in range(boxes_left.shape[0]):
        for col in range(int(boxes_left[i, 0]), int(boxes_left[i, 2]) + 1):
            pixel = depth_line[col]
            depth = 1050.0 / boxes_left[i, 3]
            if pixel == 0.0:
                depth_line[col] = depth
            elif depth < depth_line[col]:
                depth_line[col] = (depth + pixel) / 2.0

    for i in range(boxes_left.shape[0]):
        left_right[i, 0] = boxes_left[i, 0]
        left_right[i, 1] = boxes_left[i, 2]
        left_visible = True
        right_visible = True
        if depth_line[int(boxes_left[i, 0])] < 1050.0 / boxes_left[i, 3]:
            left_visible = False
        if depth_line[int(boxes_left[i, 2])] < 1050.0 / boxes_left[i, 3]:
            right_visible = False

        if right_visible == False and left_visible == False:
            left_right[i, 1] = boxes_left[i, 0]

        for col in range(int(boxes_left[i, 0]), int(boxes_left[i, 2]) + 1):
            if left_visible and depth_line[col] >= 1050.0 / boxes_left[i, 3]:
                left_right[i, 1] = col
            elif right_visible and depth_line[col] < 1050.0 / boxes_left[i, 3]:
                left_right[i, 0] = col
    return left_right


# ---------------------------------------------------- Data Writing -------------------------------------------------#
def write_detection_results(cls, result_dir, file_number, calib, box, param, score):
    '''One by one write detection results to KITTI format label files.
    '''
    if result_dir is None: return
    result_dir = result_dir + '/data'
    Px=param[4]
    Py = param[5]
    Pz = param[6]
    l=param[1]
    h = param[2]
    w = param[3]
    ori=param[0]
    if ori > 2 * pi:
        while ori > 2 * pi:
            ori -= 2 * pi
    if ori < -2 * pi:
        while ori < -2 * pi:
            ori += 2 * pi

    if ori > pi:
        ori = 2 * pi - ori
    if ori < -pi:
        ori = 2 * pi + pi

    alpha=ori - m.atan2(Px, Pz)
    # convert the object from cam2 to the cam0 frame
    dis_cam02 = calib.t_cam2_cam0[0]

    output_str = cls + ' '
    output_str += '%.2f %.d ' % (-1, -1)
    output_str += '%.2f %.2f %.2f %.2f %.2f ' % (alpha, box[0], box[1], box[2], box[3])
    output_str += '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n' % (h, w, l, Px, Py, \
                                                                  Pz, ori, score)

    # Write TXT files
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    pred_filename = result_dir + '/' + file_number + '.txt'
    with open(pred_filename, 'a') as det_file:
        det_file.write(output_str)


def calib2P(frame_calib):
    # Pad the r0_rect matrix to a 4x4
    r0_rect_mat = frame_calib.r0_rect
    r0_rect_mat = np.pad(r0_rect_mat, ((0, 1), (0, 1)),
                         'constant', constant_values=0)
    r0_rect_mat[3, 3] = 1

    t_cam2_cam0 = np.identity(4)
    t_cam2_cam0[0:3, 3] = frame_calib.t_cam2_cam0

    return


def search_truncated_border(box3d_in_image):
    cor = np.array(box3d_in_image)
    x_min, y_min = cor.min(0)
    x_max, y_max = cor.max(0)

    return [x_min, y_min, x_max, y_max]


def write_refined_results(cls, result_dir, file_number, bbox, pos, dim, orien, ellipsoid_in_image, ellipse, theta_scale,
                          theta_ori, theta_dim):
    '''One by one write detection results to KITTI format label files.
    '''
    if result_dir is None: return
    result_dir = result_dir + '/data'

    # convert the object from cam2 to the cam0 frame

    output_str = cls + ' '
    output_str += '%.2f %.d ' % (-1, -1)
    alpha = orien - m.atan2(pos[0], pos[2])
    output_str += '%.2f %.2f %.2f %.2f %.2f ' % (alpha, bbox[0], bbox[1], bbox[2], bbox[3])
    output_str += '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f ' % (dim[1], dim[2], dim[0], pos[0], pos[1], \
                                                                pos[2], orien, 1)

    output_str += '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f ' % (
        ellipsoid_in_image[0][0], ellipsoid_in_image[0][1], ellipsoid_in_image[1][0], ellipsoid_in_image[1][1],
        ellipsoid_in_image[2][0], ellipsoid_in_image[2][1], ellipsoid_in_image[3][0], ellipsoid_in_image[3][1], \
 \
        ellipsoid_in_image[
            4][0],
        ellipsoid_in_image[
            4][1],
        ellipsoid_in_image[
            5][0],
        ellipsoid_in_image[
            5][1],
        ellipsoid_in_image[
            6][0],
        ellipsoid_in_image[
            6][1],
        ellipsoid_in_image[
            7][0],
        ellipsoid_in_image[
            7][1])

    output_str += '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f ' % (
        ellipse[0], ellipse[1], ellipse[2], ellipse[3], ellipse[4][0, 0], ellipse[4][0, 1], ellipse[4][1, 0],
        ellipse[4][1, 1])

    output_str += '%.2f %.2f %.2f %.2f %.2f \n' % (theta_scale, theta_ori, theta_dim[0], theta_dim[1], theta_dim[2])
    # Write TXT files
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    pred_filename = result_dir + '/' + file_number + '.txt'
    with open(pred_filename, 'a') as det_file:
        det_file.write(output_str)


def read_points_all(file_path):
    detection_data = open(file_path, 'r')
    detections = detection_data.readlines()
    box2d=[]
    box2d_score=[]
    key_point=[]
    dim=[]
    kp_score=[]
    depth=[]
    alpha=[]
    center_point=[]
    center_score=[]
    for index in range(len(detections)):
        data_str = detections[index]
        data_list = data_str.split()
        data_list = [float(data_list[i]) for i in range(len(data_list))]
        box2d.append(np.array(data_list[0:4]))
        box2d_score.append(data_list[4])
        key_point.append(np.reshape(np.array(data_list[5:23]),(2,9),'F'))
        center_point.append(data_list[21:23])
        center_score.append(data_list[31])
        dim.append(data_list[34:37])
        kp_score.append(data_list[23:32])
        depth.append(data_list[32])
        alpha.append(data_list[33])
    # data_mat = np.array(all_det)
    # data_dim=np.array(dim_det)
    # data_dim = np.reshape(data_dim, (3, -1),order='F')
    # data_mat = np.reshape(data_mat, (2, -1), order='F').astype(np.uint)
    return box2d,box2d_score,key_point,dim,kp_score,depth,alpha,center_point,center_score
def read_points_dim_ori_center(file_path):
    detection_data = open(file_path, 'r')
    detections = detection_data.readlines()
    box2d=[]
    box2d_score=[]
    key_point=[]
    dim=[]
    kp_score=[]
    depth=[]
    alpha=[]
    center_point=[]
    center_score=[]
    for index in range(len(detections)):
        data_str = detections[index]
        data_list = data_str.split()
        data_list = [float(data_list[i]) for i in range(len(data_list))]
        key_point.append(np.reshape(np.array(data_list[3:19]),(2,8),'F'))
        center_point.append(np.reshape(np.array(data_list[:2]),(2,1),'F'))
        center_score.append(data_list[2:3])
        dim.append(data_list[28:31])
        kp_score.append(data_list[19:27])
        alpha.append(data_list[27])
    # data_mat = np.array(all_det)
    # data_dim=np.array(dim_det)
    # data_dim = np.reshape(data_dim, (3, -1),order='F')
    # data_mat = np.reshape(data_mat, (2, -1), order='F').astype(np.uint)
    return box2d,box2d_score,key_point,dim,kp_score,depth,alpha,center_point,center_score
def read_points_dim(file_path):
    detection_data = open(file_path, 'r')
    detections = detection_data.readlines()
    box2d=[]
    box2d_score=[]
    key_point=[]
    dim=[]
    kp_score=[]
    depth=[]
    alpha=[]
    center_point=[]
    center_score=[]
    for index in range(len(detections)):
        data_str = detections[index]
        data_list = data_str.split()
        data_list = [float(data_list[i]) for i in range(len(data_list))]
        box2d.append(np.array(data_list[0:4]))
        box2d_score.append(data_list[4])
        key_point.append(np.reshape(np.array(data_list[5:21]),(2,8),'F'))

        dim.append(data_list[21:24])
        kp_score.append(data_list[24:32])

    # data_mat = np.array(all_det)
    # data_dim=np.array(dim_det)
    # data_dim = np.reshape(data_dim, (3, -1),order='F')
    # data_mat = np.reshape(data_mat, (2, -1), order='F').astype(np.uint)
    return box2d,box2d_score,key_point,dim,kp_score,depth,alpha,center_point,center_score
def read_points_dim_point9(file_path):
    detection_data = open(file_path, 'r')
    detections = detection_data.readlines()
    box2d=[]
    box2d_score=[]
    key_point=[]
    dim=[]
    kp_score=[]
    depth=[]
    alpha=[]
    center_point=[]
    center_score=[]
    for index in range(len(detections)):
        data_str = detections[index]
        data_list = data_str.split()
        data_list = [float(data_list[i]) for i in range(len(data_list))]
        box2d.append(np.array(data_list[0:4]))
        box2d_score.append(data_list[4])
        key_point.append(np.reshape(np.array(data_list[5:23]),(2,9),'F'))

        dim.append(data_list[32:35])
        kp_score.append(data_list[23:32])

    # data_mat = np.array(all_det)
    # data_dim=np.array(dim_det)
    # data_dim = np.reshape(data_dim, (3, -1),order='F')
    # data_mat = np.reshape(data_mat, (2, -1), order='F').astype(np.uint)
    return box2d,box2d_score,key_point,dim,kp_score,depth,alpha,center_point,center_score
def read_points_dim_point9_rot(file_path):
    detection_data = open(file_path, 'r')
    detections = detection_data.readlines()
    box2d=[]
    box2d_score=[]
    key_point=[]
    dim=[]
    kp_score=[]
    depth=[]
    alpha=[]
    center_point=[]
    center_score=[]
    for index in range(len(detections)):
        data_str = detections[index]
        data_list = data_str.split()
        data_list = [float(data_list[i]) for i in range(len(data_list))]
        box2d.append(np.array(data_list[0:4]))
        box2d_score.append(data_list[4])
        key_point.append(np.reshape(np.array(data_list[5:23]),(2,9),'F'))
        alpha.append(data_list[35])
        dim.append(data_list[32:35])
        kp_score.append(data_list[23:32])

    # data_mat = np.array(all_det)
    # data_dim=np.array(dim_det)
    # data_dim = np.reshape(data_dim, (3, -1),order='F')
    # data_mat = np.reshape(data_mat, (2, -1), order='F').astype(np.uint)
    return box2d,box2d_score,key_point,dim,kp_score,depth,alpha,center_point,center_score
def read_points_dim_point9_rot_depth(file_path):
    detection_data = open(file_path, 'r')
    detections = detection_data.readlines()
    box2d=[]
    box2d_score=[]
    key_point=[]
    dim=[]
    kp_score=[]
    depth=[]
    alpha=[]
    center_point=[]
    center_score=[]
    prob=[]
    for index in range(len(detections)):
        data_str = detections[index]
        data_list = data_str.split()
        data_list = [float(data_list[i]) for i in range(len(data_list))]
        box2d.append(np.array(data_list[0:4]))
        box2d_score.append(data_list[4])
        key_point.append(np.reshape(np.array(data_list[5:23]),(2,9),'F'))
        alpha.append(data_list[35])
        dim.append(data_list[32:35])
        kp_score.append(data_list[23:32])
        prob.append(data_list[36])
        #off.append(data_list[36:39])
        #center_score.append(data_list[37])
    # data_mat = np.array(all_det)
    # data_dim=np.array(dim_det)
    # data_dim = np.reshape(data_dim, (3, -1),order='F')
    # data_mat = np.reshape(data_mat, (2, -1), order='F').astype(np.uint)
    return box2d,box2d_score,key_point,dim,kp_score,depth,alpha,center_point,prob
def read_points_dim_point9_rot_location(file_path):
    detection_data = open(file_path, 'r')
    detections = detection_data.readlines()
    box2d=[]
    box2d_score=[]
    key_point=[]
    dim=[]
    kp_score=[]
    depth=[]
    alpha=[]
    center_point=[]
    center_score=[]
    for index in range(len(detections)):
        data_str = detections[index]
        data_list = data_str.split()
        data_list = [float(data_list[i]) for i in range(len(data_list))]
        box2d.append(np.array(data_list[0:4]))
        box2d_score.append(data_list[4])
        key_point.append(np.reshape(np.array(data_list[5:23]),(2,9),'F'))
        alpha.append(data_list[35])
        dim.append(data_list[32:35])
        kp_score.append(data_list[23:32])
        depth.append(data_list[36:39])
    # data_mat = np.array(all_det)
    # data_dim=np.array(dim_det)
    # data_dim = np.reshape(data_dim, (3, -1),order='F')
    # data_mat = np.reshape(data_mat, (2, -1), order='F').astype(np.uint)
    return box2d,box2d_score,key_point,dim,kp_score,depth,alpha,center_point,center_score
def read_points_dim_point9_rot_depth_class(file_path):
    detection_data = open(file_path, 'r')
    detections = detection_data.readlines()
    box2d=[]
    box2d_score=[]
    key_point=[]
    dim=[]
    kp_score=[]
    depth=[]
    alpha=[]
    center_point=[]
    center_score=[]
    cls=[]
    print(file_path)
    for index in range(len(detections)):
        data_str = detections[index]
        data_list = data_str.split()
        data_list = [float(data_list[i]) for i in range(len(data_list))]
        box2d.append(np.array(data_list[0:4]))
        box2d_score.append(data_list[4])
        key_point.append(np.reshape(np.array(data_list[5:23]),(2,9),'F'))
        alpha.append(data_list[35])
        dim.append(data_list[32:35])
        kp_score.append(data_list[23:32])
        depth.append(data_list[36])
        #cls.append(data_list[37])
    # data_mat = np.array(all_det)
    # data_dim=np.array(dim_det)
    # data_dim = np.reshape(data_dim, (3, -1),order='F')
    # data_mat = np.reshape(data_mat, (2, -1), order='F').astype(np.uint)
    return box2d,box2d_score,key_point,dim,kp_score,depth,alpha,center_point,center_score
def read_points_dim_point9_rot_depth11(file_path):
    detection_data = open(file_path, 'r')
    detections = detection_data.readlines()
    box2d=[]
    box2d_score=[]
    key_point=[]
    dim=[]
    kp_score=[]
    depth=[]
    alpha=[]
    center_point=[]
    center_score=[]
    cls=[]
    for index in range(len(detections)):
        data_str = detections[index]
        data_list = data_str.split()
        data_list = [float(data_list[i]) for i in range(len(data_list))]
        box2d.append(np.array(data_list[0:4]))
        box2d_score.append(data_list[4])
        key_point.append(np.reshape(np.array(data_list[5:23]),(2,9),'F'))
        alpha.append(data_list[35])
        dim.append(data_list[32:35])
        kp_score.append(data_list[23:32])
        depth.append(data_list[36])
        #cls.append(data_list[37])
    # data_mat = np.array(all_det)
    # data_dim=np.array(dim_det)
    # data_dim = np.reshape(data_dim, (3, -1),order='F')
    # data_mat = np.reshape(data_mat, (2, -1), order='F').astype(np.uint)
    return box2d,box2d_score,key_point,dim,kp_score,depth,alpha,center_point,center_score
def triangulate_Point(calib, points2, points3):
    pass

pi=np.pi
def find_bottom_keypoint(BOX3D_in_image,BOX3D_param,box_2D):
    assert BOX3D_in_image.shape[0]==BOX3D_param.shape[0],'BOX3D shape ia not same'
    batch = BOX3D_in_image.shape[0]
    keypoint=np.zeros((batch,2))
    for i in range(batch):
        ori=BOX3D_param[i,0]
        if ori>2*pi:
            while ori>2*pi:
                ori-=2*pi
        if ori<-2*pi:
            while ori<-2*pi:
                ori+=2*pi

        if ori>pi:
            ori=2*pi-ori
        if ori<-pi:
            ori=2*pi+pi

        if ori<0 and ori>-pi/2:
           keypoint_index=2
        elif ori<-pi/2 and ori>-pi:
            keypoint_index = 3
        elif ori<pi/2 and ori>0:
            keypoint_index = 1
        elif ori<pi and ori>pi/2:
            keypoint_index = 0
        else:
            print('keypoint_error')
        key_p=BOX3D_in_image[i,keypoint_index,:]
        box_2d=box_2D[i]
        key_p[0]=max(box_2d[0],key_p[0])
        key_p[1] = max(box_2d[1], key_p[1])
        key_p[0] = min(box_2d[2], key_p[0])
        key_p[1] = min(box_2d[3], key_p[1])
        keypoint[i]=key_p
    return keypoint

def joint_point_2dbox(point,box,shape):
    min_c=np.min(point,0)
    max_c=np.max(point,0)
    min_x=min_c[0]
    min_y=min_c[1]
    max_x=max_c[0]
    max_y=max_c[1]
    if min_x<0 or min_y>0:
        point[:, 1][point[:, 1] < min_y + 2] = box[1]
        point[:, 0][point[:, 0] > max_x - 2] = box[2]
    elif max_x>shape[1] or max_y>shape[0]:
        point[:, 0][point[:, 0] < min_x + 2] = box[0]
        point[:, 1][point[:, 1] < min_y + 2] = box[1]
    else:
        point[:,0][point[:,0]<min_x+2]=box[0]
        point[:, 1][point[:, 1] < min_y + 2] = box[1]
        point[:, 0][point[:, 0] > max_x - 2] = box[2]
        point[:, 1][point[:, 1] > max_y - 2] = box[3]
    return point
def joint_point_2dbox1(point,box,shape):
    min_c=np.min(point,0)
    max_c=np.max(point,0)
    min_x=min_c[0]
    min_y=min_c[1]
    max_x=max_c[0]
    max_y=max_c[1]
    border=9
    if box[0]<border and box[3]+border>shape[0]:
        point[:, 1][point[:, 1] < min_y + 2] = box[1]
        point[:, 0][point[:, 0] > max_x - 2] = box[2]
    elif max_x>shape[1] or max_y>shape[0]:
        point[:, 0][point[:, 0] < min_x + 2] = box[0]
        point[:, 1][point[:, 1] < min_y + 2] = box[1]
    else:
        point[:,0][point[:,0]<min_x+2]=box[0]
        point[:, 1][point[:, 1] < min_y + 2] = box[1]
        point[:, 0][point[:, 0] > max_x - 2] = box[2]
        point[:, 1][point[:, 1] > max_y - 2] = box[3]
    return point






