from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
from detectors.car_pose import CarPoseDetector
from opts import opts
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['net', 'dec']

def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    opt.faster=False
    Detector = CarPoseDetector
    detector = Detector(opt)

    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      if opt.demo[-3:]=='txt':
          with open(opt.demo,'r') as f:
              lines = f.readlines()
          image_names=[os.path.join('/freespace/3ddetectionkitti/dataset/sequences/00/image_2/',img[:6]+'.png') for img in lines]
      else:
        image_names = [opt.demo]
    time_tol = 0
    num = 0
    for (image_name) in image_names:
      num+=1
      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
          time_tol=time_tol+ret[stat]
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      time_str=time_str+'{} {:.3f}s |'.format('tol', time_tol/num)
      print(time_str)
if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
