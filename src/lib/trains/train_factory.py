from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .car_pose import CarPoseTrainer
train_factory = {
  'car_pose': CarPoseTrainer,
}
