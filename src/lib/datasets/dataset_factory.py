from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.car_pose import CarPoseDataset
from .dataset.kittihp import KITTIHP
from .dataset.nusceneshp import NUSCENESHP
dataset_factory = {
  'kitti': KITTIHP,
  'nuscenes': NUSCENESHP
}
def get_dataset(dataset):
  class Dataset(dataset_factory[dataset], CarPoseDataset):
    pass
  return Dataset
  
