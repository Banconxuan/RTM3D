import time
import fire
import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result
import os

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def get_image_index_str(img_idx):
    return "{:06d}".format(img_idx)
def exists(img_indx,result_label):
    new_indx=[]
    for num,i in enumerate(img_indx):
        fram_ID=get_image_index_str(i)
        path=result_label+fram_ID+'.txt'
        if os.path.exists(path):
            new_indx.append(i)
    return new_indx
def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1):


    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    val_image_ids = exists(val_image_ids,result_path)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    dt_annos = kitti.get_label_annos(result_path, val_image_ids)
    if coco:
        print(get_coco_eval_result(gt_annos, dt_annos, current_class))
    else:
        print(get_official_eval_result(gt_annos, dt_annos, current_class))

if __name__ == '__main__':
    fire.Fire()
