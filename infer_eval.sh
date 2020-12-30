#!/bin/bash
echo "data dir is:" $3
echo "model pth is:" $4
echo "backbone is:" $5
g=$(($2<8?$2:8))
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
python ./src/demo.py --demo $3/data/kitti/val.txt --calib_dir $3/data/kitti/calib/ --load_model $3/exp/$4/model_120.pth --data_dir $3 --gpus 0 --arch $5
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
python ./src/tools/kitti-object-eval-python/evaluate.py evaluate --label_path=$3/data/kitti/label/ --label_split_file=$3/data/kitti/val.txt --current_class=0,1,2 --coco=False  --result_path=$3/exp/results/data/
