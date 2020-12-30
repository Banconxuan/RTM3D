# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
g=$(($2<8?$2:8))

srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
python ./src/tools/kitti-object-eval-python/evaluate.py evaluate --label_path=$3/data/kitti/label/ --label_split_file=$3/data/kitti/val.txt --current_class=0,1,2 --coco=False  --result_path=$3/exp/results/data/
