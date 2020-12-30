# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
g=$(($2<8?$2:8))

srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
python ./src/main.py --distribute --data_dir ./kitti_format --exp_id km3d_multi_class --arch res_18 --gpus 0,1,2,3,4,5,6,7 --batch_size 17 --lr 1.25e-4 --num_epochs 200 --save_all \
2>&1 | tee log.train
