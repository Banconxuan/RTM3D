Currently we provide the dataloader of KITTI dataset, and the NuScenes dataset is on the way.
# Training & Testing & Evaluation
## Training by python with multiple GPUs in a machine
Run following command to train model with ResNet-18 backbone.
   ~~~
   python ./src/main.py --data_dir ./kitti_format --exp_id KM3D_res18 --arch res_18 --batch_size 32 --master_batch_size 16 --lr 1.25e-4 --gpus 0,1 --num_epochs 200
   ~~~
Run following command to train model with DLA-34 backbone.
   ~~~
   python ./src/main.py --data_dir ./kitti_format --exp_id KM3D_dla34 --arch dla_34 --batch_size 16 --master_batch_size 8 --lr 1.25e-4 --gpus 0,1 --num_epochs 200
   ~~~
## Results generation
Run following command for results generation.
   ~~~
   # ResNet-18 backbone
   python ./src/faster.py --demo ./kitti_format/data/kitti/val.txt --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/exp/KM3D_res18/model_last.pth --gpus 0 --arch res_18
   # or DLA-3D backbone
   python ./src/faster.py --demo ./kitti_format/data/kitti/val.txt --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/exp/KM3D_dla34/model_last.pth --gpus 0 --arch res_18
   ~~~
## Visualization
Run following command for visualization.
   ~~~
   # ResNet-18 backbone
   python ./src/faster.py --vis --demo ./kitti_format/data/kitti/val.txt --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/exp/KM3D_res18/model_last.pth --gpus 0 --arch res_18
   # or DLA-3D backbone
   python ./src/faster.py --vis --demo ./kitti_format/data/kitti/val.txt --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/exp/KM3D_dla34/model_last.pth --gpus 0 --arch res_18
   ~~~
## All-in-one fashion
Generating results in all-in-one fashion. It simultaneously generates main center, 2D bounding box, regressed keypoints, heatmap keypoint, dim, orientation, confidence. 
      
      ~~~
        # ResNet-18 backbone
        python ./src/faster.py --demo ./kitti_format/data/kitti/val.txt --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/exp/KM3D_res18/model_last.pth --gpus 0 --arch res_18
        # or DLA-3D backbone
        python ./src/faster.py --demo ./kitti_format/data/kitti/val.txt --calib_dir ./kitti_format/data/kitti/calib/ --load_model ./kitti_format/exp/KM3D_dla34/model_last.pth --gpus 0 --arch res_18
      ~~~
## Evaluation
Run following command for evaluation.
   ~~~
   python ./src/tools/kitti-object-eval-python/evaluate.py evaluate --label_path=./kitti_format/data/kitti/label/ --label_split_file ./ImageSets/val.txt --current_class=0,1,2 --coco=False --result_path=./kitti_format/exp/results/data/
   ~~~

## Srun for training if available
    ~~~
    sh train.sh your_node 8
    ~~~
## Srun for inference and evaluation if available
    ~~~
    sh infer_eval.sh your_node 1 ./kitti_formet KM3D res_18 model_last
    ~~~
