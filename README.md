## RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving
## Monocular 3D Detection with Geometric Constraints Embedding and Semi-supervised Training (KM3D)

RTM3D(ECCV2020) and KM3D (namely RTM3D++) are efficiency and accuracy monocular 3D object detection methods for autonomous driving.

We replaced the post-processing of RTM3D with KM3D's Geometric Reasoning Module (GRM) to increase the speed of inference. 
[**KM3D**](https://arxiv.org/abs/2009.00764), [**RTM3D**](https://arxiv.org/abs/2001.03343)

## Introduction
RTM3D is a novel one-stage and keypoints-based framework for monocular 3D objects detection. RTM3D is the first real-time system (FPS>24) for monocular image 3D detection while
achieves state-of-the-art performance on the KITTI benchmark.
KM3D reformulate the geometric constraints as a differentiable version and embed it into the net-work to reduce running time while maintaining the consistency
of model outputs in an end-to-end fashion. KM3D achieves 46FPS and SOTA performance on the KITTI benchmark.
RTM3D and KM3D only require RGB images without synthetic data, instance segmentation, CAD model, or depth generator.

## Highlights
- **Fast:** 47FPS of single image test speed in KITTI benchmark with 384*1280 resolution
- **Accuracy:** SOTA on the KITTI benchmark.
- **Anchor Free:** No 2D or 3D anchor are reauired
- **Differentiable geometric reasoning module:** Promote the running efficiency and optimize outputs of
network jointly. Combining the strengths of both CNN and
geometric constraints.
- **Easy to deploy:** RTM3D and KM3D only uses conventional convolution and upsampling operations, and the geometry module only needs to solve SVD, so it is very easy to deploy and accelerate.
## KM3D Baseline and Model Zoo
All experiments are tested with Ubuntu 16.04, Pytorch 1.0.0, CUDA 9.0, Python 3.6, single NVIDIA 1080Ti

IoU Setting 1: Car IoU > 0.5, Pedestrian IoU > 0.25, Cyclist IoU > 0.25

IoU Setting 2: Car IoU > 0.7, Pedestrian IoU > 0.5, Cyclist IoU > 0.5

- Training on KITTI train split and evaluation on val split.
    - Backbone: ResNet-18
    - FPS: 46.7 
    - Model: ([Google Drive](https://drive.google.com/file/d/14ww6mxtitO9aDszZN3ai8N7U1doehvi8/view?usp=sharing)), ([Baidu Cloud](https://pan.baidu.com/s/1zt-O6UzcBVGF-6vg5LzGpA) 提取码：60ks) 
    
| Class      |AP BEV IoU Setting1      | AP 3D IoU Setting1     |AP BEV IoU Setting2      | AP 3D IoU Setting2     |
| :----:     | :----:                  | :----:                 |:----:                   | :----:                 |
| -          | Easy / Moderate / Hard  | Easy / Moderate / Hard | Easy / Moderate / Hard  | Easy / Moderate / Hard |
| Car        | 55.65, 40.95, 35.61     | 49.10, 35.75, 32.27    | 23.83, 17.94, 16.98     | 17.51, 13.99, 12.73    |
| Pedestrian | 22.35, 18.50, 17.64     | 21.68, 18.13, 16.95    | 4.50, 3.87, 3.92        | 3.62, 3.75, 3.03       | 
| Cyclist    | 21.25, 15.12, 14.80     | 21.04, 14.77, 14.65    | 10.70, 9.09, 9.09       | 10.01, 9.09, 9.09      | 

- Training on KITTI train split and evaluation on val split.
    - Backbone: DLA-34
    - FPS: 28.6
    - Model: ([Google Drive](https://drive.google.com/file/d/16IjRxXtGfS1eDv9IeDZkJUUjx4olEYnK/view?usp=sharing)), ([Baidu Cloud](https://pan.baidu.com/s/1pjr-WDY256xBBusULjqL8A) 提取码：1h6s) 
    
| Class      |AP BEV IoU Setting1      | AP 3D IoU Setting1     |AP BEV IoU Setting2      | AP 3D IoU Setting2     |
| :----:     | :----:                  | :----:                 |:----:                   | :----:                 |
| -          | Easy / Moderate / Hard  | Easy / Moderate / Hard | Easy / Moderate / Hard  | Easy / Moderate / Hard |
| Car        | 60.98,  45.74,  42.93   | 54.97, 42.68, 36.95    | 25.96, 21.88, 18.88     | 19.19/ 16.70, 16.14    |
| Pedestrian | 30.38,  26.09,  23.80   | 28.63, 25.09, 20.14    | 11.55, 11.23, 10.76     | 11.37/ 10.85, 10.11    | 
| Cyclist    | 28.69,  18.77,  18.03   | 27.68, 18.30, 17.74    | 9.67, 6.12, 6.21        |  9.14/ 5.97, 5.86      | 

- Training on KITTI train split with right images augmentation and evaluation on val split.
    - Backbone: ResNet-18
    - FPS: 46.7
    - Model: ([Google Drive](https://drive.google.com/file/d/1svqj6ef79bzkiwuNIzpiLw_inDjJnSUZ/view?usp=sharing)), ([Baidu Cloud](https://pan.baidu.com/s/1gcAe2t3vmtWaST3tZPHUrg ) 提取码：sr23)
    
| Class      |AP BEV IoU Setting1      | AP 3D IoU Setting1     |AP BEV IoU Setting2      | AP 3D IoU Setting2     |
| :----:     | :----:                  | :----:                 |:----:                   | :----:                 |
| -          | Easy / Moderate / Hard  | Easy / Moderate / Hard | Easy / Moderate / Hard  | Easy / Moderate / Hard |
| Car        | 53.79, 39.83, 34.86     | 47.54, 34.97, 31.77    | 25.03, 18.53, 17.45     | 17.50, 14.06, 12.62      |
| Pedestrian | 23.15, 19.29, 18.25     | 22.33, 18.84, 17.63    | 6.21, 6.13, 5.53        | 5.19, 5.32, 4.55       | 
| Cyclist    | 19.49, 12.43, 12.28     | 19.53, 12.43, 12.28    | 10.77, 9.58, 9.59       | 10.33, 9.09, 9.09     | 

- Training on KITTI train split with right images augmentation and evaluation on val split.
    - Backbone: DLA-34
    - FPS: 28.6
    - Model: ([Google Drive](https://drive.google.com/file/d/1oVroM_VOdxvR4qkWe40T2rtahhA795h0/view?usp=sharing)), ([Baidu Cloud](https://pan.baidu.com/s/1rT46n6fajVQ_19gtkaXU4w) 提取码：qqk6) 
    
| Class      |AP BEV IoU Setting1      | AP 3D IoU Setting1     |AP BEV IoU Setting2      | AP 3D IoU Setting2     |
| :----:     | :----:                  | :----:                 |:----:                   | :----:                 |
| -          | Easy / Moderate / Hard  | Easy / Moderate / Hard | Easy / Moderate / Hard  | Easy / Moderate / Hard |
| Car        | 63.23, 50.35, 44.56     | 59.10, 44.23, 38.04    | 30.05, 23.07, 21.86     | 22.29, 17.45, 16.86    |
| Pedestrian | 32.42, 27.20, 21.51     | 31.86, 26.75, 21.33    | 14.73, 12.54, 11.74     | 12.92, 11.62, 11.06    | 
| Cyclist    | 34.64, 21.98, 22.07     | 34.01, 21.73, 19.68    | 16.89, 11.18, 10.24     |  14.35, 9.42, 9.25     | 


## Installation
Please refer to [INSTALL.md](readme/INSTALL.md)
## Dataset preparation
Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows: 
```
KM3DNet
├── kitti_format
│   ├── data
│   │   ├── kitti
│   │   |   ├── annotations 
│   │   │   ├── calib /000000.txt .....
│   │   │   ├── image(left[0-7480] right[7481-14961] input augmentatiom)
│   │   │   ├── label /000000.txt .....
|   |   |   ├── train.txt val.txt trainval.txt
├── src
├── demo_kitti_format
├── readme
├── requirements.txt
``` 
## Quick Demo
Please refer to [DEMO.md](readme/DEMO.md) for a quick demo to test with a pretrained model and visualize the predicted results on your custom data or the original KITTI data.

## Getting Started
Please refer to [GETTING_STARTED.md](readme/GETTING_STARTED.md) to learn more usage about this project.

## Acknowledgement
- [**CenterNet**](https://github.com/xingyizhou/CenterNet)
## License

RTM3D and KM3D are released under the MIT License (refer to the LICENSE file for details).
Portions of the code are borrowed from, [CenterNet](https://github.com/xingyizhou/CenterNet), [dla](https://github.com/ucbdrive/dla) (DLA network), [DCNv2](https://github.com/CharlesShang/DCNv2)(deformable convolutions), [iou3d](https://github.com/sshaoshuai/PointRCNN) and [kitti_eval](https://github.com/prclibo/kitti_eval) (KITTI dataset evaluation). Please refer to the original License of these projects (See [NOTICE](NOTICE)).
## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @misc{2009.00764,
    Author = {Peixuan Li},
    Title = {Monocular 3D Detection with Geometric Constraints Embedding and Semi-supervised Training},
    Year = {2020},
    Eprint = {arXiv:2009.00764},
    }
    @misc{2001.03343,
    Author = {Peixuan Li and Huaici Zhao and Pengfei Liu and Feidao Cao},
    Title = {RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving},
    Year = {2020},
    Eprint = {arXiv:2001.03343},
    }
    