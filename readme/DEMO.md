Here we provide a quick demo to test a pretrained model on the custom data and visualize the predicted results.

We suppose you already followed the [INSTALL.md](INSTALL.md) to install the KM3D repo successfully.

1. Download the provided pretrained models as shown in the [README.md](../README.md), and put set pretrained models in ./demo_kitti_format/exp/KM3D/
2. Run the KM3D with a pretrained model (e.g. ResNet-18train.pth) and kitti camera data as follows:
    ~~~
        cd km3d
        python ./src/faster.py --vis --demo ./demo_kitti_format/data/kitti/image --calib_dir ./demo_kitti_format/data/kitti/calib/ --load_model ./demo_kitti_format/exp/KM3D/pretrained.pth --gpus 0 --arch res_18
    ~~~
3. Run the RTM3D(GRM) with a pretrained model (e.g. ResNet-18train.pth) and kitti camera data as follows:
    ~~~
        cd km3d
        python ./src/demo.py --vis --demo ./demo_kitti_format/data/kitti/image --calib_dir ./demo_kitti_format/data/kitti/calib/ --load_model ./demo_kitti_format/exp/KM3D/pretrained.pth --gpus 0 --arch res_18
    ~~~
4. Run the KM3D with a pretrained model (e.g. ResNet-18train.pth) and custom camera data as follows:
    ~~~
        cd km3d
        python ./src/demo.py --vis --demo ~/your image folder --calib_dir ~/your calib folder/ --load_model ~/pretrained.pth --gpus 0 --arch res_18 or dla_34
    ~~~
5. Run the RTM3D(GRM) with a pretrained model (e.g. ResNet-18train.pth) and custom camera data as follows:
    ~~~
        cd km3d
        python ./src/demo.py --vis --demo ~/your image folder --calib_dir ~/your calib folder/ --load_model ~/pretrained.pth --gpus 0 --arch res_18 or dla_34
    ~~~
