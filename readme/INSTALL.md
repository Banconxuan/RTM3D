The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v1.0.0. NVIDIA GPUs are needed for both training and testing.
After install Anaconda:
0. [Optional but recommended] create a new conda environment. 
    ~~~
    conda create --name KM3D python=3.6
    ~~~
    And activate the environment.
    ~~~
    conda activate KM3D
    ~~~
1. Install pytorch1.0.0:
    ~~~
    conda install pytorch=1.0.0 torchvision -c pytorch
    ~~~
2. Install the requirements
    ~~~
    pip install -r requirements.txt
    ~~~
3. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)).
    ~~~
    cd $KM3D_ROOT/src/lib/models/networks/ # [recommended]
    # or git clone https://github.com/CharlesShang/DCNv2/ # clone if it is not automatically downloaded by `--recursive`.
    cd DCNv2
    ./make.sh
    ~~~
4. Compile iou3d (from [pointRCNN](https://github.com/sshaoshuai/PointRCNN)). GCC>4.9, I have tested it with GCC 5.4.0 and GCC 4.9.4, both of them are ok. 
    ~~~
    cd $KM3D_ROOT/src/lib/utiles/iou3d
    python setup.py install
    ~~~
