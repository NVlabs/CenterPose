# CenterPose

![](readme/fig1.png)

## Overview

This repository is the official implementation of the paper "Single-stage Keypoint-based Category-level Object Pose Estimation from an RGB Image" by Lin et al. (full citation below). In this work, we propose a single-stage, keypoint-based approach for category-level object pose  estimation, which operates on unknown object instances within a known  category using a single RGB image input. The proposed network performs  2D  object detection,  detects 2D  keypoints,  estimates  6-DoF  pose,  and regresses relative 3D bounding cuboid dimensions.  These quantities are estimated in a sequential fashion, leveraging the recent idea of convGRU for propagating information from easier tasks to those that are more difficult.  We favor simplicity in our design choices: generic cuboid vertex coordinates, a single-stage network, and monocular  RGB  input.  We conduct extensive experiments on the challenging Objectron benchmark of real images,  outperforming state-of-the-art methods for 3D IoU metric (27.6% higher than the single-stage approach of MobilePose and 7.1% higher than the related two-stage approach). The algorithm runs at 15 fps on an NVIDIA GTX 1080Ti GPU.

## Installation

The code was tested on Ubuntu 16.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) 1.1.0. Higher versions should be possible with some accuracy difference. NVIDIA GPUs are needed for both training and testing.

1. Clone this repo:

    ~~~
    CenterPose_ROOT=/path/to/clone/CenterPose
    git clone https://github.com/NVlabs/CenterPose.git $CenterPose_ROOT
    ~~~

2. Create an Anaconda environment or create your own virtual environment
    ~~~
    conda create -n CenterPose python=3.6
    conda activate CenterPose
    pip install -r requirements.txt
    conda install -c conda-forge eigenpy
    ~~~

3. Compile the deformable convolutional layer

    ~~~
    git submodule init
    git submodule update
    cd $CenterPose_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
    ~~~

    [Optional] If you want to use a higher version of PyTorch, you need to download the latest version of [DCNv2](https://github.com/jinfagang/DCNv2_latest.git) and compile the library.
    ~~~
    git submodule set-url https://github.com/jinfagang/DCNv2_latest.git src/lib/models/networks/DCNv2
    git submodule sync
    git submodule update --init --recursive --remote
    cd $CenterPose_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
    ~~~

4. Download our [pre-trained models](https://drive.google.com/drive/folders/16HbCnUlCaPcTg4opHP_wQNPsWouUlVZe?usp=sharing) for CenterPose and move all the `.pth` files to `$CenterPose_ROOT/models/CenterPose/`.  We currently provide models for 9 categories: bike, book, bottle, camera, cereal_box, chair, cup, laptop, and shoe.

5. Prepare training/testing data

    We save all the training/testing data under `$CenterPose_ROOT/data/`.

    For the [Objectron](https://github.com/google-research-datasets/Objectron) dataset, we created our own data pre-processor to extract the data for training/testing. Refer to the [data directory](data/README.md) for more details.

## Demo

We provide supporting demos for image, videos, webcam, and image folders. See `$CenterPose_ROOT/images/CenterPose`


For category-level 6-DoF object estimation on images/video/image folders, run:

```
cd $CenterPose_ROOT/src
python demo.py --demo /path/to/image/or/folder/or/video --arch dlav1_34 --load_model ../path/to/model
```

You can also enable `--debug 4` to save all the intermediate and final outputs.

For the webcam demo (You may want to specify the camera intrinsics via --cam_intrinsic), run
```
cd $CenterPose_ROOT/src
python demo.py --demo webcam --arch dlav1_34 --load_model ../path/to/model
```

## Training

We follow the approach of [CenterNet](https://github.com/xingyizhou/CenterNet/blob/master/experiments/ctdet_coco_dla_1x.sh) for training the DLA network, reducing the learning rate by 10x after epoch 90 and 120, and stopping after 140 epochs.

For debug purposes, you can put all the local training params in the `$CenterPose_ROOT/src/main_CenterPose.py` script. You can also use the command line instead. More options are in `$CenterPose_ROOT/src/lib/opts.py`.

To start a new training job, simply do the following, which will use default parameter settings:
```
cd $CenterPose_ROOT/src
python main_CenterPose.py
```

The result will be saved in `$CenterPose_ROOT/exp/object_pose/$dataset_$category_$arch_$time` ,e.g., `objectron_bike_dlav1_34_2021-02-27-15-33`

You could then use tensorboard to visualize the training process via
```
cd $path/to/folder
tensorboard --logdir=logs --host=XX.XX.XX.XX
```

## Evaluation

We evaluate our method on the [Objectron](https://github.com/google-research-datasets/Objectron) dataset, please refer to the [objectron_eval directory](src/tools/objectron_eval/README.md) for more details.


## Citation
Please cite grasp_primitiveShape if you use this repository in your publications:

```
@article{lin2021single,
  title={Single-stage Keypoint-based Category-level Object Pose Estimation from an RGB Image},
  author={Lin, Yunzhi and Tremblay, Jonathan and Tyree, Stephen and Vela, Patricio A and Birchfield, Stan},
  journal={arXiv preprint arXiv:2109.06161},
  year={2021}
}
```

## Licence
CenterPose is licensed under the [NVIDIA Source Code License - Non-commercial](LICENSE.md).
