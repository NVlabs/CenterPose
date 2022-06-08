# CenterPose Toolbox for Objectron

The basic idea of this toolbox is to preprocess data from the [Objectron dataset](https://github.com/google-research-datasets/Objectron)  in a format compatible with [DOPE](https://github.com/NVlabs/Deep_Object_Pose).  In other words, pairs of image-annotation files, (**XX.img**, **XX.json**).

We have manually labeled the symmetric instances in both the chair and cup categories. The new labels also enforce consistency of the coordinate frame orientation with respect to the mug handle. All info is saved in **label/**.

Most of the utility library comes from the official release of Objectron, with our new code below:

- `download.py` downloads the original data, including videos and their corresponding annotations. 

- `preprocess.py` preprocesses the downloaded data. We provide multiple options. For more details, refer to the script.

Example code:

`python download.py --c chair`

`python preprocess.py --c chair --outf outf_all --frame_rate 1`

Note that we set frame_rate as 15 for CenterPose while we set frame_rate as 1 for CenterPoseTrack. 