# CenterPose Eval for Objectron

## Overview
For a fair comparison with other methods on the Objectron benchmark, we develop our evaluation code based on the officially released [evaluation code](https://github.com/google-research-datasets/Objectron/blob/master/objectron/dataset/eval.py).

We offer two different variants:

- `eval_image_official.py` runs on the original officially released preprocessed dataset. 
(Not used in our experiment.)

- `eval_video_official.py` runs on the re-sorted officially released preprocessed dataset. 

Note that all the `load_model` paths have to be re-configured to your new location.

For more evaluation options, please refer to `eval_opts.py`

Before you start evaluation process (only for `eval_video_official.py`), please prepare the dataset first, where we provide two scripts:

- `download_test_video.py` downloads the officially released preprocessed dataset.

- `prepare_test_video.py` re-sorts the shuffled officially released preprocessed dataset to arrange them into videos.

## Utils

`group_report_new.py` collects the results from videos. It also provides the option to ignore specific samples or only collect the result from specific samples.

## Evaluation with pre-defined configurations

To evaluate on multiple categories, we wrap the evaluation code into two scripts: 

`shell_eval_image_CenterPose.py` runs on the original officially released preprocessed dataset (image).
(We do not use it for our paper.)

`shell_eval_video_CenterPose.py` and `shell_eval_video_CenterPoseTrack.py` run on the re-sorted officially released preprocessed dataset (video).

