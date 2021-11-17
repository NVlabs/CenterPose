# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

import subprocess
import tqdm

categories = [
    "bike",
    "book",
    "bottle",
    "camera",
    "cereal_box",
    "chair",
    "cup",
    "laptop",
    "shoe"
]

OUTF = ' --outf debug/CenterPoseTrack'
REPORTF = ' --reportf report/CenterPoseTrack'
CONTINUE = ' --eval_continue'

MODE_0 = f' --eval_arch dlav1_34'

MODE_1 = f' --eval_arch dlav1_34 --eval_refined_Kalman --eval_gt_pre_hm_hmhp_first'

MODE_2 = f' --eval_arch dla_34 --eval_rep_mode 1 --eval_tracking_task ' \
         f'--eval_kalman --eval_scale_pool --eval_pre_hm --eval_pre_hm_hp'

MODE_3 = f' --eval_arch dla_34 --eval_rep_mode 1 --eval_tracking_task ' \
         f'--eval_kalman --eval_scale_pool --eval_pre_hm --eval_pre_hm_hp ' \
         f'--eval_gt_pre_hm_hmhp_first'

MODE_4 = f' --eval_arch dla_34 --eval_rep_mode 1 --eval_tracking_task ' \
         f'--eval_kalman --eval_scale_pool --eval_pre_hm --eval_pre_hm_hp ' \
         f'--eval_gt_pre_hm_hmhp_first --eval_add_noise'

MODE_5 = f' --eval_arch dla_34 --eval_rep_mode 1 --eval_tracking_task ' \
         f'--eval_kalman --eval_scale_pool --eval_pre_hm --eval_pre_hm_hp ' \
         f'--eval_gt_pre_hm_hmhp_first --eval_CenterPose_initialization'

MODE_6 = f' --eval_arch dla_34 --eval_rep_mode 1 --eval_tracking_task ' \
         f'--eval_pre_hm --eval_pre_hm_hp --eval_gt_pre_hm_hmhp_first'

MODE_7 = f' --eval_arch dla_34 --eval_rep_mode 1 --eval_tracking_task ' \
         f'--eval_kalman --eval_scale_pool --eval_gt_pre_hm_hmhp_first'

MODE_8 = f' --eval_arch dla_34 --eval_rep_mode 1 --eval_tracking_task ' \
         f'--eval_kalman --eval_scale_pool --eval_pre_hm --eval_pre_hm_hp ' \
         f'--eval_gt_pre_hm_hmhp_first --eval_empty_pre_hm'

def run(c, command):
    if c == 'bottle' or c == 'cup':
        command = command + ' --eval_num_symmetry 100'
    subprocess_ret = subprocess.Popen('python eval_video_official.py ' + command, shell=True)
    return subprocess_ret


def test_main(c, mode):
    # One by one because we have multi-processing enabled
    command = f'--eval_c {c}' + mode + CONTINUE + OUTF + REPORTF

    subprocess_ret = run(c, command)
    subprocess_ret.communicate()


if __name__ == "__main__":

    for c in tqdm.tqdm(categories):
        # CenterPose
        test_main(c, MODE_0)
        test_main(c, MODE_1)

        # Ablation
        test_main(c, MODE_6)
        # test_main(c, MODE_7) # Not used
        test_main(c, MODE_8)

        # Main
        test_main(c, MODE_3)

        # For stability experiment
        test_main(c, MODE_2)
        test_main(c, MODE_4)
        test_main(c, MODE_5)
