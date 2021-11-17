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

OUTF = ' --outf debug/CenterPose'
REPORTF = ' --reportf report/CenterPose'
CONTINUE = ' --eval_continue'

MODE_0 = f' --eval_arch dlav1_34 --eval_rep_mode 0'

MODE_1 = f' --eval_arch dlav1_34 --eval_rep_mode 1'

MODE_2 = f' --eval_arch dlav1_34 --eval_rep_mode 2'

MODE_3 = f' --eval_arch dlav1_34 --eval_rep_mode 3'

MODE_4 = f' --eval_arch dlav1_34 --eval_rep_mode 4'

MODE_5 = f' --eval_arch dla_34'

MODE_6 = f' --eval_arch dlav1_34 --eval_MobilePose_postprocessing'

MODE_7 = f' --eval_arch dlav1_34 --eval_gt_scale'


def run(c, command):
    if c == 'bottle' or c == 'cup':
        command = command + ' --eval_num_symmetry 100'
    subprocess_ret = subprocess.Popen('python eval_video_official.py ' + command, shell=True)
    return subprocess_ret


def test_main(c, mode):
    # One by one because we have multi-processing enabled
    command = f'--eval_c {c}' + mode + CONTINUE + OUTF + REPORTF
    # command = f'--eval_c {c}' + mode
    subprocess_ret = run(c, command)
    subprocess_ret.communicate()


if __name__ == "__main__":

    for c in tqdm.tqdm(categories):
        # Main experiment
        test_main(c, MODE_1)

        # rep
        test_main(c, MODE_0)
        test_main(c, MODE_2)
        test_main(c, MODE_3)
        test_main(c, MODE_4)

        # scale
        test_main(c, MODE_5)
        test_main(c, MODE_6)
        test_main(c, MODE_7)
