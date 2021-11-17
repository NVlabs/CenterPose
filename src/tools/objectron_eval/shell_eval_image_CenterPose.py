# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

import subprocess
import tqdm

categories = [
    "bike",
    "book",
    "camera",
    "cereal_box",
    "chair",
    "laptop",
    "shoe",
    "bottle",
    "cup",
]


def run(c, command):
    if c == 'bottle' or c == 'cup':
        command = command + ' --eval_num_symmetry 100'
    subprocess_list = subprocess.Popen('python eval_image_official.py ' + command, shell=True)
    return subprocess_list


def test_main():
    subprocess_list = []
    for c in tqdm.tqdm(categories):
        print(c)

        command = f'--eval_c {c} --eval_arch dlav1_34 --eval_rep_mode 1 --eval_confidence_thresh 0.3 '
        subprocess_list.append(run(c, command))
        command = f'--eval_c {c} --eval_arch dla_34 --eval_rep_mode 1 --eval_confidence_thresh 0.3'
        subprocess_list.append(run(c, command))

        Checking = True
        while Checking:
            n = 0
            for p in subprocess_list:
                if p.poll() is None:
                    n = n + 1
            if n < 6:
                Checking = False

    Checking = True
    while Checking:
        n = 0
        for p in subprocess_list:
            if p.poll() is None:
                n = n + 1
        if n == 0:
            Checking = False


def test_ablation_scale():
    subprocess_list = []
    for c in tqdm.tqdm(categories):
        print(c)

        command = f'--eval_c {c} --eval_arch dlav1_34 --eval_rep_mode 1 --eval_MobilePose_postprocessing'
        subprocess_list.append(run(c, command))
        command = f'--eval_c {c} --eval_arch dlav1_34 --eval_rep_mode 1 --eval_gt_scale'
        subprocess_list.append(run(c, command))

        Checking = True
        while Checking:
            n = 0
            for p in subprocess_list:
                if p.poll() is None:
                    n = n + 1
            if n < 6:
                Checking = False

    # For final run
    Checking = True
    while Checking:
        n = 0
        for p in subprocess_list:
            if p.poll() is None:
                n = n + 1
        if n == 0:
            Checking = False


def test_ablation_rep():
    subprocess_list = []
    for c in tqdm.tqdm(categories):
        print(c)

        command = f'--eval_c {c} --eval_arch dlav1_34 --eval_rep_mode 0'
        subprocess_list.append(run(c, command))
        command = f'--eval_c {c} --eval_arch dlav1_34 --eval_rep_mode 2'
        subprocess_list.append(run(c, command))

        Checking = True
        while Checking:
            n = 0
            for p in subprocess_list:
                if p.poll() is None:
                    n = n + 1
            if n < 6:
                Checking = False

        command = f'--eval_c {c} --eval_arch dlav1_34 --eval_rep_mode 3'
        subprocess_list.append(run(c, command))
        command = f'--eval_c {c} --eval_arch dlav1_34 --eval_rep_mode 4'
        subprocess_list.append(run(c, command))

        Checking = True
        while Checking:
            n = 0
            for p in subprocess_list:
                if p.poll() is None:
                    n = n + 1
            if n < 6:
                Checking = False

    # For final run
    Checking = True
    while Checking:
        n = 0
        for p in subprocess_list:
            if p.poll() is None:
                n = n + 1
        if n == 0:
            Checking = False


if __name__ == "__main__":
    test_main()
    test_ablation_scale()
    test_ablation_rep()
