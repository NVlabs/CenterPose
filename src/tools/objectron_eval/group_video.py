# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

"""
Create videos from image sequences (in json_save)
"""

import glob
import subprocess
import numpy as np
import os
import json
import tqdm
import functools

json_save_only = False
c = 'shoe'
save_folder = 'video'
json_save = 'hard_cases.json'
root_folder = f'debug/{c}_0/'

if os.path.isdir(f'{save_folder}'):
    print(f'folder {save_folder}/ exists')
else:
    os.mkdir(f'{save_folder}')
    print(f'created folder {save_folder}/')

video_list = glob.glob(root_folder + '*/')


# Sort according to video names
def compfunc(item1_ori, item2_ori):
    #    E.g., book_batch-1_8

    item1 = os.path.split(os.path.split(item1_ori)[0])[1]
    item2 = os.path.split(os.path.split(item2_ori)[0])[1]

    item1_num1 = int(item1[item1.find('-') + 1:item1.rfind('_')])
    item1_num2 = int(item1[item1.rfind('_') + 1:])

    item2_num1 = int(item2[item2.find('-') + 1:item2.rfind('_')])
    item2_num2 = int(item2[item2.rfind('_') + 1:])

    if item1_num1 > item2_num1:
        return 1  # larger
    elif item1_num1 == item2_num1:
        if item1_num2 > item2_num2:
            return 1
        else:
            return -1  # smaller
    else:
        return -1


video_list = sorted(video_list, key=functools.cmp_to_key(compfunc))

if os.path.exists(json_save):
    with open(json_save) as fp:
        dict_save = json.load(fp)

for video in tqdm.tqdm(video_list):
    if json_save_only and os.path.split(os.path.split(video)[0])[1] not in np.array(dict_save[c]).flatten():
        continue
    command = f'ffmpeg -y -framerate 30 -pattern_type glob -i  "{video}/*_output_pred.png"  -b 2000k -ab 1920k -crf 20 -hide_banner -loglevel panic -vb 100M {save_folder}/{os.path.split(os.path.split(video)[0])[1]}_pred.wmv'
    subprocess.call(command, shell=True)
    command = f'ffmpeg -y -framerate 30 -pattern_type glob -i  "{video}/*_output_gt.png"  -b 2000k -ab 1920k -crf 20 -hide_banner -loglevel panic -vb 100M {save_folder}/{os.path.split(os.path.split(video)[0])[1]}_gt.wmv'
    subprocess.call(command, shell=True)
print('Done')
