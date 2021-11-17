# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

"""
Gather all the shuffled images from the offical dataset into different video tfrecord
The data would be saved in save_dir ('video_tfrecord_sorted/' by default)
"""

import tensorflow as tf
import tqdm
import os
import objectron.dataset.parser as parser
import functools
import collections
import requests
import numpy as np

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


class Objectronreader(object):
    def __init__(self, height=800, width=600):
        self.encoder = parser.ObjectronParser(height, width)


# Sort according to video names
#
def compfunc(item1, item2):
    #    E.g., (book_batch-1_8, ...)
    item1 = item1[0]
    item2 = item2[0]
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


# Equally split
#
def partition(lst, n):
    division = len(lst) / float(n) if n else len(lst)
    return [lst[int(np.round(division * i)): int(np.round(division * (i + 1)))] for i in range(n)]


# Group according to video_id & image_id, only save the videos tfrecord in the subset
#
def group_video(ds, video_ids_split):
    videos = {}
    for serialized in tqdm.tqdm(ds):

        example = tf.train.Example.FromString(serialized.numpy())

        # Group according to video_id & image_id
        video_id, image_id = reader.encoder.parse_example_basic(example)

        # Sometimes, data is too big to save, so we only focus on a small subset instead.
        if video_id not in video_ids_split:
            continue

        if video_id in videos:
            videos[video_id].append((image_id, example))
        else:
            videos[video_id] = []
            videos[video_id].append((image_id, example))

    num = 0
    # Sort frames
    for key in videos:
        videos[key].sort(key=lambda x: x[0])

        num += len(videos[key])

    videos = collections.OrderedDict(sorted(videos.items(), key=functools.cmp_to_key(compfunc)))

    # Save
    if os.path.isdir(f'{save_dir}'):
        print(f'folder {save_dir}/ exists')
    else:
        os.mkdir(f'{save_dir}')
        print(f'created folder {save_dir}/')

    if os.path.isdir(f'{save_dir}/{c}'):
        print(f'folder {save_dir}/{c} exists')
    else:
        os.mkdir(f'{save_dir}/{c}')
        print(f'created folder {save_dir}/{c}')

    for video_id in tqdm.tqdm(videos):
        with tf.io.TFRecordWriter(f'{save_dir}/{c}/{video_id}.tfrecord') as file_writer:
            for image_data in videos[video_id]:
                record_bytes = image_data[1].SerializeToString()
                file_writer.write(record_bytes)


if __name__ == '__main__':

    reader = Objectronreader()

    # objectron_buckett = 'gs://objectron/v1/records_shuffled'

    # Recommend to download the videos to local first
    objectron_buckett = 'video_tfrecord_shuffled'

    public_url = "https://storage.googleapis.com/objectron"
    for c in categories:

        # Check the lost of the testing videos
        blob_path = public_url + f"/v1/index/{c}_annotations_test"
        video_ids = requests.get(blob_path).text
        video_ids = [i.replace('/', '_') for i in video_ids.split('\n')]

        eval_data = f'/{c}/{c}_test*'
        save_dir = 'video_tfrecord_sorted'

        eval_shards = tf.io.gfile.glob(objectron_buckett + eval_data)
        ds = tf.data.TFRecordDataset(eval_shards).take(-1)

        # Work on a subset of the videos for each round, where the subset is equally split
        video_ids_split = partition(video_ids, int(np.floor(len(video_ids) / 100)))
        for i in video_ids_split:
            group_video(ds, i)

        print('Done')
