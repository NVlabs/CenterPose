# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

import tensorflow as tf
import tqdm
import os

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

save_dir = 'video_tfrecord_shuffled'

objectron_buckett = 'gs://objectron/v1/records_shuffled'
public_url = "https://storage.googleapis.com/objectron"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for c in categories:
    eval_data = f'/{c}/{c}_test*'

    eval_shards = tf.io.gfile.glob(objectron_buckett + eval_data)
    ds = tf.data.TFRecordDataset(eval_shards).take(-1)

    with tf.io.TFRecordWriter(f'{save_dir}/{c}.tfrecord') as file_writer:
        for serialized in tqdm.tqdm(ds):
            example = tf.train.Example.FromString(serialized.numpy())
            record_bytes = example.SerializeToString()
            file_writer.write(record_bytes)
