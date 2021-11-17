# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

"""
Gather results from all the videos while the AP result is not incorporated with predicted confidence.
"""
import random

import numpy as np
import glob
import re
import os
import tqdm
import json
import objectron.dataset.metrics as metrics

def iter_find(haystack, needle):
    return [i for i in range(0, len(haystack)) if haystack[i:].startswith(needle)]


categories = [
    "bike",
    # "book",
    # "bottle",
    # "camera",
    # "cereal_box",
    # "chair",
    # "cup",
    # "laptop",
    # "shoe"
]

_NUM_BINS = 21
_MAX_PIXEL_ERROR = 0.1
_MAX_AZIMUTH_ERROR = 30.
_MAX_POLAR_ERROR = 20.


def safe_divide(i1, i2):
    divisor = float(i2) if i2 > 0 else 1e-6
    return i1 / divisor


class Evaluator(object):
    """Class for evaluating a deep pursuit model."""

    def __init__(self):

        self._scale_ap = metrics.AveragePrecision(_NUM_BINS)
        self._iou_ap = metrics.AveragePrecision(_NUM_BINS)
        self._pixel_ap = metrics.AveragePrecision(_NUM_BINS)
        self._azimuth_ap = metrics.AveragePrecision(_NUM_BINS)
        self._polar_ap = metrics.AveragePrecision(_NUM_BINS)

        self._error_scale = 0.

        self._error_2d = 0.
        self._matched = 0

        self._iou_3d = 0.
        self._azimuth_error = 0.
        self._polar_error = 0.

        self._scale_thresholds = np.linspace(0.0, 1., num=_NUM_BINS)

        self._iou_thresholds = np.linspace(0.0, 1., num=_NUM_BINS)
        self._pixel_thresholds = np.linspace(0.0, _MAX_PIXEL_ERROR, num=_NUM_BINS)
        self._azimuth_thresholds = np.linspace(
            0.0, _MAX_AZIMUTH_ERROR, num=_NUM_BINS)
        self._polar_thresholds = np.linspace(0.0, _MAX_POLAR_ERROR, num=_NUM_BINS)

    def finalize(self):
        """Computes average precision curves."""
        self._scale_ap.compute_ap_curve()
        self._iou_ap.compute_ap_curve()
        self._pixel_ap.compute_ap_curve()
        self._azimuth_ap.compute_ap_curve()
        self._polar_ap.compute_ap_curve()

    def write_report(self, report_file=None):
        """Writes a report of the evaluation."""

        def report_array(f, label, array):
            f.write(label)
            for val in array:
                f.write('{:.4f},\t'.format(val))
            f.write('\n')

        with open(report_file, 'w') as f:
            f.write('Mean Error Scale: {}\n'.format(
                safe_divide(self._error_scale, self._matched)))
            f.write('Mean Error 2D: {}\n'.format(
                safe_divide(self._error_2d, self._matched)))
            f.write('Mean 3D IoU: {}\n'.format(
                safe_divide(self._iou_3d, self._matched)))
            f.write('Mean Azimuth Error: {}\n'.format(
                safe_divide(self._azimuth_error, self._matched)))
            f.write('Mean Polar Error: {}\n'.format(
                safe_divide(self._polar_error, self._matched)))

            f.write('\n')
            f.write('Scale Thresh: ')
            for threshold in self._scale_thresholds:
                f.write('{:.4f},\t'.format(threshold))
            f.write('\n')
            report_array(f, 'AP @Scale    : ', self._scale_ap.aps)

            f.write('\n')
            f.write('IoU Thresholds: ')
            for threshold in self._iou_thresholds:
                f.write('{:.4f},\t'.format(threshold))
            f.write('\n')
            report_array(f, 'AP @3D IoU    : ', self._iou_ap.aps)

            f.write('\n')
            f.write('2D Thresholds : ')
            for threshold in self._pixel_thresholds:
                f.write('{:.4f},\t'.format(threshold))
            f.write('\n')
            report_array(f, 'AP @2D Pixel  : ', self._pixel_ap.aps)
            f.write('\n')

            f.write('Azimuth Thresh: ')
            for threshold in self._azimuth_thresholds:
                f.write('{:.4f},\t'.format(threshold * 0.1))  # For better visualization in the txt file
            f.write('\n')
            report_array(f, 'AP @Azimuth   : ', self._azimuth_ap.aps)
            f.write('\n')

            f.write('Polar Thresh  : ')
            for threshold in self._polar_thresholds:
                f.write('{:.4f},\t'.format(threshold * 0.1))
            f.write('\n')
            report_array(f, 'AP @Polar     : ', self._polar_ap.aps)

    def update(self, data_json):
        def combine(name, a, b):
            # name is XX
            # a is XX_ap
            # b is data_json

            # Old
            a.true_positive = np.concatenate([a.true_positive, b[name]['tp']], axis=1).tolist()
            a.false_positive = np.concatenate([a.false_positive, b[name]['fp']], axis=1).tolist()

            a._total_instances += b[name]['num']

        # Update mean related
        self._error_scale += data_json['error_scale']
        self._error_2d += data_json['error_2d']
        self._matched += data_json['matched']
        self._iou_3d += data_json['iou_3d']
        self._azimuth_error += data_json['azimuth_error']
        self._polar_error += data_json['polar_error']

        # Update AP related
        combine('scale', self._scale_ap, data_json)
        combine('iou', self._iou_ap, data_json)
        combine('pixel', self._pixel_ap, data_json)
        combine('azimuth', self._azimuth_ap, data_json)
        combine('polar', self._polar_ap, data_json)


if __name__ == '__main__':


    # Current folder
    report_folder = 'report/CenterPose'

    good_list = []
    bad_list = []

    for c in categories:

        for eval_id in range(0, 4, 2):
            # ALL
            evaluator = Evaluator()
            json_list = glob.glob(f'{report_folder}/{c}_{eval_id}/*.json')
            random.shuffle(json_list)
            if json_list:
                for json_file in tqdm.tqdm(json_list):

                    # Discard results from the videos with bad annotations
                    skip_flag = False
                    for string in bad_list:
                        if string in json_file:
                            skip_flag = True
                            break
                    if skip_flag:
                        continue

                    # Only focus on the samples in the good list
                    if good_list:
                        skip_flag = True
                        for string in good_list:
                            if string in json_file:
                                skip_flag = False
                        if skip_flag:
                            continue
                        print(json_file)

                    with open(json_file) as fp:
                        data_json = json.load(fp)
                    evaluator.update(data_json)
                evaluator.finalize()
                evaluator.write_report(f'{c}_{eval_id}_combined_all.txt')
            else:
                print("No instances found!")
                continue
