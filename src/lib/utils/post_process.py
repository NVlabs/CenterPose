# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds

def object_pose_post_process(dets, c, s, h, w, opt, Inference=False):

    # A scale factor
    coefficient = 0.32

    # Scale bbox & pts and Regroup
    if not ('scores' in dets):
        return [[{}]]

    ret = []
    for i in range(dets['scores'].shape[0]):

        preds = []

        for j in range(len(dets['scores'][i])):
            item = {}
            item['score'] = float(dets['scores'][i][j])
            item['cls'] = int(dets['clses'][i][j])
            item['obj_scale'] = dets['obj_scale'][i][j]
            item['obj_scale_uncertainty'] = dets['obj_scale_uncertainty'][i][j]

            kps_displacement_std = dets['kps_displacement_std'][i, j] * (s[i] / max(w, h)) * coefficient
            item['kps_displacement_std'] = kps_displacement_std.reshape(-1, 16).flatten()

            # from w,h to c[i], s[i]
            bbox = transform_preds(dets['bboxes'][i, j].reshape(-1, 2), c[i], s[i], (w, h))
            item['bbox'] = bbox.reshape(-1, 4).flatten()

            item['ct'] = [(item['bbox'][0] + item['bbox'][2]) / 2, (item['bbox'][1] + item['bbox'][3]) / 2]

            kps = transform_preds(dets['kps'][i, j].reshape(-1, 2), c[i], s[i], (w, h))
            item['kps'] = kps.reshape(-1, 16).flatten()

            tracking = dets['tracking'][i, j] * (s[i] / max(w, h))
            item['tracking'] = tracking.reshape(-1, 2).flatten()

            tracking_hp = dets['tracking_hp'][i, j] * (s[i] / max(w, h))
            item['tracking_hp'] = tracking_hp.reshape(-1, 16).flatten()

            # To save some time, only perform this step when it is inference time
            if Inference == True:
                kps_displacement_mean = transform_preds(dets['kps_displacement_mean'][i, j].reshape(-1, 2), c[i], s[i],
                                                        (w, h))
                item['kps_displacement_mean'] = kps_displacement_mean.reshape(-1, 16).flatten()

                kps_heatmap_mean = transform_preds(dets['kps_heatmap_mean'][i, j].reshape(-1, 2), c[i], s[i], (w, h))
                item['kps_heatmap_mean'] = kps_heatmap_mean.reshape(-1, 16).flatten()

                kps_heatmap_std = dets['kps_heatmap_std'][i, j] * (s[i] / max(w, h)) * coefficient
                item['kps_heatmap_std'] = kps_heatmap_std.reshape(-1, 16).flatten()

                item['kps_heatmap_height'] = dets['kps_heatmap_height'][i, j]

            preds.append(item)

        ret.append(preds)
    return ret
