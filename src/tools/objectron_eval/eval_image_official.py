# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

"""
We develop our own evaluation script from the official example Evaluation script for
Objectron dataset. It reads a tfrecord, runs evaluation, and outputs a summary report with name
specified in report_file argument. When adopting this for your own model, you
have to modify the Evaluator.predict() function, which basically takes an image and produces
a 3D bounding box.
"""

import eigenpy

eigenpy.switchToNumpyArray()

import math
import os
import warnings

import argparse

import glob
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as rotation_util
import tensorflow as tf
import tqdm

import objectron.dataset.iou as IoU3D
import objectron.dataset.box as Box

# We incorporate confidence into the AP calculation
# METRIC_UPDATED = False
METRIC_UPDATED = True
if METRIC_UPDATED:
    import objectron.dataset.metrics_nvidia as metrics
else:
    import objectron.dataset.metrics as metrics

import objectron.dataset.parser as parser

import sys
sys.path.insert(0, '../..')

from lib.utils.pnp.cuboid_pnp_shell import pnp_shell

import shutil
import simplejson as json

from lib.detectors.detector_factory import detector_factory
from lib.opts import opts

import cv2

from eval_opts import eval_opts
from eval_utils import draw_axes

_MAX_PIXEL_ERROR = 0.1
_MAX_AZIMUTH_ERROR = 30.
_MAX_POLAR_ERROR = 20.
_MAX_SCALE_ERROR = 2.
_MAX_DISTANCE = 1.0  # In meters
_NUM_BINS = 21

dimension_ref = {
    'bike': [[0.65320896, 1.021797894, 1.519635599, 0.6520559199, 1.506392621],
             [0.1179380561, 0.176747817, 0.2981715678, 0.1667947895, 0.3830536275]],
    'book': [[0.225618019, 0.03949624326, 0.1625821624, 7.021850281, 5.064694187],
             [0.1687487664, 0.07391230822, 0.06436673199, 3.59629568, 2.723290812]],
    'bottle': [
        [0.07889784977450116, 0.24127451915330908, 0.0723714257114412, 0.33644069262302545, 0.3091134992864717, ],
        [0.02984649578071775, 0.06381390122918497, 0.03088144838560917, 0.11052240441921059,
         0.13327627592012867, ]],
    'camera': [[0.11989848700326843, 0.08226238775595619, 0.09871718158089632, 1.507216484439368, 1.1569407159290284, ],
               [0.021177290310316968, 0.02158788017191602, 0.055673710278419844, 0.28789183678046854,
                0.5342094080365904, ]],
    'cereal_box': [
        [0.19202754401417296, 0.2593114001714919, 0.07723794925413519, 0.7542602699204104, 0.29441151268928173, ],
        [0.08481640897407464, 0.09999915952084068, 0.09495429981036707, 0.19829004029411457, 0.2744797990483879, ]],
    'chair': [[0.5740664085137888, 0.8434027515832329, 0.6051523831888338, 0.6949691013776601, 0.7326891354260606, ],
              [0.12853104253707456, 0.14852086453095492, 0.13428881418587957, 0.16897092539619352,
               0.18636134566748525, ]],
    'cup': [[0.08587637391801063, 0.12025228955138188, 0.08486836104868696, 0.7812126934904675, 0.7697895244331658, ],
            [0.05886805978497525, 0.06794896438246326, 0.05875681990718713, 0.2887038681446475, 0.283821205157399, ]],
    'mug': [[0.14799136566553112, 0.09729087667918128, 0.08845449667169905, 1.3875694883045138, 1.0224997119392225, ],
            [1.0488828523223728, 0.2552672927963539, 0.039095350310480705, 0.3947832854104711, 0.31089415283872546, ]],
    'laptop': [[0.33685059747485196, 0.1528068814247063, 0.2781020624738614, 35.920214652427696, 23.941173992376903, ],
               [0.03529983948867832, 0.07017080198389423, 0.0665823136876069, 391.915687801732, 254.21325950495455, ]],
    'shoe': [[0.10308848289662519, 0.10932616184503478, 0.2611737789760352, 1.0301976264129833, 2.6157393112424328, ],
             [0.02274768925924402, 0.044958380226590516, 0.04589720205423542, 0.3271000267177176,
              0.8460337534776092, ]],
}

epnp_alpha_default = np.array([4.0, -1.0, -1.0, -1.0, 2.0, -1.0, -1.0, 1.0, 2.0,
                               -1.0, 1.0, -1.0, 0.0, -1.0, 1.0, 1.0, 2.0, 1.0, -1.0, -1.0,
                               0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, -2.0, 1.0, 1.0,
                               1.0]).reshape(8, 4)


def CHECK_EQ(a, b):
    if a != b:
        print('Error!')
        exit()


def safe_divide(i1, i2):
    divisor = float(i2) if i2 > 0 else 1e-6
    return i1 / divisor


def rotation_y_matrix(theta):
    M_R = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                    [0, 1, 0, 0],
                    [-np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]])
    return M_R


# For debug
import matplotlib.pyplot as plt

RADIUS = 10
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0), (128, 0, 128),
          (0, 128, 128), (255, 255, 255), (0, 0, 0)]

from mpl_toolkits import mplot3d


class Evaluator(object):
    """Class for evaluating a deep pursuit model."""

    def __init__(self, opt, height=1920, width=1440):
        self.opt = opt
        self.height, self.width = int(height / self.opt.eval_resolution_ratio), int(
            width / self.opt.eval_resolution_ratio)
        self.encoder = parser.ObjectronParser(self.opt.c, self.height, self.width)
        self._vis_thresh = 0.1
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
        self._add_thresholds = np.linspace(0.0, _MAX_DISTANCE, num=_NUM_BINS)
        self._adds_thresholds = np.linspace(0.0, _MAX_DISTANCE, num=_NUM_BINS)

        self._scale_ap = metrics.AveragePrecision(_NUM_BINS)
        self._iou_ap = metrics.AveragePrecision(_NUM_BINS)
        self._pixel_ap = metrics.AveragePrecision(_NUM_BINS)
        self._azimuth_ap = metrics.AveragePrecision(_NUM_BINS)
        self._polar_ap = metrics.AveragePrecision(_NUM_BINS)
        self._add_ap = metrics.AveragePrecision(_NUM_BINS)
        self._adds_ap = metrics.AveragePrecision(_NUM_BINS)

        # Init the detector

        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpus_str
        Detector = detector_factory[self.opt.task]

        if self.opt.c != 'cup':
            self.detector = Detector(self.opt)
        else:
            # Two detectors for the cup category
            if 'v1' in self.opt.arch:
                self.opt.load_model = f"../../../models/CenterPose/cup_mug_v1_140.pth"
                self.detector_mug = Detector(self.opt)
                self.opt.load_model = f"../../../models/CenterPose/cup_cup_v1_sym_12_140.pth"
                self.detector_cup = Detector(self.opt)
            else:
                self.opt.load_model = f"../../../models/CenterPose/cup_mug_140.pth"
                self.detector_mug = Detector(self.opt)
                self.opt.load_model = f"../../../models/CenterPose/cup_cup_sym_12_140.pth"
                self.detector_cup = Detector(self.opt)

        if self.opt.c == 'cup' and self.opt.mug_only:
            self.opt.dimension_ref = dimension_ref['mug']
        else:
            self.opt.dimension_ref = dimension_ref[self.opt.c]

        if self.opt.eval_use_absolute_scale:
            self.opt.dimension_ref = self.opt.dimension_ref[0][0:3]
        else:
            # Relative scale
            self.opt.dimension_ref = [self.opt.dimension_ref[0][3], 1, self.opt.dimension_ref[0][4]]

        # Sample id starts from 1
        self.NUM_SAMPLE = 0

        self.filename_list = []

    def predict(self, image, camera_matrix, projection_matrix, filename, sample_id, MugFlag_instance=[]):
        """
          Implement your own function/model to predict the box's 2D and 3D
          keypoint from the input images.
          Note that the predicted 3D bounding boxes are correct upto an scale.
          You can use the ground planes to re-scale your boxes if necessary.

          Returns:
            A list of list of boxes for objects in images in the batch. Each box is
            a tuple of (point_2d, point_3d) that includes the predicted 2D and 3D vertices.
        """

        # Todo: For safety, one image one time for now, need to be updated

        meta = {}
        meta['camera_matrix'] = camera_matrix

        # Change to BGR space
        image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        if MugFlag_instance == []:
            ret = self.detector.run(image, filename=filename + f'_{sample_id}', meta_inp=meta)
        elif MugFlag_instance == True:
            ret = self.detector_mug.run(image, filename=filename + f'_{sample_id}', meta_inp=meta)
        elif MugFlag_instance == False:
            ret = self.detector_cup.run(image, filename=filename + f'_{sample_id}', meta_inp=meta)

        boxes = ret['boxes']

        return boxes

    def predict_gt_scale(self, bbox, scale, camera_matrix):

        meta = {}
        meta['camera_matrix'] = camera_matrix
        meta['width'] = self.width
        meta['height'] = self.height

        points = np.array(bbox['kps']).reshape(-1, 2)
        points = [(x[0], x[1]) for x in points]
        ret = pnp_shell(self.opt, meta, bbox, points, np.array(scale) / scale[1])
        if ret is not None:
            return ret
        else:
            return None

    def evaluate(self, batch):
        """Evaluates a batch of serialized tf.Example protos."""
        images, labels, projs, cam_intrinsics, planes, views = [], [], [], [], [], []
        filenames = []
        for serialized in batch:
            example = tf.train.Example.FromString(serialized)
            image, label, filename = self.encoder.parse_example(example)
            proj, view, cam_intrinsic = self.encoder.parse_camera(example)

            images.append(image)
            labels.append(label)
            filenames.append(filename)
            projs.append(proj)
            views.append(view)
            cam_intrinsics.append(cam_intrinsic)
            plane = self.encoder.parse_plane(example)
            planes.append(plane)

            # Save to check the number of views
            self.filename_list.append(filename)

        # It can be incorporated into the next for block if we support batch processing.
        # Since we use pnp here, not valid for now.
        local_id = 0
        results = []
        for image, label, cam_intrinsic, filename, projection_matrix, view in zip(images, labels, cam_intrinsics,
                                                                                  filenames, projs, views):

            local_id = local_id + 1
            if self.NUM_SAMPLE % self.opt.batch_size == 0:
                global_id = self.NUM_SAMPLE - self.opt.batch_size + local_id
            else:
                global_id = self.NUM_SAMPLE - self.NUM_SAMPLE % self.opt.batch_size + local_id

            if type(self.opt.eval_skip) is list:
                if global_id in self.opt.eval_skip:
                    results.append([])
                    continue
            else:
                if global_id < self.opt.eval_skip:
                    results.append([])
                    continue

            if self.opt.eval_subset_list is not None and global_id not in self.opt.eval_subset_list:
                results.append([])
                continue

            # The camera intrinsics have to be updated
            cam_intrinsic[:2, :3] = cam_intrinsic[:2, :3] / self.opt.eval_resolution_ratio
            cx = cam_intrinsic[0, 2]
            cy = cam_intrinsic[1, 2]
            cam_intrinsic[0, 2] = cy
            cam_intrinsic[1, 2] = cx

            if self.opt.c == 'cup':
                if all(label['MugFlag_instance']) == True:
                    results.append(self.predict(image, cam_intrinsic, projection_matrix, filename, global_id, True))
                elif all(np.invert(label['MugFlag_instance'])) == True:
                    results.append(self.predict(image, cam_intrinsic, projection_matrix, filename, global_id, False))
                else:
                    results.append(self.predict(image, cam_intrinsic, projection_matrix, filename, global_id, True))
                    # Todo: May assume that we already know it is cup or mug

            else:
                results.append(self.predict(image, cam_intrinsic, projection_matrix, filename, global_id))

        local_id = 0
        for boxes, label, plane, image, filename, cam_intrinsic, projection_matrix, view in zip(results, labels, planes,
                                                                                                images, filenames,
                                                                                                cam_intrinsics, projs,
                                                                                                views):
            local_id = local_id + 1
            if self.NUM_SAMPLE % self.opt.batch_size == 0:
                global_id = self.NUM_SAMPLE - self.opt.batch_size + local_id
            else:
                global_id = self.NUM_SAMPLE - self.NUM_SAMPLE % self.opt.batch_size + local_id

            if type(self.opt.eval_skip) is list:
                if global_id in self.opt.eval_skip:
                    continue
            else:
                if global_id < self.opt.eval_skip:
                    continue

            if self.opt.eval_subset_list is not None and global_id not in self.opt.eval_subset_list:
                continue

            # Extract gt info
            instances_scale = label['scale_instance']
            instances_2d = label['2d_instance']
            instances_3d = label['3d_instance']
            instances_Mo2c = label['Mo2c_instance']
            if self.opt.c == 'cup':
                instances_MugFlag = label['MugFlag_instance']

                if self.opt.mug_only == True:
                    # Only count the case with mug
                    if all(np.invert(label['MugFlag_instance'])) == True:
                        continue


                elif self.opt.mug_only == False:
                    # Only count the case with cup
                    if all(np.invert(label['MugFlag_instance'])) == False:
                        continue

            visibilities = label['visibility']

            num_instances = 0
            for instance, instance_3d, visibility in zip(
                    instances_2d, instances_3d, visibilities):
                if (visibility > self._vis_thresh and
                        self._is_visible(instance[0]) and instance_3d[0, 2] < 0):
                    num_instances += 1

            # We don't have negative examples in evaluation.
            if num_instances == 0:
                continue

            scale_hit_miss = metrics.HitMiss(self._scale_thresholds)
            iou_hit_miss = metrics.HitMiss(self._iou_thresholds)
            azimuth_hit_miss = metrics.HitMiss(self._azimuth_thresholds)
            polar_hit_miss = metrics.HitMiss(self._polar_thresholds)
            pixel_hit_miss = metrics.HitMiss(self._pixel_thresholds)
            add_hit_miss = metrics.HitMiss(self._add_thresholds)
            adds_hit_miss = metrics.HitMiss(self._adds_thresholds)

            # For debug
            pred_box_list = []
            gt_box_list = []

            # Save gt info for Stephen
            M_c2w = np.linalg.inv(view)
            dict_save = {
                'filename': filename,
                'camera_pose': M_c2w.tolist(),  # M_c2w
                'camera_intrinsics': cam_intrinsic.tolist(),  # has been transformed to list
                'image_id': int(label['image_id']),
                "objects": [],
            }

            num_matched = 0
            for idx_box, box in enumerate(boxes):

                # Correspond to one prediction in one image
                box_point_2d, box_point_3d, relative_scale, box_point_2d_ori, result_ori = box

                # Todo:
                if self.opt.eval_MobilePose_postprocessing == True:
                    box_point_2d, box_point_3d = self.Lift2DTo3D(projection_matrix, result_ori, image.shape[0],
                                                                 image.shape[1])

                index = self.match_box(box_point_2d, instances_2d, visibilities)
                if index >= 0:
                    num_matched += 1

                    # Apply gt_scale to recalculate pnp
                    if self.opt.eval_gt_scale == True:
                        result_gt_scale = self.predict_gt_scale(result_ori, instances_scale[index], cam_intrinsic)
                        if result_gt_scale is not None:
                            box_point_2d, box_point_3d, _, _, _ = result_gt_scale

                    # If you only compute the 3D bounding boxes from RGB images,
                    # your 3D keypoints may be upto scale. However the ground truth
                    # is at metric scale. There is a hack to re-scale your box using
                    # the ground planes (assuming your box is sitting on the ground).
                    # However many models learn to predict depths and scale correctly.

                    if not self.opt.use_absolute_scale:
                        scale = self.compute_scale(box_point_3d, plane)
                        box_point_3d = box_point_3d * scale
                        boxes[idx_box] = list(boxes[idx_box])
                        boxes[idx_box].append(box_point_3d)

                    print(f'Sample {global_id}')
                    print(f'GT: {instances_scale[index] / instances_scale[index][1]}')
                    print(f'Pred: {relative_scale / relative_scale[1]}')
                    if self.opt.c == 'cup':
                        pixel_error = self.evaluate_2d(box_point_2d, instances_2d[index], instances_3d[index],
                                                       instances_Mo2c[index], projection_matrix,
                                                       instances_MugFlag[index])
                        azimuth_error, polar_error, iou, pred_box, gt_box, add, adds = self.evaluate_3d(box_point_3d,
                                                                                                        instances_3d[
                                                                                                            index],
                                                                                                        instances_Mo2c[
                                                                                                            index],
                                                                                                        instances_MugFlag[
                                                                                                            index])


                    else:
                        pixel_error = self.evaluate_2d(box_point_2d, instances_2d[index], instances_3d[index],
                                                       instances_Mo2c[index], projection_matrix)
                        azimuth_error, polar_error, iou, pred_box, gt_box, add, adds = self.evaluate_3d(box_point_3d,
                                                                                                        instances_3d[
                                                                                                            index],
                                                                                                        instances_Mo2c[
                                                                                                            index])

                    # Record some predictions
                    M_o2w = M_c2w @ instances_Mo2c[index]
                    instances_3d_w = M_c2w @ np.hstack(
                        (instances_3d[index], np.ones((instances_3d[index].shape[0], 1)))).T

                    instances_3d_w = instances_3d_w[:3, :].T

                    keypoint_2d_gt = [np.multiply(keypoint, np.asarray([self.width, self.height], np.float32)) for
                                      keypoint in instances_2d[index]]

                    result_pnp = [np.multiply(keypoint, np.asarray([self.width, self.height], np.float32)) for
                                  keypoint in box_point_2d]

                    scale_error = self.evaluate_scale(relative_scale, instances_scale[index])

                    print(f'Scale_error: {scale_error}')
                    print('\n')

                    dict_obj = {
                        'class': self.opt.c,
                        'keypoint_2d_pred_displacement': np.array(result_ori['kps_displacement_mean']).reshape(1,
                                                                                                               -1).tolist(),
                        'keypoint_2d_pred_heatmap': np.array(result_ori['kps_heatmap_mean']).reshape(1, -1).tolist(),
                        'keypoint_2d_pred_pnp': np.array(result_pnp).reshape(1, -1).tolist(),
                        'keypoint_2d_gt': np.array(keypoint_2d_gt).reshape(1, -1).tolist(),
                        'relative_scale': relative_scale.tolist(),
                        'object_pose_gt_w': M_o2w.tolist(),  # 4x4 matrix
                        'keypoint_3d_gt_w': instances_3d_w.tolist(),  # 9x3 array
                        'keypoint_3d_pred_unscaled_c': np.array(boxes[idx_box][1]).reshape(1, -1).tolist(),  # 27 list
                        'keypoint_3d_pred_scaled_c': np.array(boxes[idx_box][5]).reshape(1, -1).tolist(),  # 27 list
                        '3DIoU': iou,
                        'error_2Dpixel': pixel_error,
                        'error_azimuth': azimuth_error,
                        'error_polar_error': polar_error,
                        'error_scale': scale_error
                    }
                    dict_save['objects'].append(dict_obj)

                    pred_box_list.append(pred_box)
                    gt_box_list.append(gt_box)

                    conf = result_ori['score']

                else:
                    conf = 0
                    pixel_error = _MAX_PIXEL_ERROR
                    azimuth_error = _MAX_AZIMUTH_ERROR
                    polar_error = _MAX_POLAR_ERROR
                    iou = 0.
                    add = _MAX_DISTANCE
                    adds = _MAX_DISTANCE
                    scale_error = _MAX_SCALE_ERROR

                if METRIC_UPDATED:
                    # New
                    scale_hit_miss.record_hit_miss([scale_error, conf], greater=False)
                    iou_hit_miss.record_hit_miss([iou, conf])
                    add_hit_miss.record_hit_miss([add, conf], greater=False)
                    adds_hit_miss.record_hit_miss([adds, conf], greater=False)
                    pixel_hit_miss.record_hit_miss([pixel_error, conf], greater=False)
                    azimuth_hit_miss.record_hit_miss([azimuth_error, conf], greater=False)
                    polar_hit_miss.record_hit_miss([polar_error, conf], greater=False)
                else:
                    # Old
                    scale_hit_miss.record_hit_miss(scale_error, greater=False)
                    iou_hit_miss.record_hit_miss(iou)
                    add_hit_miss.record_hit_miss(add, greater=False)
                    adds_hit_miss.record_hit_miss(adds, greater=False)
                    pixel_hit_miss.record_hit_miss(pixel_error, greater=False)
                    azimuth_hit_miss.record_hit_miss(azimuth_error, greater=False)
                    polar_hit_miss.record_hit_miss(polar_error, greater=False)

            if self.opt.eval_debug_json == True:
                json_filename = f'{self.opt.outf}/{self.opt.c}_{self.opt.eval_save_id}/{filename}_{global_id}_record.json'
                with open(json_filename, 'w+') as fp:
                    json.dump(dict_save, fp, indent=4, sort_keys=True)

            # For debug
            if self.opt.eval_debug == True:
                # if self.opt.eval_debug == True and iou<self.opt.eval_debug_save_thresh:
                self.debug(image.copy(), num_instances, instances_2d, instances_3d, projection_matrix, boxes,
                           instances_scale, filename, pred_box_list, gt_box_list, global_id)

            self._scale_ap.append(scale_hit_miss, len(instances_2d))
            self._iou_ap.append(iou_hit_miss, len(instances_2d))
            self._pixel_ap.append(pixel_hit_miss, len(instances_2d))
            self._azimuth_ap.append(azimuth_hit_miss, len(instances_2d))
            self._polar_ap.append(polar_hit_miss, len(instances_2d))
            self._add_ap.append(add_hit_miss, len(instances_2d))
            self._adds_ap.append(adds_hit_miss, len(instances_2d))
            self._matched += num_matched

    def draw_boxes(self, filename, sample_id, boxes=[], clips=[], colors=['b', 'g', 'r', 'k']):
        """Draw a list of boxes.

            The boxes are defined as a list of vertices
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        for i, b in enumerate(boxes):
            x, y, z = b[:, 0], b[:, 1], b[:, 2]
            ax.scatter(x, y, z, c='r')

            for idx, pos in enumerate(zip(x, y, z)):
                ax.text(pos[0], pos[1], pos[2], f'{idx}')

            for e in Box.EDGES:
                ax.plot(x[e], y[e], z[e], linewidth=2, c=colors[i % len(colors)])

            for e in Box.BOTTOM:
                ax.plot(x[e], y[e], z[e], linewidth=2, c=colors[i % len(colors)])

        if (len(clips)):
            points = np.array(clips)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=100, c='k')

        plt.gca().patch.set_facecolor('white')
        ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
        ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
        ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))

        # rotate the axes and update
        ax.view_init(30, 12)
        ax.set_box_aspect((1, 1, 1))
        plt.draw()

        if self.opt.eval_debug_display:
            plt.show()

        plt.savefig(f'{self.opt.outf}/{self.opt.c}_{self.opt.eval_save_id}/{filename}_{sample_id}_3DIoU.png',
                    bbox_inches='tight')

    def debug(self, image_src, num_instances, instances_2d, instances_3d, projection_matrix, boxes, instances_scale,
              filename, pred_box_list, gt_box_list, sample_id):

        # cv2.imwrite(f'{self.opt.outf}/{self.opt.c}/{filename}_{sample_id}_output_ori.png',
        #             cv2.cvtColor(image_debug, cv2.COLOR_RGB2BGR))

        image_debug = image_src.copy()
        # GT label Green
        for object_id in range(num_instances):
            for kp_id in range(1, 9):
                kp_pixel = instances_2d[object_id, kp_id, :]
                cv2.circle(image_debug,
                           (int(image_debug.shape[1] * kp_pixel[0]), int(image_debug.shape[0] * kp_pixel[1])),
                           RADIUS, (0, 255, 0), -1)

            for edge in Box.EDGES:
                start_kp = instances_2d[object_id, edge[0], :]
                start_x = int(image_debug.shape[1] * start_kp[0])
                start_y = int(image_debug.shape[0] * start_kp[1])

                end_kp = instances_2d[object_id, edge[1], :]
                end_x = int(image_debug.shape[1] * end_kp[0])
                end_y = int(image_debug.shape[0] * end_kp[1])

                cv2.line(image_debug, (start_x, start_y), (end_x, end_y),
                         (0, 255, 0), 3)

            # Draw pose axes
            draw_axes(image_debug, instances_3d[object_id], projection_matrix, image_debug.shape[0],
                      image_debug.shape[1])

        cv2.imwrite(f'{self.opt.outf}/{self.opt.c}_{self.opt.eval_save_id}/{filename}_{sample_id}_output_gt.png',
                    cv2.cvtColor(image_debug, cv2.COLOR_RGB2BGR))

        image_debug = image_src.copy()
        # PnP results Blue
        for object_id in range(len(boxes)):
            for kp_id in range(1, 9):
                kp_pixel = boxes[object_id][0][kp_id, :]
                cv2.circle(image_debug,
                           (int(image_debug.shape[1] * kp_pixel[0]), int(image_debug.shape[0] * kp_pixel[1])),
                           RADIUS, (0, 255, 255), -1)

            for edge in Box.EDGES:
                start_kp = boxes[object_id][0][edge[0], :]
                start_x = int(image_debug.shape[1] * start_kp[0])
                start_y = int(image_debug.shape[0] * start_kp[1])

                end_kp = boxes[object_id][0][edge[1], :]
                end_x = int(image_debug.shape[1] * end_kp[0])
                end_y = int(image_debug.shape[0] * end_kp[1])

                cv2.line(image_debug, (start_x, start_y), (end_x, end_y),
                         (0, 255, 255), 3)

            # Sometimes, the predicted result does not have a match, e.g., a chair who is only partially visible
            if len(boxes) >= 5:
                draw_axes(image_debug, boxes[object_id][5], projection_matrix, image_debug.shape[0],
                          image_debug.shape[1])
        cv2.imwrite(f'{self.opt.outf}/{self.opt.c}_{self.opt.eval_save_id}/{filename}_{sample_id}_output_pred.png',
                    cv2.cvtColor(image_debug, cv2.COLOR_RGB2BGR))

        # # Original 2D points Red
        # for object_id in range(len(boxes)):
        #     for kp_id in range(9):
        #         kp_pixel = boxes[object_id][3][kp_id, :]
        #         cv2.circle(image_debug,
        #                    (int(image_debug.shape[1] * kp_pixel[0]), int(image_debug.shape[0] * kp_pixel[1])),
        #                    RADIUS, (255, 0, 0), -1)
        #
        #     for edge in Box.EDGES:
        #         start_kp = boxes[object_id][3][edge[0], :]
        #         start_x = int(image_debug.shape[1] * start_kp[0])
        #         start_y = int(image_debug.shape[0] * start_kp[1])
        #
        #         end_kp = boxes[object_id][3][edge[1], :]
        #         end_x = int(image_debug.shape[1] * end_kp[0])
        #         end_y = int(image_debug.shape[0] * end_kp[1])
        #
        #         cv2.line(image_debug, (start_x, start_y), (end_x, end_y),
        #                  (255, 0, 0), 2)

        # Save output with
        # cv2.imwrite(f'{self.opt.outf}/{self.opt.c}/{filename}_{sample_id}_output.png', cv2.cvtColor(image_debug, cv2.COLOR_RGB2BGR))

        # # Show/Save 3D IoU
        # for b1, b2 in zip(pred_box_list, gt_box_list):
        #     if b1 is not None and b2 is not None:
        #         self.draw_boxes(filename,sample_id, [b1.vertices, b2.vertices])

    def evaluate_scale(self, relative_scale, instance):
        relative_scale_normalized = relative_scale / relative_scale[1]
        instance_normalized = instance / instance[1]

        error = np.sum(np.absolute(relative_scale_normalized - instance_normalized) / instance_normalized)
        # error = np.mean(np.linalg.norm(relative_scale_normalized - instance_normalized))
        self._error_scale += error
        return error

    def evaluate_2d(self, box, instance_2d, instance_3d, Mo2c, proj, instances_MugFlag=[]):
        """Evaluates a pair of 2D projections of 3D boxes.

        It computes the mean normalized distances of eight vertices of a box.

        Args:
          box: A 9*2 array of a predicted box.
          instance_2d: A 9*2 array of an annotated box.
          instance_3d: A 9*3 array of an annotated box.
          Mo2c: A gt transformation matrix from object frame to camera frame
          proj: Projection matrix
          instances_MugFlag: A Flag if the object is a mug or not

        Returns:
          Pixel error
        """

        # error = np.mean(np.linalg.norm(box[1:] - instance_2d[1:], axis=1))
        # self._error_2d += error
        #
        # return error

        Mc2o = np.linalg.inv(Mo2c)
        error_best = np.inf
        for id_symmetry in range(self.opt.eval_num_symmetry):
            theta = 2 * np.pi / self.opt.eval_num_symmetry
            M_R = rotation_y_matrix(theta * id_symmetry)
            M_trans = proj @ Mo2c @ M_R @ Mc2o
            instance_new = M_trans @ np.hstack((instance_3d, np.ones((instance_3d.shape[0], 1)))).T

            pp2 = (instance_new / instance_new[3])[:2]
            viewport_point = (pp2 + 1.0) / 2.0
            viewport_point[[0, 1]] = viewport_point[[1, 0]]
            instance_new = viewport_point.T
            error = np.mean(np.linalg.norm(box[1:] - instance_new[1:], axis=1))
            if error_best > error:
                print(f'{id_symmetry}: {error}')
                error_best = error

            if self.opt.eval_mug_symmetric == False:
                # If instances_MugFlag == [] or False, loop with eval_num_symmetry
                if instances_MugFlag == True:
                    break

        self._error_2d += error_best

        return error_best

    def _get_rotated_box(self, box_point_3d, angle):
        """Rotate a box along its vertical axis.
        Args:
          box: Input box.
          angle: Rotation angle in rad.
        Returns:
          A rotated box
        """
        CENTER = 0
        BACK_TOP_LEFT = 3
        BACK_BOTTOM_LEFT = 1
        up_vector = box_point_3d[BACK_TOP_LEFT] - box_point_3d[BACK_BOTTOM_LEFT]
        rot_vec = angle * up_vector / np.linalg.norm(up_vector)
        rotation = rotation_util.from_rotvec(rot_vec).as_dcm()
        box_center = box_point_3d[CENTER]
        box_point_3d_rotated = np.matmul((box_point_3d - box_center), rotation) + box_center
        return box_point_3d_rotated

    def evaluate_3d(self, box_point_3d, instance_3d, Mo2c, instances_MugFlag=[]):
        """Evaluates a box in 3D.

        It computes metrics of view angle and 3D IoU.

        Args:
          box: A predicted box.
          instance_3d: A 9*3 array of an annotated box, in metric level.
          Mo2c: A transformation matrix from object frame to camera frame
          instances_MugFlag: A Flag if the object is a mug or not
        Returns:
          The 3D IoU (float)
        """
        # azimuth_error, polar_error = self.evaluate_viewpoint(box_point_3d, instance)
        # iou = self.evaluate_iou(box_point_3d, instance)
        # return azimuth_error, polar_error, iou

        azimuth_error, polar_error = self.evaluate_viewpoint(box_point_3d, instance_3d)
        avg_distance, avg_sym_distance = self.compute_average_distance(box_point_3d,
                                                                       instance_3d)
        Mc2o = np.linalg.inv(Mo2c)
        iou_best = 0
        pred_box_best = None
        gt_box_best = None
        avg_distance_best = _MAX_DISTANCE
        avg_sym_distance_best = _MAX_DISTANCE

        # Adapted from the official one: rotate the estimated one
        for id_symmetry, theta in enumerate(np.linspace(0, np.pi * 2, self.opt.eval_num_symmetry)):

            box_point_3d_rotated = self._get_rotated_box(box_point_3d, theta)
            iou, pred_box, gt_box = self.evaluate_iou(box_point_3d_rotated, instance_3d)

            if iou > iou_best:
                azimuth_error, polar_error = self.evaluate_viewpoint(box_point_3d_rotated,
                                                                     instance_3d)
                avg_distance, avg_sym_distance = self.compute_average_distance(box_point_3d_rotated,
                                                                               instance_3d)
                print(f'{id_symmetry}: {iou}/{azimuth_error}/{polar_error}/{avg_distance}/{avg_sym_distance}')
                iou_best = iou
                pred_box_best = pred_box
                gt_box_best = gt_box
                avg_distance_best = avg_distance
                avg_sym_distance_best = avg_sym_distance

            if self.opt.eval_mug_symmetric == False:
                # If instances_MugFlag == [] or False, loop with eval_num_symmetry
                if instances_MugFlag == True:
                    break

        self._iou_3d += iou_best
        self._azimuth_error += azimuth_error
        self._polar_error += polar_error

        return azimuth_error, polar_error, iou_best, pred_box_best, gt_box_best, avg_distance_best, avg_sym_distance_best

    def compute_scale(self, box, plane):
        """Computes scale of the given box sitting on the plane."""
        center, normal = plane
        vertex_dots = [np.dot(vertex, normal) for vertex in box[1:]]
        vertex_dots = np.sort(vertex_dots)
        center_dot = np.dot(center, normal)
        scales = center_dot / vertex_dots[:4]

        return np.mean(scales)

    def Lift2DTo3D(self, projection_matrix, estimated_box, height, width, epnp_alpha_=epnp_alpha_default):

        fx = projection_matrix[0, 0]
        fy = projection_matrix[1, 1]
        cx = projection_matrix[0, 2]
        cy = projection_matrix[1, 2]

        m = np.zeros((16, 12))
        u = None
        v = None

        keypoint2d_list = estimated_box['kps'].flatten()
        for i in range(8):
            v = (keypoint2d_list[i * 2] / width) * 2 - 1
            u = keypoint2d_list[i * 2 + 1] / height * 2 - 1

            for j in range(4):
                # For each of the 4 control points, formulate two rows of the
                # m matrix (two equations).
                control_alpha = epnp_alpha_[i, j]

                m[i * 2, j * 3] = fx * control_alpha
                m[i * 2, j * 3 + 2] = (cx + u) * control_alpha
                m[i * 2 + 1, j * 3 + 1] = fy * control_alpha
                m[i * 2 + 1, j * 3 + 2] = (cy + v) * control_alpha

        mt_m = m.transpose() @ m
        es = eigenpy.SelfAdjointEigenSolver(mt_m)
        V = es.eigenvectors()
        D = es.eigenvalues()

        CHECK_EQ(12, len(D))

        eigen_vec = V[:, 0]
        control_matrix = eigen_vec.reshape(4, 3)

        if control_matrix[0, 2] > 0:
            control_matrix = -control_matrix

        keypoint3d_list = []
        keypoint3d_list.append([control_matrix[0, 0], control_matrix[0, 1], control_matrix[0, 2]])

        vertices = epnp_alpha_ @ control_matrix

        for i in range(8):
            keypoint3d_list.append([vertices[i, 0], vertices[i, 1], vertices[i, 2]])

        keypoint2d_list = []
        for keypoint3d in keypoint3d_list:
            # Official OpenGL way
            k_3d = np.array([keypoint3d[0], keypoint3d[1], keypoint3d[2], 1])
            pp2 = np.matmul(projection_matrix, k_3d.reshape(4, 1))
            pp2 = (pp2 / pp2[3])[:3]
            viewport_point = (pp2 + 1.0) / 2.0
            viewport_point = [viewport_point[1][0], viewport_point[0][0]]
            keypoint2d_list.append(viewport_point)

        return np.array(keypoint2d_list), np.array(keypoint3d_list)

    def compute_ray(self, box):
        """Computes a ray from camera to box centroid in box frame.

        For vertex in camera frame V^c, and object unit frame V^o, we have
          R * Vc + T = S * Vo,
        where S is a 3*3 diagonal matrix, which scales the unit box to its real size.

        In fact, the camera coordinates we get have scale ambiguity. That is, we have
          Vc' = 1/beta * Vc, and S' = 1/beta * S
        where beta is unknown. Since all box vertices should have negative Z values,
        we can assume beta is always positive.

        To update the equation,
          R * beta * Vc' + T = beta * S' * Vo.

        To simplify,
          R * Vc' + T' = S' * Vo,
        where Vc', S', and Vo are known. The problem is to compute
          T' = 1/beta * T,
        which is a point with scale ambiguity. It forms a ray from camera to the
        centroid of the box.

        By using homogeneous coordinates, we have
          M * Vc'_h = (S' * Vo)_h,
        where M = [R|T'] is a 4*4 transformation matrix.

        To solve M, we have
          M = ((S' * Vo)_h * Vc'_h^T) * (Vc'_h * Vc'_h^T)_inv.
        And T' = M[:3, 3:].

        Args:
          box: A 9*3 array of a 3D bounding box.

        Returns:
          A ray represented as [x, y, z].
        """
        if box[0, -1] > 0:
            warnings.warn('Box should have negative Z values.')

        size_x = np.linalg.norm(box[5] - box[1])
        size_y = np.linalg.norm(box[3] - box[1])
        size_z = np.linalg.norm(box[2] - box[1])
        size = np.asarray([size_x, size_y, size_z])
        box_o = Box.UNIT_BOX * size
        box_oh = np.ones((4, 9))
        box_oh[:3] = np.transpose(box_o)

        box_ch = np.ones((4, 9))
        box_ch[:3] = np.transpose(box)
        box_cht = np.transpose(box_ch)

        box_oct = np.matmul(box_oh, box_cht)
        try:
            box_cct_inv = np.linalg.inv(np.matmul(box_ch, box_cht))
        except:
            box_cct_inv = np.linalg.pinv(np.matmul(box_ch, box_cht))

        transform = np.matmul(box_oct, box_cct_inv)
        return transform[:3, 3:].reshape((3))

    def compute_average_distance(self, box, instance):
        """Computes Average Distance (ADD) metric."""
        add_distance = 0.
        for i in range(Box.NUM_KEYPOINTS):
            delta = np.linalg.norm(box[i, :] - instance[i, :])
            add_distance += delta
        add_distance /= Box.NUM_KEYPOINTS

        # Computes the symmetric version of the average distance metric.
        # From PoseCNN https://arxiv.org/abs/1711.00199
        # For each keypoint in predicttion, search for the point in ground truth
        # that minimizes the distance between the two.
        add_sym_distance = 0.
        for i in range(Box.NUM_KEYPOINTS):
            # Find nearest vertex in instance
            distance = np.linalg.norm(box[i, :] - instance[0, :])
            for j in range(Box.NUM_KEYPOINTS):
                d = np.linalg.norm(box[i, :] - instance[j, :])
                if d < distance:
                    distance = d
            add_sym_distance += distance
        add_sym_distance /= Box.NUM_KEYPOINTS

        return add_distance, add_sym_distance

    def compute_viewpoint(self, box):
        """Computes viewpoint of a 3D bounding box.

        We use the definition of polar angles in spherical coordinates
        (http://mathworld.wolfram.com/PolarAngle.html), expect that the
        frame is rotated such that Y-axis is up, and Z-axis is out of screen.

        Args:
          box: A 9*3 array of a 3D bounding box.

        Returns:
          Two polar angles (azimuth and elevation) in degrees. The range is between
          -180 and 180.
        """
        x, y, z = self.compute_ray(box)
        theta = math.degrees(math.atan2(z, x))
        phi = math.degrees(math.atan2(y, math.hypot(x, z)))
        return theta, phi

    def evaluate_viewpoint(self, box, instance):
        """Evaluates a 3D box by viewpoint.

        Args:
          box: A 9*3 array of a predicted box.
          instance: A 9*3 array of an annotated box, in metric level.
        Returns:
          Two viewpoint angle errors.
        """
        predicted_azimuth, predicted_polar = self.compute_viewpoint(box)
        gt_azimuth, gt_polar = self.compute_viewpoint(instance)

        polar_error = abs(predicted_polar - gt_polar)
        # Azimuth is from (-180,180) and a spherical angle so angles -180 and 180
        # are equal. E.g. the azimuth error for -179 and 180 degrees is 1'.
        # azimuth_error = abs(predicted_azimuth - gt_azimuth)

        # Todo: May need further updates, e.g., inf symmetry
        azimuth_error = abs(predicted_azimuth - gt_azimuth) % (360 / self.opt.eval_num_symmetry)

        if azimuth_error > 180:
            azimuth_error = 360 - azimuth_error

        # Todo: Add later
        # self._azimuth_error += azimuth_error
        # self._polar_error += polar_error
        return azimuth_error, polar_error

    def evaluate_rotation(self, box, instance):
        """Evaluates rotation of a 3D box.

        1. The L2 norm of rotation angles
        2. The rotation angle computed from rotation matrices
              trace(R_1^T R_2) = 1 + 2 cos(theta)
              theta = arccos((trace(R_1^T R_2) - 1) / 2)

        3. The rotation angle computed from quaternions. Similar to the above,
           except instead of computing the trace, we compute the dot product of two
           quaternion.
             theta = 2 * arccos(| p.q |)
           Note the distance between quaternions is not the same as distance between
           rotations.

        4. Rotation distance from "3D Bounding box estimation using deep learning
           and geometry""
               d(R1, R2) = || log(R_1^T R_2) ||_F / sqrt(2)

        Args:
          box: A 9*3 array of a predicted box.
          instance: A 9*3 array of an annotated box, in metric level.

        Returns:
          Magnitude of the rotation angle difference between the box and instance.
        """
        prediction = Box.Box(box)
        annotation = Box.Box(instance)
        gt_rotation_inverse = np.linalg.inv(annotation.rotation)
        rotation_error = np.matmul(prediction.rotation, gt_rotation_inverse)

        error_angles = np.array(
            rotation_util.from_dcm(rotation_error).as_euler('zxy'))
        abs_error_angles = np.absolute(error_angles)
        abs_error_angles = np.minimum(
            abs_error_angles, np.absolute(math.pi * np.ones(3) - abs_error_angles))
        error = np.linalg.norm(abs_error_angles)

        # Compute the error as the angle between the two rotation
        rotation_error_trace = abs(np.matrix.trace(rotation_error))
        angular_distance = math.acos((rotation_error_trace - 1.) / 2.)

        # angle = 2 * acos(|q1.q2|)
        box_quat = np.array(rotation_util.from_dcm(prediction.rotation).as_quat())
        gt_quat = np.array(rotation_util.from_dcm(annotation.rotation).as_quat())
        quat_distance = 2 * math.acos(np.dot(box_quat, gt_quat))

        # The rotation measure from "3D Bounding box estimation using deep learning
        # and geometry"
        rotation_error_log = scipy.linalg.logm(rotation_error)
        rotation_error_frob_norm = np.linalg.norm(rotation_error_log, ord='fro')
        rotation_distance = rotation_error_frob_norm / 1.4142

        return (error, quat_distance, angular_distance, rotation_distance)

    def evaluate_iou(self, box, instance):
        """Evaluates a 3D box by 3D IoU.

        It computes 3D IoU of predicted and annotated boxes.

        Args:
          box: A 9*3 array of a predicted box.
          instance: A 9*3 array of an annotated box, in metric level.

        Returns:
          3D Intersection over Union (float)
        """
        # Computes 3D IoU of the two boxes.
        prediction = Box.Box(box)
        annotation = Box.Box(instance)
        iou = IoU3D.IoU(prediction, annotation)

        try:
            iou_result = iou.iou()
        except:
            iou_result = 0
        # Todo: Add values in the end
        # self._iou_3d += iou_result
        return iou_result, prediction, annotation

    def match_box(self, box, instances, visibilities):
        """Matches a detected box with annotated instances.

        For a predicted box, finds the nearest annotation in instances. This means
        we always assume a match for a prediction. If the nearest annotation is
        below the visibility threshold, the match can be skipped.

        Args:
          box: A 9*2 array of a predicted box.
          instances: A ?*9*2 array of annotated instances. Each instance is a 9*2
            array.
          visibilities: An array of the visibilities of the instances.

        Returns:
          Index of the matched instance; otherwise -1.
        """
        norms = np.linalg.norm(instances[:, 1:, :] - box[1:, :], axis=(1, 2))
        i_min = np.argmin(norms)
        if visibilities[i_min] < self._vis_thresh:
            return -1
        return i_min

    def write_report(self, report_file=None):
        """Writes a report of the evaluation."""

        def report_array(f, label, array):
            f.write(label)
            for val in array:
                f.write('{:.4f},\t'.format(val))
            f.write('\n')

        if report_file == None:
            report_file = self.opt.report_file

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
            f.write('\n')

            f.write('ADD Thresh    : ')
            for threshold in self._add_thresholds:
                f.write('{:.4f},\t'.format(threshold))
            f.write('\n')
            report_array(f, 'AP @ADD       : ', self._add_ap.aps)
            f.write('\n')

            f.write('ADDS Thresh   : ')
            for threshold in self._adds_thresholds:
                f.write('{:.4f},\t'.format(threshold))
            f.write('\n')
            report_array(f, 'AP @ADDS      : ', self._adds_ap.aps)

    def finalize(self):
        """Computes average precision curves."""
        self._scale_ap.compute_ap_curve()
        self._iou_ap.compute_ap_curve()
        self._pixel_ap.compute_ap_curve()
        self._azimuth_ap.compute_ap_curve()
        self._polar_ap.compute_ap_curve()
        self._add_ap.compute_ap_curve()
        self._adds_ap.compute_ap_curve()

    def _is_visible(self, point):
        """Determines if a 2D point is visible."""
        return point[0] > 0 and point[0] < 1 and point[1] > 0 and point[1] < 1


def main(opt):
    # Todo: Resize the image to what we use, easier to preprocess the camera intrinsics directly
    evaluator = Evaluator(opt)
    objectron_buckett = 'gs://objectron/v1/records_shuffled'
    eval_shards = tf.io.gfile.glob(objectron_buckett + opt.eval_data)
    ds = tf.data.TFRecordDataset(eval_shards).take(opt.eval_max_num)

    batch = []
    for serialized in tqdm.tqdm(ds):
        # For debug
        evaluator.NUM_SAMPLE = evaluator.NUM_SAMPLE + 1

        batch.append(serialized.numpy())
        if len(batch) == opt.batch_size:
            evaluator.evaluate(batch)
            batch.clear()

        # if evaluator.NUM_SAMPLE in [1000,5000,10000,20000,50000,100000]:
        #     evaluator.finalize()
        #     evaluator.write_report( report_file= os.path.splitext(opt.report_file)[0] + f'_{evaluator.NUM_SAMPLE}.txt')

    if batch:
        evaluator.evaluate(batch)

    evaluator.finalize()
    evaluator.write_report()


if __name__ == '__main__':

    # Default param setting for opt_eval
    opt_eval = eval_opts().parser.parse_args()

    # Default param setting for opt_detector
    opt_detector = opts().parser.parse_args([])

    # Todo: Some important settings to change, should be commented if not using Pycharm but .sh instead
    opt_eval.outf = 'debug/CenterPose'
    # opt_eval.eval_num_symmetry = 1
    # opt_eval.eval_confidence_thresh=0.3
    opt_eval.eval_debug_save_thresh = 100
    opt_eval.eval_max_num = 10000

    # # # Comment this when using shell script
    # opt_eval.eval_c = 'bike'
    # opt_eval.eval_arch = 'dlav1_34'
    # opt_eval.eval_rep_mode = 1
    # opt_eval.eval_MobilePose_postprocessing = True
    # opt_eval.eval_gt_scale = True

    # More options:
    opt_detector.nms = True

    # Read the subset
    if opt_eval.eval_subset == True:
        json_path = 'selected_frames.json'
        with open(json_path) as f:
            data_json = json.load(f)
        opt_eval.eval_subset_list = list(map(int, data_json[opt_eval.eval_c]))
        opt_eval.eval_subset_list.sort()
    else:
        opt_eval.eval_subset_list = None

    # For debug purpose
    # opt_eval.eval_skip=35 # Run images on [eval_skip, eval_max_num]
    # opt_detector.batch_size=1
    # opt_eval.eval_debug = True # Whether to save img for debug
    opt_eval.eval_debug_json = True  # Whether to save json for debug
    opt_eval.eval_debug_clean = True

    opt_detector.debug = 0  # do not save extra visualization in demo/ for debug, e.g., heatmap
    # opt_detector.debug = 6  # save extra visualization in demo/ for debug, e.g., heatmap

    # Objectron paper https://arxiv.org/abs/2012.09988 assumes mug is also symmetric, for fair comparison we also have this option
    opt_eval.eval_mug_symmetric = True

    # True: only evaluate mug case, False: only evaluate cup case, None: Evaluate them all
    opt_eval.mug_only = None

    # Align opts from opt_detector with ones from opt_eval
    opt_detector.c = opt_eval.eval_c
    opt_detector.arch = opt_eval.eval_arch
    opt_detector.rep_mode = opt_eval.eval_rep_mode
    opt_detector.vis_thresh = opt_eval.eval_confidence_thresh

    # No symmetry
    if 'v1' in opt_detector.arch:
        opt_eval.report_file = f'{opt_detector.c}_v1_report_{opt_eval.eval_confidence_thresh}.txt'
        opt_detector.load_model = f"../../../models/CenterPose/{opt_detector.c}_v1_{opt_eval.eval_weight_id}.pth"
    else:
        opt_eval.report_file = f'{opt_detector.c}_report_{opt_eval.eval_confidence_thresh}.txt'
        opt_detector.load_model = f"../../../models/CenterPose/{opt_detector.c}_{opt_eval.eval_weight_id}.pth"

    # Symmetry exists, just bottle while cup has been hard-coded
    if opt_detector.c == 'bottle':
        if 'v1' in opt_detector.arch:
            opt_eval.report_file = f'{opt_detector.c}_v1_{opt_eval.eval_num_symmetry}_sym_report_{opt_eval.eval_confidence_thresh}.txt'
            opt_detector.load_model = f"../../../models/CenterPose/{opt_detector.c}_v1_sym_12_{opt_eval.eval_weight_id}.pth"
        else:
            opt_eval.report_file = f'{opt_detector.c}_report_{opt_eval.eval_confidence_thresh}.txt'
            opt_detector.load_model = f"../../../models/CenterPose/{opt_detector.c}_sym_12_{opt_eval.eval_weight_id}.pth"

    # Some exp naming rules
    opt_eval.eval_save_id = opt_eval.eval_rep_mode
    if opt_detector.nms == True:
        opt_eval.report_file = os.path.splitext(opt_eval.report_file)[0] + '_nms.txt'
    if opt_detector.rep_mode == 0:
        opt_eval.report_file = os.path.splitext(opt_eval.report_file)[0] + '_8rep.txt'
    elif opt_detector.rep_mode == 1:
        opt_eval.report_file = os.path.splitext(opt_eval.report_file)[0] + '_16rep.txt'
    elif opt_detector.rep_mode == 2:
        opt_eval.report_file = os.path.splitext(opt_eval.report_file)[0] + '_samplerep.txt'
    elif opt_detector.rep_mode == 3:
        opt_eval.report_file = os.path.splitext(opt_eval.report_file)[0] + '_disrep.txt'
    elif opt_detector.rep_mode == 4:
        opt_eval.report_file = os.path.splitext(opt_eval.report_file)[0] + '_hmrep.txt'

    if opt_eval.eval_arch == 'dla_34':
        opt_eval.eval_save_id = 5
    if opt_eval.eval_MobilePose_postprocessing == True:
        opt_eval.report_file = os.path.splitext(opt_eval.report_file)[0] + '_MobilePose.txt'
        opt_eval.eval_save_id = 6
    if opt_eval.eval_gt_scale == True:
        opt_eval.report_file = os.path.splitext(opt_eval.report_file)[0] + '_gtscale.txt'
        opt_eval.eval_save_id = 7
    if opt_eval.eval_mug_symmetric == False:
        opt_eval.report_file = os.path.splitext(opt_eval.report_file)[0] + '_partsymmetry.txt'
        opt_eval.eval_save_id = 8

    opt_eval.report_file = os.path.splitext(opt_eval.report_file)[0] + f'_{opt_eval.eval_max_num}.txt'
    opt_eval.report_file = os.path.splitext(opt_eval.report_file)[0] + f'_{opt_eval.eval_exp_id}.txt'
    opt_eval.report_file = os.path.splitext(opt_eval.report_file)[0] + f'_{opt_eval.eval_weight_id}.txt'

    if opt_eval.eval_debug == True or opt_eval.eval_debug_json == True:
        if opt_eval.eval_debug_clean == True and opt_eval.eval_continue != True:
            # Clean up debug/
            if os.path.isdir(f'{opt_eval.outf}/{opt_detector.c}_{opt_eval.eval_save_id}'):
                shutil.rmtree(f'{opt_eval.outf}/{opt_detector.c}_{opt_eval.eval_save_id}')
            # Clean up demo/
            if os.path.exists(
                    os.path.join('demo/', f'{os.path.splitext(os.path.basename(opt_detector.load_model))[0]}')):
                shutil.rmtree(
                    os.path.join('demo/', f'{os.path.splitext(os.path.basename(opt_detector.load_model))[0]}'))

        if os.path.isdir(f'{opt_eval.outf}'):
            print(f'folder {opt_eval.outf}/ exists')
        else:
            os.mkdir(f'{opt_eval.outf}')
            print(f'created folder {opt_eval.outf}/')

        if os.path.isdir(f'{opt_eval.outf}/{opt_detector.c}_{opt_eval.eval_save_id}'):
            print(f'folder {opt_eval.outf}/{opt_detector.c}_{opt_eval.eval_save_id} exists')
        else:
            os.mkdir(f'{opt_eval.outf}/{opt_detector.c}_{opt_eval.eval_save_id}')
            print(f'created folder {opt_eval.outf}/{opt_detector.c}_{opt_eval.eval_save_id}')

    opt_detector.obj_scale = True
    opt_detector.use_pnp = True

    opt_detector = opts().parse(opt_detector)
    opt_detector = opts().init(opt_detector)
    opt_detector.eval_data = f'/{opt_detector.c}/{opt_detector.c}_test*'

    opt_combined = argparse.Namespace(**vars(opt_eval), **vars(opt_detector))
    main(opt_combined)
