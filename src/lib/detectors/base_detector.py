# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
from progress.bar import Bar
import time
import torch
import math

from lib.models.model import create_model, load_model
from lib.utils.image import get_affine_transform, affine_transform
from lib.utils.debugger import Debugger

from lib.utils.pnp.cuboid_pnp_shell import pnp_shell

from lib.utils.tracker import Tracker
from lib.utils.tracker_baseline import Tracker_baseline

from lib.utils.image import draw_umich_gaussian, gaussian_radius, draw_nvidia_gaussian
from sklearn import mixture
import scipy


class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True

        self.pre_images = None

        if opt.tracking_task:
            self.tracker = Tracker(opt)

        if opt.refined_Kalman:
            self.tracker = Tracker_baseline(opt)

    # def process(self, images, return_time=False):
    def process(self, images, pre_images=None, pre_hms=None,
                pre_inds=None, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1, pre_hms=None, pre_hm_hp=None):
        raise NotImplementedError

    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def save_results(self, debugger, image, results, image_or_path_or_tensor):
        raise NotImplementedError

    def _trans_bbox(self, bbox, trans, width, height):
        '''
        Transform bounding boxes according to image crop.
        '''
        bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
        bbox[:2] = affine_transform(bbox[:2], trans)
        bbox[2:] = affine_transform(bbox[2:], trans)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
        return bbox

    def pre_process(self, image, scale, input_meta={}):
        '''
              Prepare input image in different testing modes.
                Currently support: fix short size/ center crop to a fixed size/
                keep original resolution but pad to a multiplication of 32
        '''
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)

        if self.opt.fix_short > 0:
            if height < width:
                inp_height = self.opt.fix_short
                inp_width = (int(width / height * self.opt.fix_short) + 63) // 64 * 64
            else:
                inp_height = (int(height / width * self.opt.fix_short) + 63) // 64 * 64
                inp_width = self.opt.fix_short
            c = np.array([width / 2, height / 2], dtype=np.float32)
            s = np.array([width, height], dtype=np.float32)
        elif self.opt.fix_res:
            # We usually use this
            inp_height, inp_width = self.opt.input_h, self.opt.input_w
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
            # s = np.array([inp_width, inp_height], dtype=np.float32)
        else:
            inp_height = (new_height | self.opt.pad) + 1
            inp_width = (new_width | self.opt.pad) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])

        out_height = inp_height // self.opt.down_ratio
        out_width = inp_width // self.opt.down_ratio
        trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

        images = torch.from_numpy(images)
        meta = {'c': c, 's': s, 'height': height, 'width': width,
                'out_height': out_height, 'out_width': out_width,
                'inp_height': inp_height, 'inp_width': inp_width,
                'trans_input': trans_input, 'trans_output': trans_output}

        if 'pre_dets' in input_meta:
            meta['pre_dets'] = input_meta['pre_dets']
        if 'camera_matrix' in input_meta:
            meta['camera_matrix'] = input_meta['camera_matrix']
        if 'id' in input_meta:
            meta['id'] = input_meta['id']
        return images, meta

    def _get_additional_inputs(self, dets, meta, with_hm=True, with_hm_hp=True):
        '''
        Render input heatmap from previous trackings.
        '''

        # Note that XX like 'trans_input' is a transform relationship
        trans_input, trans_output = meta['trans_input'], meta['trans_output']
        inp_width, inp_height = meta['inp_width'], meta['inp_height']
        out_width, out_height = meta['out_width'], meta['out_height']
        ori_width, ori_height = meta['width'], meta['height']

        input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32) if with_hm else None
        input_hm_hp = np.zeros((8, inp_height, inp_width), dtype=np.float32) if with_hm_hp else None

        output_inds = []

        if self.opt.empty_pre_hm == False:
            if self.opt.gt_pre_hm_hmhp == True or (self.opt.gt_pre_hm_hmhp_first == True and meta['id'] == 0):
                # Only need det['bbox'] & det['kps_gt'] (9 of them normalized)
                for det in dets:
                    bbox = self._trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
                    bbox_out = self._trans_bbox(
                        det['bbox'], trans_output, out_width, out_height)
                    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    if (h > 0 and w > 0):
                        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                        radius = max(0, int(radius))
                        ct = np.array(
                            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                        ct_int = ct.astype(np.int32)
                        if with_hm:
                            draw_umich_gaussian(input_hm[0], ct_int, radius)
                        ct_out = np.array(
                            [(bbox_out[0] + bbox_out[2]) / 2,
                             (bbox_out[1] + bbox_out[3]) / 2], dtype=np.int32)
                        output_inds.append(ct_out[1] * out_width + ct_out[0])
                        if with_hm_hp:
                            # Todo: Currently, hp_radius follows the same way as radius
                            hp_radius = radius
                            # hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                            # hp_radius = self.opt.hm_gauss \
                            #     if self.opt.mse_loss else max(0, int(hp_radius))

                            pts_ori = np.array(det['kps_gt'][1:])
                            pts_ori[:, 0] = pts_ori[:, 0] * ori_width
                            pts_ori[:, 1] = pts_ori[:, 1] * ori_height

                            # Change visibility, following the protocal of COCO
                            pts = np.zeros((8, 3), dtype='int64')
                            for idx, p in enumerate(pts_ori):
                                if p[0] >= ori_width or p[0] < 0 or p[1] < 0 or p[1] >= ori_height:
                                    pts[idx] = [p[0], p[1], 1]  # labeled but not visible
                                else:
                                    pts[idx] = [p[0], p[1], 2]  # labeled and visible

                            for j in range(8):
                                pts[j, :2] = affine_transform(pts[j, :2], trans_input)

                                pt_int = pts[j, :2].astype(np.int32)
                                draw_umich_gaussian(input_hm_hp[j], pt_int, hp_radius, k=1)
            else:
                if self.opt.use_pnp:
                    for det in dets:
                        if det['score'] < self.opt.pre_thresh:
                            continue

                        bbox = self._trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
                        bbox_out = self._trans_bbox(
                            det['bbox'], trans_output, out_width, out_height)

                        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                        if (h > 0 and w > 0):
                            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                            radius = max(0, int(radius))
                            ct = np.array(
                                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                            ct_int = ct.astype(np.int32)
                            if with_hm:
                                if self.opt.render_hm_mode == 0:
                                    draw_umich_gaussian(input_hm[0], ct_int, radius)
                                elif self.opt.render_hm_mode == 1:
                                    draw_umich_gaussian(input_hm[0], ct_int, radius, k=det['score'])
                            ct_out = np.array(
                                [(bbox_out[0] + bbox_out[2]) / 2,
                                 (bbox_out[1] + bbox_out[3]) / 2], dtype=np.int32)
                            output_inds.append(ct_out[1] * out_width + ct_out[0])

                            if with_hm_hp:
                                # Todo: Currently, hp_radius follows the same way as radius
                                hp_radius = radius
                                # hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                                # hp_radius = self.opt.hm_gauss \
                                #     if self.opt.mse_loss else max(0, int(hp_radius))

                                if self.opt.render_hmhp_mode == 0 or self.opt.render_hmhp_mode == 1:
                                    pts_ori = np.array(det['kps_ori'][1:])
                                elif self.opt.render_hmhp_mode == 2 or self.opt.render_hmhp_mode == 3:
                                    if self.opt.kalman == True or self.opt.scale_pool == True:
                                        if 'kps_pnp_kf' in det:
                                            pts_ori = np.array(det['kps_pnp_kf'][1:])
                                        else:
                                            pts_ori = np.array(det['kps_mean_kf'][1:])
                                    else:
                                        if 'kps_pnp' in det:
                                            pts_ori = np.array(det['kps_pnp'][1:])
                                        else:
                                            # No result if pnp failed
                                            pts_ori = np.zeros((8, 2))
                                pts_ori[:, 0] = pts_ori[:, 0] * ori_width
                                pts_ori[:, 1] = pts_ori[:, 1] * ori_height

                                # Todo: Seems not working
                                if self.opt.hps_uncertainty == True:
                                    radius_detector = (np.array(det['kps_fusion_std']).reshape(-1, 2)).astype(
                                        np.int32)
                                else:
                                    radius_detector = (np.array(det['kps_heatmap_std']).reshape(-1, 2)).astype(
                                        np.int32)

                                #  Todo: Need further update
                                if self.opt.kalman == True and 'kf' in det:
                                    conf_hp_detector = []
                                    for i in range(8):
                                        std_combined = np.sqrt(
                                            (det['kf'].P[4 * i, 4 * i]) + (det['kf'].P[4 * i + 1, 4 * i + 1]))
                                        # std_combined = 15 * np.sqrt(
                                        #     (det['kf'].P[4 * i, 4 * i]) / (det['bbox'][2] - det['bbox'][0])
                                        #     + (det['kf'].P[4 * i + 1, 4 * i + 1]) / (det['bbox'][3] - det['bbox'][1]))
                                        conf_hp_detector.append(np.maximum(1 - np.exp(np.log(0.15) / (
                                                self.opt.conf_border[self.opt.c][0] - self.opt.conf_border[self.opt.c][
                                            1])) ** (std_combined - self.opt.conf_border[self.opt.c][1]), 0))

                                elif self.opt.hps_uncertainty:
                                    # Todo: Need further update
                                    conf_hp_detector = []
                                    for i in range(8):
                                        std_combined = np.sqrt(
                                            det['kps_fusion_std'][i * 2] + (det['kps_fusion_std'][i * 2 + 1]))
                                        # std_combined = 15 * np.sqrt(
                                        #     (det['kps_fusion_std'][i*2] / (det['bbox'][2] - det['bbox'][0]))**2
                                        #     + (det['kps_fusion_std'][i*2+1] / (det['bbox'][3] - det['bbox'][1]))**2)

                                        conf_hp_detector.append(np.maximum(1 - np.exp(np.log(0.15) / (
                                                self.opt.conf_border[self.opt.c][0] - self.opt.conf_border[self.opt.c][
                                            1])) ** (std_combined - self.opt.conf_border[self.opt.c][1]), 0))
                                else:
                                    conf_hp_detector = np.array(det['kps_heatmap_height'])

                                # Change visibility, following the protocal of COCO
                                pts = np.zeros((8, 3), dtype='int64')
                                for idx, p in enumerate(pts_ori):
                                    if p[0] >= ori_width or p[0] < 0 or p[1] < 0 or p[1] >= ori_height:
                                        pts[idx] = [p[0], p[1], 1]  # labeled but not visible
                                    else:
                                        pts[idx] = [p[0], p[1], 2]  # labeled and visible

                                for j in range(8):
                                    pts[j, :2] = affine_transform(pts[j, :2], trans_input)

                                    if pts[j, 2] > 1:  # Check visibility
                                        if pts[j, 0] >= 0 and pts[j, 0] < inp_width and \
                                                pts[j, 1] >= 0 and pts[j, 1] < inp_height:
                                            pt_int = pts[j, :2].astype(np.int32)

                                            if self.opt.render_hmhp_mode == 1 or self.opt.render_hmhp_mode == 3:
                                                draw_umich_gaussian(input_hm_hp[j], pt_int, hp_radius, k=1)
                                            elif self.opt.render_hmhp_mode == 0 or self.opt.render_hmhp_mode == 2:
                                                # Sometimes, heatmap is missing
                                                if radius_detector[j, 0] > 0:
                                                    # draw_nvidia_gaussian(input_hm_hp[j], pt_int,
                                                    #                      radius_detector[j, :],
                                                    #                      k=conf_hp_detector[j])
                                                    draw_umich_gaussian(input_hm_hp[j], pt_int,
                                                                        hp_radius,
                                                                        k=conf_hp_detector[j])
                else:
                    for det in dets:
                        if det['score'] < self.opt.pre_thresh:
                            continue

                        bbox = self._trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
                        bbox_out = self._trans_bbox(
                            det['bbox'], trans_output, out_width, out_height)

                        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                        if (h > 0 and w > 0):
                            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                            radius = max(0, int(radius))
                            ct = np.array(
                                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                            ct_int = ct.astype(np.int32)
                            if with_hm:
                                if self.opt.render_hm_mode == 0:
                                    draw_umich_gaussian(input_hm[0], ct_int, radius)
                                elif self.opt.render_hm_mode == 1:
                                    draw_umich_gaussian(input_hm[0], ct_int, radius, k=det['score'])
                            ct_out = np.array(
                                [(bbox_out[0] + bbox_out[2]) / 2,
                                 (bbox_out[1] + bbox_out[3]) / 2], dtype=np.int32)
                            output_inds.append(ct_out[1] * out_width + ct_out[0])

                            if with_hm_hp:
                                # Todo: Currently, hp_radius follows the same way as radius
                                hp_radius = radius
                                # hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                                # hp_radius = self.opt.hm_gauss \
                                #     if self.opt.mse_loss else max(0, int(hp_radius))

                                # Change visibility, following the protocal of COCO
                                pts_ori = np.array(det['kps']).reshape((-1, 2))
                                pts = np.zeros((8, 3), dtype='int64')
                                for idx, p in enumerate(pts_ori):
                                    if p[0] >= ori_width or p[0] < 0 or p[1] < 0 or p[1] >= ori_height:
                                        pts[idx] = [p[0], p[1], 1]  # labeled but not visible
                                    else:
                                        pts[idx] = [p[0], p[1], 2]  # labeled and visible

                                for j in range(8):
                                    pts[j, :2] = affine_transform(pts[j, :2], trans_input)

                                    if pts[j, 2] > 1:  # Check visibility
                                        if pts[j, 0] >= 0 and pts[j, 0] < inp_width and \
                                                pts[j, 1] >= 0 and pts[j, 1] < inp_height:
                                            pt_int = pts[j, :2].astype(np.int32)

                                            draw_umich_gaussian(input_hm_hp[j], pt_int, hp_radius, k=1)

        if with_hm:
            input_hm = input_hm[np.newaxis]  # add another dimension for batch size to be consistent
            input_hm = torch.from_numpy(input_hm).to(self.opt.device)

        if with_hm_hp:
            input_hm_hp = input_hm_hp[np.newaxis]  # add another dimension for batch size to be consistent
            input_hm_hp = torch.from_numpy(input_hm_hp).to(self.opt.device)

        # Not used yet
        output_inds = np.array(output_inds, np.int64).reshape(1, -1)
        output_inds = torch.from_numpy(output_inds).to(self.opt.device)
        return input_hm, input_hm_hp, output_inds

    def run(self, image_or_path_or_tensor, filename=None, meta_inp={}, preprocessed_flag=False):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, track_time, pnp_time, tot_time = 0, 0, 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()

        # pre_processed = False
        pre_processed = preprocessed_flag

        # File input
        if isinstance(image_or_path_or_tensor, np.ndarray):
            # For eval or for CenterPose as data generator
            image = image_or_path_or_tensor

            # We usually use image_or_path_or_tensor to represent filename
            if filename is not None:
                image_or_path_or_tensor = filename

        # String input
        elif type(image_or_path_or_tensor) == type(''):
            # For demo
            image = cv2.imread(image_or_path_or_tensor)
        else:
            # Not used yet
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        for scale in self.scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta_inp)
            else:

                # Used for data generation
                # 1 * 3 * 512 * 512
                images = np.expand_dims(image, axis=0)
                # images = image.reshape(1, 3, meta_inp['inp_height'], meta_inp['inp_width'])
                images = torch.from_numpy(images)
                meta = meta_inp

            images = images.to(self.opt.device)

            # initializing tracker
            pre_hms, pre_hm_hp, pre_inds = None, None, None

            if self.opt.refined_Kalman:
                self.tracker.init_track(
                    meta)

            if self.opt.tracking_task:
                # initialize the first frame
                if self.pre_images is None:
                    print('Initialize tracking!')
                    self.pre_images = images
                    self.tracker.init_track(
                        meta)

                # Initialize if given gt_pre_hm_hmhp
                elif self.opt.gt_pre_hm_hmhp or (self.opt.gt_pre_hm_hmhp_first and meta['id'] == 0):
                    self.tracker.init_track(
                        meta)

                if self.opt.pre_hm or self.opt.pre_hm_hp:
                    # render input heatmap from tracker status
                    # pre_inds is not used in the current version.
                    # We used pre_inds for learning an offset from previous image to
                    # the current image.
                    pre_hms, pre_hm_hp, pre_inds = self._get_additional_inputs(
                        self.tracker.tracks, meta, with_hm=self.opt.pre_hm, with_hm_hp=self.opt.pre_hm_hp)

            torch.cuda.synchronize()
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time

            # run the network
            # output: the output feature maps, only used for visualizing
            # dets: output tensors after extracting peaks
            output, dets, forward_time = self.process(
                images, self.pre_images, pre_hms, pre_hm_hp, pre_inds, return_time=True)

            torch.cuda.synchronize()
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time

            if self.opt.debug >= 2:
                # Mainly save keypoint heatmap & displacement for debug
                self.debug(debugger, images, copy.deepcopy(dets), output, scale, pre_hms, pre_hm_hp)

            # convert the cropped and 4x downsampled output coordinate system
            # back to the input image coordinate system
            dets = self.post_process(dets, meta, scale)
            torch.cuda.synchronize()
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(dets)

        # Mainly apply NMS
        results = self.merge_outputs(detections)
        torch.cuda.synchronize()
        merge_outputs_time = time.time()
        merge_time += merge_outputs_time - post_process_time

        # Gaussian fusion on kps
        if self.opt.tracking_task or self.opt.refined_Kalman:
            def gaussian_fusion(det):

                kps_fusion_mean = []
                kps_fusion_std = []

                for i in range(16):

                    if self.opt.hps_uncertainty == True:
                        # Todo: apply some filters here according to kps_displacement_std and kps_heatmap_height
                        if det['kps_heatmap_mean'][i] < 0 or det['kps_heatmap_std'][i] < 0:
                            std = det['kps_displacement_std'][i]
                            mean = det['kps_displacement_mean'][i]

                            # if std>5:
                            #     std = 100

                        else:

                            std = (det['kps_displacement_std'][i] ** -2 + det['kps_heatmap_std'][i] ** -2) ** -0.5
                            mean = std ** 2 * (det['kps_displacement_std'][i] ** -2 * det['kps_displacement_mean'][i] +
                                               det['kps_heatmap_std'][i] ** -2 * det['kps_heatmap_mean'][i])

                            # if det['kps_displacement_std'][i] > 5 or det['kps_heatmap_std'][i] > 8:
                            #     std = 100
                    else:
                        if det['kps_heatmap_mean'][i] < 0 or det['kps_heatmap_std'][i] < 0:
                            std = 20
                            mean = det['kps_displacement_mean'][i]
                        else:
                            std = det['kps_heatmap_std'][i] / np.sqrt(2)
                            mean = std ** 2 * (det['kps_heatmap_std'][i] ** -2 * det['kps_displacement_mean'][i] +
                                               det['kps_heatmap_std'][i] ** -2 * det['kps_heatmap_mean'][i])

                    kps_fusion_mean.append(mean)
                    kps_fusion_std.append(std)

                return kps_fusion_mean, kps_fusion_std

            for det in results:
                # Note that sometimes kps may conclude missing points
                kps_fusion_mean, kps_fusion_std = gaussian_fusion(det)
                det['kps_fusion_mean'] = np.array(kps_fusion_mean)
                det['kps_fusion_std'] = np.array(kps_fusion_std)

        # The goal is to get 2d projection of keypoints & 6-DoF & 3d keypoint in camera frame
        boxes = []
        if self.opt.use_pnp == True:

            for bbox in results:
                # Point processing according to different rep_modes
                if self.opt.rep_mode == 0 or self.opt.rep_mode == 3 or self.opt.rep_mode == 4:

                    # 8 representation from centernet
                    points = [(x[0], x[1]) for x in np.array(bbox['kps']).reshape(-1, 2)]
                    points_filtered = points

                elif self.opt.rep_mode == 1:

                    # 16 representation
                    points_1 = np.array(bbox['kps_displacement_mean']).reshape(-1, 2)
                    points_1 = [(x[0], x[1]) for x in points_1]
                    points_2 = np.array(bbox['kps_heatmap_mean']).reshape(-1, 2)
                    points_2 = [(x[0], x[1]) for x in points_2]
                    points = np.hstack((points_1, points_2)).reshape(-1, 2)
                    points_filtered = points

                elif self.opt.rep_mode == 2:

                    points = []

                    N_sample = 20

                    confidence_list = []
                    dis_list = []
                    weight_list = []

                    keypoint_heatmap_mean_list = []
                    keypoint_heatmap_std_list = []

                    keypoint_displacement_mean_list = []
                    keypoint_displacement_std_list = []

                    GMM_list = []

                    for i in range(8):

                        # Normalized L2
                        keypoint_displacement_norm = np.array(
                            [bbox['kps_displacement_mean'][i * 2] / meta['width'],
                             bbox['kps_displacement_mean'][i * 2 + 1] / meta['height']])
                        keypoint_heatmap_norm = np.array(
                            [bbox['kps_heatmap_mean'][i * 2] / meta['width'],
                             bbox['kps_heatmap_mean'][i * 2 + 1] / meta['height']])
                        dis = np.linalg.norm(keypoint_displacement_norm - keypoint_heatmap_norm)

                        confidence_list.append(bbox['kps_heatmap_height'][i])
                        dis_list.append(dis)

                        def gaussian(dist, sigma=10.):
                            return math.e ** (-dist ** 2 / 2 / sigma ** 2)

                        # Calculate new weight list according to confidence & gaussian distribution on dis
                        weight_list.append(confidence_list[i] * gaussian(dis))

                        # 1. Heatmap
                        keypoint_heatmap_mean = [bbox['kps_heatmap_mean'][i * 2], bbox['kps_heatmap_mean'][i * 2 + 1]]
                        keypoint_heatmap_std = [bbox['kps_heatmap_std'][i * 2], bbox['kps_heatmap_std'][i * 2 + 1]]

                        # 2. Displacement
                        kps_displacement_mean = [bbox['kps_displacement_mean'][i * 2],
                                                 bbox['kps_displacement_mean'][i * 2 + 1]]
                        kps_displacement_std = keypoint_heatmap_std

                        # Fit a GMM by sampling from keypoint_displacement &  keypoint_heatmap distributions
                        X_train = []
                        if keypoint_heatmap_mean[0] < -5000 or keypoint_heatmap_mean[1] < -5000:
                            kps_displacement_std = [5, 5]
                            points_sample = np.random.multivariate_normal(
                                np.array(kps_displacement_mean),
                                np.array([[kps_displacement_std[0], 0], [0, kps_displacement_std[1]]]), size=1000)
                            X_train.append(points_sample)
                        else:
                            points_sample = np.random.multivariate_normal(
                                np.array(keypoint_heatmap_mean),
                                np.array([[keypoint_heatmap_mean[0], 0], [0, keypoint_heatmap_mean[1]]]), size=500)
                            X_train.append(points_sample)

                            points_sample = np.random.multivariate_normal(
                                np.array(kps_displacement_mean),
                                np.array([[kps_displacement_std[0], 0], [0, kps_displacement_std[1]]]), size=500)
                            X_train.append(points_sample)

                        keypoint_heatmap_mean_list.append(keypoint_heatmap_mean)
                        keypoint_heatmap_std_list.append(keypoint_heatmap_std)
                        keypoint_displacement_mean_list.append(kps_displacement_mean)
                        keypoint_displacement_std_list.append(kps_displacement_std)

                        X_train = np.array(X_train).reshape(-1, 2)
                        clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
                        clf.fit(X_train)
                        GMM_list.append(clf)

                        points_sample = clf.sample(N_sample)
                        points_sample = np.hstack((points_sample[0], np.array(points_sample[1]).reshape(-1, 1)))
                        points.append(points_sample)

                    points = np.array(points).reshape(-1, 3)
                    # Do not need labels for pnp
                    points_filtered = points[:, 0:2]

                ret = pnp_shell(self.opt, meta, bbox, points_filtered, bbox['obj_scale'], OPENCV_RETURN=self.opt.show_axes)
                if ret is not None:
                    boxes.append(ret)

        pnp_process_time = time.time()
        pnp_time += pnp_process_time - merge_outputs_time

        # Tracker update
        if self.opt.tracking_task:
            results, boxes = self.tracker.step(results, boxes)
            self.pre_images = images
        # For baseline (CenterPose + Kalman)
        elif self.opt.refined_Kalman:
            results, boxes = self.tracker.step(results, boxes)

        end_time = time.time()
        track_time += end_time - pnp_process_time
        tot_time += end_time - start_time

        # Dict is for output debug
        dict_out = {"camera_data": [], "objects": []}
        if 'camera_matrix' in meta:
            camera_matrix = meta['camera_matrix']
            dict_out['camera_data'] = camera_matrix.tolist()

        if self.opt.tracking_task or self.opt.refined_Kalman:
            for track in self.tracker.tracks:
                # Basic part
                dict_obj = {
                    'class': self.opt.c,
                    'ct': track['ct'],
                    'bbox': np.array(track['bbox']).tolist(),
                    'confidence': track['score'],
                    'kps_displacement_mean': track['kps_displacement_mean'].tolist(),
                    'kps_heatmap_mean': track['kps_heatmap_mean'].tolist(),

                    'kps_heatmap_std': track['kps_heatmap_std'].tolist(),
                    'kps_heatmap_height': track['kps_heatmap_height'].tolist(),
                    'obj_scale': (track['obj_scale']/track['obj_scale'][1]).tolist(),

                    'tracking_id':track['tracking_id'],
                }

                # Optional part
                if self.opt.use_pnp:
                    if 'location' in track:
                        dict_obj['location'] = track['location']
                        dict_obj['quaternion_xyzw'] = track['quaternion_xyzw'].tolist()
                    if 'kps_pnp' in track:
                        dict_obj['kps_pnp'] = track['kps_pnp'].tolist()
                        dict_obj['kps_3d_cam'] = track['kps_3d_cam'].tolist()

                if self.opt.obj_scale_uncertainty:
                    dict_obj['obj_scale_uncertainty'] = track['obj_scale_uncertainty'].tolist()

                if self.opt.kalman:
                    dict_obj['kps_mean_kf'] = track['kps_mean_kf'].tolist()
                    dict_obj['kps_std_kf'] = track['kps_std_kf']
                    if self.opt.use_pnp and 'kps_pnp_kf' in track:
                        dict_obj['kps_pnp_kf'] = track['kps_pnp_kf'].tolist()
                        dict_obj['kps_3d_cam_kf'] = track['kps_3d_cam_kf'].tolist()

                if self.opt.scale_pool == True:
                    dict_obj['obj_scale_kf'] = (track['obj_scale_kf']/track['obj_scale_kf'][1]).tolist()
                    dict_obj['obj_scale_uncertainty_kf'] = track['obj_scale_uncertainty_kf'].tolist()

                if self.opt.hps_uncertainty:
                    dict_obj['kps_displacement_std'] = track['kps_displacement_std'].tolist()
                    dict_obj['kps_fusion_mean'] = track['kps_fusion_mean'].tolist()
                    dict_obj['kps_fusion_std'] = track['kps_fusion_std'].tolist()

                if self.opt.tracking:
                    dict_obj['tracking'] = track['tracking'].tolist()
                if self.opt.tracking_hp:
                    dict_obj['tracking_hp'] = track['tracking_hp'].tolist()

                dict_out['objects'].append(dict_obj)
        else:
            for box in boxes:
                # Basic part
                dict_obj = {
                    'class': self.opt.c,
                    'ct': box[4]['ct'],
                    'bbox': np.array(box[4]['bbox']).tolist(),
                    'confidence': box[4]['score'],
                    'kps_displacement_mean': box[4]['kps_displacement_mean'].tolist(),
                    'kps_heatmap_mean': box[4]['kps_heatmap_mean'].tolist(),

                    'kps_heatmap_std': box[4]['kps_heatmap_std'].tolist(),
                    'kps_heatmap_height': box[4]['kps_heatmap_height'].tolist(),
                    'obj_scale': box[4]['obj_scale'].tolist(),
                }

                # Optional part
                if self.opt.use_pnp:
                    if 'location' in box[4]:
                        dict_obj['location'] = box[4]['location']
                        dict_obj['quaternion_xyzw'] = box[4]['quaternion_xyzw'].tolist()
                    if 'kps_pnp' in box[4]:
                        dict_obj['kps_pnp'] = box[4]['kps_pnp'].tolist()
                        dict_obj['kps_3d_cam'] = box[4]['kps_3d_cam'].tolist()

                dict_out['objects'].append(dict_obj)


        if self.opt.debug >= 1 and self.opt.debug < 4:
            self.show_results(debugger, image, results)

        # Saving path is specific for demo folder structure
        elif self.opt.debug == 4:
            self.save_results(debugger, image, results, image_or_path_or_tensor, dict_out)

        # Saving path is specific for evaluation
        elif self.opt.debug == 6:
            self.save_results_eval(debugger, image, results, image_or_path_or_tensor, dict_out)

        # Save results for debug, boxes for evaluation, output is the original network output
        # Todo: Actually results could be combined with boxes
        return {'results': results, 'boxes': boxes, 'output': output, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time, 'pnp': pnp_time, 'track': track_time}

    def reset_tracking(self):
        self.tracker.reset()
        self.pre_images = None
