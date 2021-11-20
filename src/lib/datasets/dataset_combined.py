# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import json
import cv2
import torch
import os
from lib.utils.image import flip, color_aug
from lib.utils.image import get_affine_transform, affine_transform
from lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, draw_nvidia_gaussian
from lib.utils.image import draw_dense_reg
import math
import copy
import scipy.stats as stats

import albumentations as A

from scipy.spatial.transform import Rotation as R

from os.path import exists
import glob
from lib.opts import opts
from lib.detectors.detector_factory import detector_factory


def rotation_y_matrix(theta):
    M_R = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                    [0, 1, 0, 0],
                    [-np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]])
    return M_R


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]


def bounding_box_rotation(points, trans):
    coordinates_transformed = []
    for x, y, _ in points:
        coordinates_transformed.append(affine_transform([x, y], trans))

    return bounding_box(coordinates_transformed)


class ObjectPoseDataset(data.Dataset):
    num_classes = 1
    num_joints = 8
    default_resolution = [512, 512]

    # Todo: not important for now
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    # Swap the index horizontally
    flip_idx = [[1, 5], [3, 7], [2, 6], [4, 8]]

    # obj_scale = True # Determined by opt rather than dataset

    # Not used yet
    # mu: x,y,z,x_normalized,z_normalized
    # std: x,y,z,x_normalized,z_normalized
    dimension_ref = {
        'bike': [[0.65320896, 1.021797894, 1.519635599, 0.6520559199, 1.506392621],
                 [0.1179380561, 0.176747817, 0.2981715678, 0.1667947895, 0.3830536275]],
        'book': [[0.225618019, 0.03949624326, 0.1625821624, 7.021850281, 5.064694187],
                 [0.1687487664, 0.07391230822, 0.06436673199, 3.59629568, 2.723290812]],
        'bottle': [
            [0.07889784977450116, 0.24127451915330908, 0.0723714257114412, 0.33644069262302545, 0.3091134992864717, ],
            [0.02984649578071775, 0.06381390122918497, 0.03088144838560917, 0.11052240441921059,
             0.13327627592012867, ]],
        'camera': [
            [0.11989848700326843, 0.08226238775595619, 0.09871718158089632, 1.507216484439368, 1.1569407159290284, ],
            [0.021177290310316968, 0.02158788017191602, 0.055673710278419844, 0.28789183678046854,
             0.5342094080365904, ]],
        'cereal_box': [
            [0.19202754401417296, 0.2593114001714919, 0.07723794925413519, 0.7542602699204104, 0.29441151268928173, ],
            [0.08481640897407464, 0.09999915952084068, 0.09495429981036707, 0.19829004029411457, 0.2744797990483879, ]],
        'chair': [
            [0.5740664085137888, 0.8434027515832329, 0.6051523831888338, 0.6949691013776601, 0.7326891354260606, ],
            [0.12853104253707456, 0.14852086453095492, 0.13428881418587957, 0.16897092539619352,
             0.18636134566748525, ]],
        'cup': [
            [0.08587637391801063, 0.12025228955138188, 0.08486836104868696, 0.7812126934904675, 0.7697895244331658, ],
            [0.05886805978497525, 0.06794896438246326, 0.05875681990718713, 0.2887038681446475, 0.283821205157399, ]],
        'mug': [
            [0.14799136566553112, 0.09729087667918128, 0.08845449667169905, 1.3875694883045138, 1.0224997119392225, ],
            [1.0488828523223728, 0.2552672927963539, 0.039095350310480705, 0.3947832854104711, 0.31089415283872546, ]],
        'laptop': [
            [0.33685059747485196, 0.1528068814247063, 0.2781020624738614, 35.920214652427696, 23.941173992376903, ],
            [0.03529983948867832, 0.07017080198389423, 0.0665823136876069, 391.915687801732, 254.21325950495455, ]],
        'shoe': [
            [0.10308848289662519, 0.10932616184503478, 0.2611737789760352, 1.0301976264129833, 2.6157393112424328, ],
            [0.02274768925924402, 0.044958380226590516, 0.04589720205423542, 0.3271000267177176, 0.8460337534776092, ]],
    }

    def __init__(self, opt, split):
        super(ObjectPoseDataset, self).__init__()
        self.edges = [[2, 4], [2, 6], [6, 8], [4, 8],
                      [1, 2], [3, 4], [5, 6], [7, 8],
                      [1, 3], [1, 5], [3, 7], [5, 7]]

        # Todo: need to fix the path name
        if opt.tracking_task == True:
            self.data_dir = os.path.join(opt.data_dir, 'outf_all')
        else:
            self.data_dir = os.path.join(opt.data_dir, 'outf')

        # # Debug only
        # self.data_dir = os.path.join(opt.data_dir, 'outf_all_test')

        self.img_dir = os.path.join(self.data_dir, f"{opt.c}_{split}")

        # Todo: take the test split as validation
        if split == 'val' and not os.path.isdir(self.img_dir):
            self.img_dir = os.path.join(self.data_dir, f"{opt.c}_test")

        self.max_objs = 10
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt

        # Init detector for data generation
        if self.opt.data_generation_mode_ratio > 0:
            print(f'==> initializing trained detector for data generation.')

            opt = opts().parser.parse_args([])
            opt_detector = opts().parse(opt)

            opt_detector.c = self.opt.c
            opt_detector.debug = 0
            opt_detector.pre_img = False
            opt_detector.arch = 'dlav1_34'
            opt_detector.use_pnp = True
            opt_detector.rep_mode = 1
            opt_detector.obj_scale = True
            opt_detector = opts().init(opt_detector)
            # Init the detector
            Detector_CenterPose = detector_factory[opt_detector.task]

            if opt_detector.c != 'cup':
                # Todo: path to be updated
                # Path is related to the entry script
                if opt_detector.c != 'bottle':
                    opt_detector.load_model = f"../models/CenterPose/{opt_detector.c}_v1_140.pth"
                else:
                    opt_detector.load_model = f"../models/CenterPose/bottle_v1_sym_12_140.pth"
                self.detector = Detector_CenterPose(opt_detector)
            else:
                # Two detectors for the cup category

                opt_detector.load_model = f"../models/CenterPose/cup_mug_v1_140.pth"
                self.detector_mug = Detector_CenterPose(opt_detector)
                opt_detector.load_model = f"../models/CenterPose/cup_cup_v1_sym_12_140.pth"
                self.detector_cup = Detector_CenterPose(opt_detector)

        print(f'==> initializing objectron {opt.c}_{split} data.')

        #   Copy from DOPE code
        def loadimages(root, datastyle="json", extensions=['png']):
            imgs = []
            loadimages.extensions = extensions

            def add_json_files(path, ):
                for ext in loadimages.extensions:
                    for imgpath in glob.glob(path + "/*.{}".format(ext.replace('.', ''))):
                        if exists(imgpath) and exists(imgpath.replace(ext, "json")):
                            # Save img_path, video_id, frame_id, json_path
                            video_id = os.path.split(os.path.split(imgpath)[0])[1]
                            frame_id = os.path.splitext(os.path.basename(imgpath))[0]

                            imgs.append((imgpath, video_id, frame_id,
                                         imgpath.replace(ext, "json")))

            def explore(path):
                if not os.path.isdir(path):
                    return
                folders = [os.path.join(path, o) for o in os.listdir(path)
                           if os.path.isdir(os.path.join(path, o))]
                if len(folders) > 0:
                    for path_entry in folders:
                        explore(path_entry)
                else:
                    add_json_files(path)

            explore(root)

            return imgs

        def load_data(path, extensions):
            imgs = loadimages(path, extensions=extensions)
            return imgs

        self.images = []
        print(self.img_dir)
        self.images += load_data(self.img_dir, extensions=["png", 'jpg'])
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

        # Group all the info by video_id
        # self.images save all the img_path, video_id, frame_id, json_path
        # Organize all the images (a dict of info) corresponding to the same video_id
        print('Creating video index!')
        self.videos = {}
        for i in self.images:
            new_item = i[1]  # according to the video_id
            if new_item in self.videos:
                self.videos[new_item].append(i)
            else:
                self.videos[new_item] = []
                self.videos[new_item].append(i)

    def __len__(self):
        return self.num_samples

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    # Add noise
    def _get_aug_param(self, c_ori, s, width, height, disturb=False):
        c = c_ori.copy()
        if (not self.opt.not_rand_crop) and not disturb:
            # Training for current frame
            aug_s = np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, width)
            h_border = self._get_border(128, height)
            c[0] = np.random.randint(low=w_border, high=width - w_border)
            c[1] = np.random.randint(low=h_border, high=height - h_border)
        else:
            # Training for previous frame
            sf = self.opt.scale
            cf = self.opt.shift

            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            aug_s = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.opt.aug_rot:
            rf = self.opt.rotate  # 0 - 180
            rot = 2 * (np.random.rand() - 0.5) * rf
            # rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        else:
            rot = 0

        return c, aug_s, rot

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

    def _get_input(self, img, trans_input):
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def __getitem__(self, index):
        # # Debug only
        # np.random.seed(100)

        # <editor-fold desc="Data initialization">

        path_img, video_id, frame_id, path_json = self.images[index]
        img_path = path_img
        with open(path_json) as f:
            anns = json.load(f)
        num_objs = min(len(anns['objects']), self.max_objs)

        try:
            img = cv2.imread(img_path)
        except:
            return None

        if self.opt.new_data_augmentation:
            # Only apply albumentations on spatial data augmentation, nothing to do with gt label
            transform = A.Compose([
                A.MotionBlur(blur_limit=3, p=0.1),
                A.Downscale(scale_min=0.6, scale_max=0.8, p=0.1),
                A.GaussNoise(p=0.2),
                # A.Blur(p=0.2),
                # A.RandomBrightnessContrast(p=0.2),
            ],
            )
            transformed = transform(image=img)
            # Update image
            img = transformed["image"]

        try:
            height, width = img.shape[0], img.shape[1]
        except:
            return None
        c_ori = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s_ori = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0

        flipped = False
        if self.split == 'train':

            c, aug_s, rot = self._get_aug_param(c_ori, s_ori, width, height, disturb=False)
            s = s_ori * aug_s

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]

                c[0] = width - c[0] - 1
        else:
            c = c_ori
            s = s_ori

        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_res, self.opt.input_res])

        try:
            inp = self._get_input(img, trans_input)
        except:
            return None

        output_res = self.opt.output_res
        num_joints = self.num_joints
        trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])

        # Parameter initialization
        # Set the rotational symmetry, will be varied according to the category
        if self.opt.c == 'chair':
            theta = 2 * np.pi / 4
            num_symmetry = 4
        elif (self.opt.c == 'cup' and self.opt.mug == False) or self.opt.c == 'bottle':
            num_symmetry = self.opt.num_symmetry
            theta = 2 * np.pi / num_symmetry
        else:
            # No symmetry
            num_symmetry = 1

        # All the gt info:
        hm = np.zeros((num_symmetry, self.num_classes, output_res, output_res), dtype=np.float32)
        hm_hp = np.zeros((num_symmetry, num_joints, output_res, output_res), dtype=np.float32)
        dense_kps = np.zeros((num_symmetry, num_joints, 2, output_res, output_res),
                             dtype=np.float32)
        dense_kps_mask = np.zeros((num_symmetry, num_joints, output_res, output_res),
                                  dtype=np.float32)
        wh = np.zeros((num_symmetry, self.max_objs, 2), dtype=np.float32)
        scale = np.zeros((num_symmetry, self.max_objs, 3), dtype=np.float32)
        scale_uncertainty = np.zeros((num_symmetry, self.max_objs, 3), dtype=np.float32)

        kps = np.zeros((num_symmetry, self.max_objs, num_joints * 2), dtype=np.float32)
        kps_displacement_std = np.zeros((num_symmetry, self.max_objs, num_joints * 2), dtype=np.float32)

        reg = np.zeros((num_symmetry, self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((num_symmetry, self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((num_symmetry, self.max_objs), dtype=np.uint8)
        kps_mask = np.zeros((num_symmetry, self.max_objs, num_joints * 2), dtype=np.uint8)
        hp_offset = np.zeros((num_symmetry, self.max_objs * num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((num_symmetry, self.max_objs * num_joints), dtype=np.int64)
        hp_mask = np.zeros((num_symmetry, self.max_objs * num_joints), dtype=np.int64)

        tracking = np.zeros((num_symmetry, self.max_objs, 2), dtype=np.float32)
        tracking_mask = np.zeros((num_symmetry, self.max_objs), dtype=np.uint8)
        tracking_hp = np.zeros((num_symmetry, self.max_objs, num_joints * 2), dtype=np.float32)
        tracking_hp_mask = np.zeros((num_symmetry, self.max_objs, num_joints * 2), dtype=np.uint8)

        draw_gaussian = draw_umich_gaussian
        # Todo: Need updates if adding more modalities, however, it is not easy to be changed to dict
        gt_det_pad = np.zeros((num_symmetry, self.max_objs, 60), dtype=np.float32)

        # </editor-fold>

        # <editor-fold desc="Step1: Work on the previous frame">
        if self.opt.tracking_task == True:

            cts_pre_list = []
            track_ids = []
            pts_pre_list = []
            pts_mask_pre_list = []

            # Group info
            video_group = self.videos[video_id]
            # Get info
            # If training, random sample nearby frames as the "previous" frame
            # If testing, get the exact previous frame
            # (img_path, video_id, frame_id, json_path) in video_group
            if 'train' in self.split:
                img_ids = [image for image in video_group if
                           abs(int(image[2]) - int(frame_id)) < self.opt.max_frame_dist]
            else:
                img_ids = [image for image in video_group if int(image[2]) - int(frame_id) == -1]

                if len(img_ids) == 0:
                    # If the previous one is not available, use the exact same image instead
                    img_ids = [image for image in video_group if abs(int(image[2]) - int(frame_id)) == 0]
            rand_id = np.random.choice(len(img_ids))
            path_img_pre, video_id_pre, frame_id_pre, path_json_pre = img_ids[rand_id]

            frame_dist = abs(int(frame_id) - int(frame_id_pre))

            image_pre = cv2.imread(path_img_pre)
            with open(path_json_pre) as f:
                anns_pre = json.load(f)

            # Flip
            if flipped:
                image_pre = image_pre[:, ::-1, :].copy()
                # c[0] = width - c[0] - 1 # Have been updated

            if self.opt.same_aug_pre and frame_dist != 0:
                trans_input_pre = trans_input

                # Not used yet
                trans_output_rot_pre = trans_output_rot
            else:
                # Keep the same rotation as the new one
                c_pre, aug_s_pre, _ = self._get_aug_param(
                    c_ori, s_ori, width, height, disturb=True)
                s_pre = s_ori * aug_s_pre
                trans_input_pre = get_affine_transform(
                    c_pre, s_pre, rot, [self.opt.input_w, self.opt.input_h])

                # Not used actually
                trans_output_rot_pre = get_affine_transform(c_pre, s_pre, rot, [output_res, output_res])

            # Get img_pre
            try:
                img_pre = self._get_input(image_pre, trans_input_pre)
            except:
                return None

            # No symmetry ambiguity in the previous frame
            hm_pre = np.zeros((self.num_classes, self.opt.input_h, self.opt.input_w), dtype=np.float32)
            hm_hp_pre = np.zeros((num_joints, self.opt.input_h, self.opt.input_w), dtype=np.float32)

            # 0: Noise simulation; 1: CenterPose as data generator
            data_generation_mode = 1 if np.random.random() < self.opt.data_generation_mode_ratio else 0

            # Detector data generation mode
            if data_generation_mode == 1:
                intrinsic = np.identity(3)
                intrinsic[0, 0] = anns_pre['camera_data']['intrinsics']['fx']
                intrinsic[0, 2] = anns_pre['camera_data']['intrinsics']['cx']
                intrinsic[1, 1] = anns_pre['camera_data']['intrinsics']['fy']
                intrinsic[1, 2] = anns_pre['camera_data']['intrinsics']['cy']

                # trans_input_pre: from raw input to processed input (e.g., 512);
                # trans_output_rot_pre: from raw input to unscaled output (e.g., 128)

                if self.opt.same_aug_pre and frame_dist != 0:
                    meta_inp = {'c': c, 's': s, 'height': height, 'width': width,
                                'out_height': output_res, 'out_width': output_res,
                                'inp_height': self.opt.input_h, 'inp_width': self.opt.input_w,
                                'trans_input': trans_input_pre, 'trans_output': trans_output_rot_pre,
                                'camera_matrix': intrinsic}
                else:
                    meta_inp = {'c': c_pre, 's': s_pre, 'height': height, 'width': width,
                                'out_height': output_res, 'out_width': output_res,
                                'inp_height': self.opt.input_h, 'inp_width': self.opt.input_w,
                                'trans_input': trans_input_pre, 'trans_output': trans_output_rot_pre,
                                'camera_matrix': intrinsic}

                # preprocessed_image : 3*512*512
                if self.opt.c != 'cup':
                    ret = self.detector.run(img_pre, filename=path_img_pre,
                                            meta_inp=meta_inp, preprocessed_flag=True)
                else:
                    if self.opt.mug == True:
                        ret = self.detector_mug.run(img_pre, filename=path_img_pre,
                                                    meta_inp=meta_inp, preprocessed_flag=True)
                    else:
                        ret = self.detector_cup.run(img_pre, filename=path_img_pre,
                                                    meta_inp=meta_inp, preprocessed_flag=True)
                # boxes: keypoint_2d_pnp, keypoint_3d, predicted_scale, keypoint_2d_ori, result_ori for debug

                # Extract all the gt label for 2d keypoints (normalized)
                instances_2d = []
                for ann_pre in anns_pre['objects']:

                    projected_cuboid = np.array(ann_pre['projected_cuboid']).reshape((9, 2)).astype('float32')

                    if flipped:
                        projected_cuboid[:, 0] = width - projected_cuboid[:, 0] - 1
                        for e in self.opt.flip_idx:
                            temp_1 = e[1] - 1
                            temp_0 = e[0] - 1
                            projected_cuboid[temp_0], projected_cuboid[temp_1] = projected_cuboid[temp_1].copy(), \
                                                                                 projected_cuboid[temp_0].copy()
                            # pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

                    projected_cuboid[:, 0] = projected_cuboid[:, 0] / width
                    projected_cuboid[:, 1] = projected_cuboid[:, 1] / height

                    def _is_visible(point):
                        """Determines if a 2D point is visible."""
                        return point[0] > 0 and point[0] < 1 and point[1] > 0 and point[1] < 1

                    if _is_visible(projected_cuboid[0]):
                        instances_2d.append(projected_cuboid)
                    # Todo: may have a problem
                    else:
                        # Cannot skip any instances because we need their order
                        instances_2d.append(np.ones((9, 2)) * -10000)
                instances_2d = np.array(instances_2d).reshape((-1, 9, 2))

                # Todo: Currently it does not support symmetrical objects
                # Match each detection with one GT
                # Mainly for tracking label
                match_detector = []
                norms_list = []  # If one GT is matched with several prediction, choose the nearest
                for box in ret['boxes']:
                    box_point_2d, box_point_3d, relative_scale, box_point_2d_ori, result_ori = box

                    norms = np.linalg.norm(instances_2d[:, 1:, :] - box_point_2d[1:, :], axis=(1, 2))
                    i_min = np.argmin(norms)

                    # the order number in the list (box) corresponds to the order number in the annotation (gt)
                    match_detector.append(i_min)
                    norms_list.append(norms)

                match_detector = np.array(match_detector)
                norms_list = np.array(norms_list)

            # Todo: For convenience, always run noise simulation to get some info but may not use it if using CenterPose

            # Calculate some info
            cam_projection_matrix = anns_pre['camera_data']['camera_projection_matrix']

            id_symmetry_pre_list = []
            for idx_obj, ann_pre in enumerate(anns_pre['objects']):

                # Todo: Only for chair category for now
                if 'symmetric' in ann_pre:
                    if ann_pre['symmetric'] == 'True':
                        num_symmetry = 4
                    else:
                        num_symmetry = 1

                if self.opt.c == 'cup':
                    if (self.opt.mug == False and ann_pre['mug'] == True) or \
                            self.opt.mug == True and ann_pre['mug'] == False:
                        id_symmetry_pre_list.append(None)
                        continue

                # Todo: Fixed as 0 for now
                cls_id = 0
                pts_ori_pre = np.array(ann_pre['projected_cuboid'])

                # Randomly choose the symmetry id
                id_symmetry_pre = np.random.choice(num_symmetry)

                # Update later if successfully render a heat for the center
                id_symmetry_pre_list.append(None)
                # Update info if there exists symmetry
                if num_symmetry != 1:
                    object_rotations = ann_pre['quaternion_xyzw']
                    object_translations = ann_pre['location']
                    keypoints_3d = np.array(ann_pre['keypoints_3d'])

                    M_o2c = np.identity(4)
                    M_o2c[:3, :3] = R.from_quat(object_rotations).as_matrix()
                    M_o2c[:3, 3] = object_translations

                    M_c2o = np.linalg.inv(M_o2c)

                    M_R = rotation_y_matrix(theta * id_symmetry_pre)

                    # Project the rotated 3D keypoint to the image plane
                    M_trans = cam_projection_matrix @ M_o2c @ M_R @ M_c2o

                    new_keypoints_2d = []
                    for i in range(9):
                        projected_point_ori = M_trans @ np.vstack((keypoints_3d[i].reshape(3, -1), 1))
                        projected_point_ori = (projected_point_ori / projected_point_ori[3])[:3]
                        viewport_point = (projected_point_ori + 1.0) / 2.0 * np.array([height, width, 1.0]).reshape(
                            3, 1)
                        new_keypoints_2d.append([int(viewport_point[1]), int(viewport_point[0])])

                    pts_ori_pre = new_keypoints_2d

                ct_ori_pre = pts_ori_pre[0]  # center
                pts_ori_pre = pts_ori_pre[1:]  # 8 corners

                # Change visibility, following the protocol of COCO
                pts_pre = np.zeros((len(pts_ori_pre), 3), dtype='int64')
                for idx, p in enumerate(pts_ori_pre):
                    if p[0] >= width or p[0] < 0 or p[1] < 0 or p[1] >= height:
                        pts_pre[idx] = [p[0], p[1], 1]  # labeled but not visible
                    else:
                        pts_pre[idx] = [p[0], p[1], 2]  # labeled and visible

                # Horizontal flip
                if flipped:
                    pts_pre[:, 0] = width - pts_pre[:, 0] - 1
                    for e in self.opt.flip_idx:
                        temp_1 = e[1] - 1
                        temp_0 = e[0] - 1
                        pts_pre[temp_0], pts_pre[temp_1] = pts_pre[temp_1].copy(), pts_pre[temp_0].copy()

                bbox = np.array(bounding_box_rotation(pts_pre, trans_input_pre))

                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.input_w - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.input_h - 1)

                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

                # Filter out GT if most of the keypoints are not visible (more than 4)
                visible_flag = True
                if ct_ori_pre[0] >= width or ct_ori_pre[0] < 0 or ct_ori_pre[1] < 0 or ct_ori_pre[1] >= height:
                    if pts_pre[:, 2].sum() <= 12:  # 12 = 2*6
                        visible_flag = False

                if ((h > 0 and w > 0) or (rot != 0)) and visible_flag:

                    # Todo: Find the first matching result for each GT
                    if data_generation_mode == 1:
                        match_detector_idx = None
                        if idx_obj in match_detector:
                            # E.g., [2,1,0] means 0(pred) <-> 2(gt)
                            # match_detector_idx = match_detector.index(idx_obj)

                            # Still a list
                            match_detector_idx = np.where(match_detector == idx_obj)[0]

                            if len(match_detector_idx) == 1:
                                # Convert to int
                                match_detector_idx = match_detector_idx[0]
                            else:
                                # Choose the nearest pred to GT
                                match_detector_idx = match_detector_idx[
                                    np.argmin(norms_list[match_detector_idx, idx_obj])]
                                if np.min(norms_list[match_detector_idx, idx_obj]) > 1000:
                                    match_detector_idx = None

                            if match_detector_idx is not None:
                                # Todo: Currently it does not support symmetrical objects
                                ct_detector = ret['boxes'][match_detector_idx][4]['ct']
                                ct_detector = affine_transform(ct_detector, trans_input_pre)

                                # Only keep 8 keypoints
                                if self.opt.render_hmhp_mode == 0 or self.opt.render_hmhp_mode == 1:
                                    # distance kps
                                    pts_detector = ret['boxes'][match_detector_idx][3][1:]
                                elif self.opt.render_hmhp_mode == 2 or self.opt.render_hmhp_mode == 3:
                                    # pnp
                                    pts_detector = ret['boxes'][match_detector_idx][0][1:]

                                # Both need re-scale
                                pts_detector[:, 0] = pts_detector[:, 0] * width
                                pts_detector[:, 1] = pts_detector[:, 1] * height

                                # Not used yet
                                if self.opt.same_aug_pre and frame_dist != 0:
                                    radius_detector = (np.array(
                                        ret['boxes'][match_detector_idx][4]['kps_heatmap_std']).reshape(-1,
                                                                                                        2) * aug_s).astype(
                                        np.int32)
                                else:
                                    radius_detector = (np.array(
                                        ret['boxes'][match_detector_idx][4]['kps_heatmap_std']).reshape(-1,
                                                                                                        2) * aug_s_pre).astype(
                                        np.int32)

                                # For CenterPose, we directly use its heatmap height
                                conf_hp_detector = np.array(ret['boxes'][match_detector_idx][4]['kps_heatmap_height'])

                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))

                    if self.opt.center_3D == False:
                        # Todo: Need modification, bbox is not accurate enough as we do not have gt info
                        ct = np.array(
                            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    else:
                        # Todo: Right row, do not consider objects whose center is out of the image
                        if flipped:
                            ct_ori_pre[0] = width - ct_ori_pre[0] - 1
                        ct = affine_transform(ct_ori_pre, trans_input_pre)


                    ct0 = ct.copy()

                    lower, upper = -3, 3
                    mu, sigma = 0, 1
                    X = stats.truncnorm(
                        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                    ct_x_noise, ct_y_noise = X.rvs(2)

                    ct[0] = ct[0] + ct_x_noise * self.opt.hm_disturb * w
                    ct[1] = ct[1] + ct_y_noise * self.opt.hm_disturb * h

                    ct_int = ct.astype(np.int32)

                    if data_generation_mode == 0:

                        # No need to do anything if the rendered center is out of the input image
                        if ct_int[0] >= self.opt.input_w or ct_int[1] >= self.opt.input_h or ct_int[0] < 0 or ct_int[
                            1] < 0:
                            continue

                        # conf = 1 if np.random.random() > self.opt.lost_disturb else 0 # from CenterTrack
                        if np.random.random() > self.opt.lost_disturb:

                            if self.opt.hm_heat_random:
                                conf = np.random.random()
                            else:
                                conf = 1
                        else:
                            conf = 0

                        if conf == 0:  # w/ lost disturbance
                            if self.opt.tracking_label_mode == 0:
                                # Todo: Not sure if we should append nothing in this case
                                cts_pre_list.append(ct / self.opt.down_ratio)  # Original from CenterTrack
                            else:
                                cts_pre_list.append(None)
                        else:
                            # w/o lost disturbance
                            if self.opt.tracking_label_mode == 0:
                                cts_pre_list.append(ct0 / self.opt.down_ratio)
                            else:
                                cts_pre_list.append(ct / self.opt.down_ratio)

                    else:
                        if match_detector_idx == None:
                            if self.opt.tracking_label_mode == 0:
                                # No prediction
                                cts_pre_list.append(ct / self.opt.down_ratio)  # Original from CenterTrack
                            else:
                                cts_pre_list.append(None)
                        else:
                            if self.opt.tracking_label_mode == 0:
                                cts_pre_list.append(ct0 / self.opt.down_ratio)
                            else:
                                cts_pre_list.append(ct_detector / self.opt.down_ratio)

                    track_ids.append(self.opt.c + ann_pre['name'].split('_')[1])

                    # From CenterNet, not used in our case
                    if self.opt.pre_hm:
                        if data_generation_mode == 0:
                            num_kpts = pts_pre[:, 2].sum()
                            if num_kpts == 0:
                                if self.opt.pre_hm:
                                    hm_pre[cls_id, ct_int[1], ct_int[0]] = 0.9999

                    # For corner points
                    # Todo: Currently, hp_radius follows the same way as radius
                    hp_radius = radius
                    # hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    # hp_radius = self.opt.hm_gauss \
                    #     if self.opt.mse_loss else max(0, int(hp_radius))

                    pts_single_pre = np.zeros((num_joints, 2), dtype=np.float32)
                    pts_mask_single_pre = np.zeros((num_joints, 1), dtype=np.uint8)

                    conf_hp_sum = []

                    for j in range(num_joints):

                        # Every point no matter if it is visible or not will be converted first
                        pts_pre[j, :2] = affine_transform(pts_pre[j, :2], trans_input_pre)

                        if data_generation_mode and match_detector_idx is not None:
                            # Do the same thing for the CenterPose output
                            pts_detector[j, :] = affine_transform(pts_detector[j, :], trans_input_pre)

                        if pts_pre[j, 2] > 1:  # Check visibility
                            # If do not assign visible label, it will still be considered if it is in the padding area
                            if pts_pre[j, 0] >= 0 and pts_pre[j, 0] < self.opt.input_w and \
                                    pts_pre[j, 1] >= 0 and pts_pre[j, 1] < self.opt.input_h:

                                pt0 = pts_pre[j, :2].copy()  # ground truth

                                # Add noise
                                lower, upper = -3, 3
                                mu, sigma = 0, 1
                                X = stats.truncnorm(
                                    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                                pts_x_noise, pts_y_noise = X.rvs(2)

                                pts_pre[j, 0] = pts_pre[j, 0] + pts_x_noise * self.opt.hm_hp_disturb * w
                                pts_pre[j, 1] = pts_pre[j, 1] + pts_y_noise * self.opt.hm_hp_disturb * h

                                # conf_hp = 1 if np.random.random() > self.opt.hp_lost_disturb else 0 # CenterTrack idea
                                if np.random.random() > self.opt.hp_lost_disturb:
                                    if self.opt.hm_hp_heat_random:

                                        # Heat is based on the noise level
                                        conf_hp = np.maximum(
                                            1 - 2 ** (np.sqrt(pts_x_noise ** 2 + pts_y_noise ** 2) - 4.5), 0)
                                    else:
                                        conf_hp = 1
                                else:
                                    conf_hp = 0

                                conf_hp_sum.append(conf_hp)

                                pt_int = pts_pre[j, :2].astype(np.int32)

                                if data_generation_mode == 0:
                                    if conf_hp == 0:  # w/ lost disturbance
                                        if self.opt.tracking_label_mode == 0:
                                            # Todo: I think all the keypoints should be given the gt point, which is different from CenterTrack
                                            pts_single_pre[j, :2] = pts_pre[j, :2]  # Original from CenterTrack
                                            pts_mask_single_pre[j] = 1
                                        else:
                                            pts_single_pre[j, :2] = None
                                            pts_mask_single_pre[j] = 0

                                    else:
                                        # w/o lost disturbance
                                        if self.opt.tracking_label_mode == 0:
                                            pts_single_pre[j, :2] = pt0
                                            pts_mask_single_pre[j] = 1
                                        else:
                                            if conf != 0:
                                                pts_single_pre[j, :2] = pts_pre[j, :2]
                                                pts_mask_single_pre[j] = 1
                                            else:
                                                pts_single_pre[j, :2] = None
                                                pts_mask_single_pre[j] = 0
                                else:

                                    if match_detector_idx is None:  # Similar to the lost situation
                                        if self.opt.tracking_label_mode == 0:
                                            pts_single_pre[j, :2] = pts_pre[j, :2]  # Original from CenterTrack
                                            # pts_single_pre[j, :2] = pt0
                                            pts_mask_single_pre[j] = 1
                                        else:
                                            pts_single_pre[j, :2] = None
                                            pts_mask_single_pre[j] = 0
                                    else:
                                        if self.opt.tracking_label_mode == 0:
                                            pts_single_pre[j, :2] = pt0
                                            pts_mask_single_pre[j] = 1
                                        else:
                                            pts_single_pre[j, :2] = pts_detector[j, :2]
                                            pts_mask_single_pre[j] = 1

                                if self.opt.pre_hm_hp:

                                    if data_generation_mode == 0:

                                        # If center point is lost, do nothing with the input.
                                        if conf != 0:
                                            draw_umich_gaussian(hm_hp_pre[j], pt_int, hp_radius, k=conf_hp)

                                            # FP may be unnecessary for keypoint
                                            if np.random.random() < self.opt.hp_fp_disturb:
                                                pt2 = pt0.copy()
                                                # Hard code heatmap disturbance ratio, haven't tried other numbers.
                                                pt2[0] = pt2[0] + np.random.randn() * 0.05 * w
                                                pt2[1] = pt2[1] + np.random.randn() * 0.05 * h
                                                pt2_int = pt2.astype(np.int32)

                                                # draw_umich_gaussian(hm_hp_pre[j], pt2_int, hp_radius, k=conf_hp)
                                                # Todo: fp should not have a very high conf in our case
                                                draw_umich_gaussian(hm_hp_pre[j], pt2_int, hp_radius,
                                                                    k=np.random.uniform(0, 0.3))

                                    else:
                                        if match_detector_idx is not None:

                                            if pts_detector[j, 0] >= 0 and pts_detector[
                                                j, 0] < self.opt.input_w and \
                                                    pts_detector[j, 1] >= 0 and pts_detector[
                                                j, 1] < self.opt.input_h:

                                                pts_detector_int = pts_detector[j, :2].astype(np.int32)

                                                if self.opt.render_hmhp_mode == 1 or self.opt.render_hmhp_mode == 3:
                                                    draw_umich_gaussian(hm_hp_pre[j], pts_detector_int, hp_radius,
                                                                        k=1)
                                                elif self.opt.render_hmhp_mode == 0 or self.opt.render_hmhp_mode == 2:
                                                    # Sometimes, heatmap is missing
                                                    if radius_detector[j, 0] > 0:
                                                        # draw_nvidia_gaussian(hm_hp_pre[j], pts_detector_int,
                                                        #                      radius_detector[j, :],
                                                        #                      k=conf_hp_detector[j])
                                                        draw_umich_gaussian(hm_hp_pre[j], pts_detector_int,
                                                                            hp_radius,
                                                                            k=conf_hp_detector[j])

                    # Collect all the results from the single one
                    pts_pre_list.append(pts_single_pre / self.opt.down_ratio)
                    pts_mask_pre_list.append(pts_mask_single_pre)

                    # For center point
                    if self.opt.pre_hm:

                        if data_generation_mode == 0:
                            if conf != 0 and self.opt.hm_heat_random:
                                conf = np.maximum(1 - 2 ** (np.sqrt(ct_x_noise ** 2 + ct_y_noise ** 2) - 4.5), 0)
                            draw_umich_gaussian(hm_pre[cls_id], ct_int, radius, k=conf)

                            if conf != 0:
                                id_symmetry_pre_list[idx_obj] = id_symmetry_pre

                            # False positive
                            if np.random.random() < self.opt.fp_disturb and self.opt.pre_hm:
                                ct2 = ct0.copy()
                                # Hard code heatmap disturb ratio, haven't tried other numbers.
                                ct2[0] = ct2[0] + np.random.randn() * 0.05 * w
                                ct2[1] = ct2[1] + np.random.randn() * 0.05 * h
                                ct2_int = ct2.astype(np.int32)

                                # Todo: fp should not have a very high conf in our case
                                draw_umich_gaussian(hm_pre[cls_id], ct2_int, radius, k=np.random.uniform(0, 0.4))

                        else:
                            if match_detector_idx != None:

                                ct_int = ct_detector.astype(np.int32)
                                if self.opt.render_hm_mode == 0:
                                    draw_umich_gaussian(hm_pre[0], ct_int, radius)

                                    id_symmetry_pre_list[idx_obj] = id_symmetry_pre

                                elif self.opt.render_hm_mode == 1:
                                    draw_umich_gaussian(hm_pre[0], ct_int, radius, k=result_ori['score'])

                                    if result_ori['score'] != 0:
                                        id_symmetry_pre_list[idx_obj] = id_symmetry_pre

        # </editor-fold>

        # <editor-fold desc="Step2: Work on the current frame">
        cam_projection_matrix = anns['camera_data']['camera_projection_matrix']
        for k in range(num_objs):
            ann = anns['objects'][k]

            # Todo: Only for chair category for now
            if 'symmetric' in ann:
                if ann['symmetric'] == 'True':
                    num_symmetry = 4
                else:
                    num_symmetry = 1

            if self.opt.c == 'cup':
                if self.opt.tracking_task == True and \
                        ((self.opt.mug == False and ann_pre['mug'] == True) or \
                         (self.opt.mug == True and ann_pre['mug'] == False)):
                    continue

            # Todo: Fixed as 0 for now
            cls_id = 0
            pts_ori = np.array(ann['projected_cuboid'])

            # Only apply rotation on gt annotation when symmetry exists
            for id_symmetry in range(num_symmetry):

                if num_symmetry != 1:

                    if self.opt.tracking_task == True and self.opt.pre_hm_hp:
                        # Be consistent with the previous prediction of the corresponding object in tracking_task
                        # if None, generate multiple ground truths
                        if id_symmetry_pre_list[k] is not None and id_symmetry != id_symmetry_pre_list[k]:
                            continue

                    object_rotations = ann['quaternion_xyzw']
                    object_translations = ann['location']
                    keypoints_3d = np.array(ann['keypoints_3d'])

                    M_o2c = np.identity(4)
                    M_o2c[:3, :3] = R.from_quat(object_rotations).as_matrix()
                    M_o2c[:3, 3] = object_translations

                    M_c2o = np.linalg.inv(M_o2c)

                    M_R = rotation_y_matrix(theta * id_symmetry)

                    # Project the rotated 3D keypoint to the image plane
                    M_trans = cam_projection_matrix @ M_o2c @ M_R @ M_c2o

                    new_keypoints_2d = []
                    for i in range(9):
                        projected_point_ori = M_trans @ np.vstack((keypoints_3d[i].reshape(3, -1), 1))
                        projected_point_ori = (projected_point_ori / projected_point_ori[3])[:3]
                        viewport_point = (projected_point_ori + 1.0) / 2.0 * np.array([height, width, 1.0]).reshape(3,
                                                                                                                    1)
                        new_keypoints_2d.append([int(viewport_point[1]), int(viewport_point[0])])

                    pts_ori = new_keypoints_2d

                ct_ori = pts_ori[0]  # center
                pts_ori = pts_ori[1:]  # 8 corners

                # Change visibility, following the protocol of COCO
                pts = np.zeros((len(pts_ori), 3), dtype='int64')
                for idx, p in enumerate(pts_ori):
                    if p[0] >= width or p[0] < 0 or p[1] < 0 or p[1] >= height:
                        pts[idx] = [p[0], p[1], 1]  # labeled but not visible
                    else:
                        pts[idx] = [p[0], p[1], 2]  # labeled and visible

                # Horizontal flip
                if flipped:
                    pts[:, 0] = width - pts[:, 0] - 1
                    for e in self.opt.flip_idx:
                        temp_1 = e[1] - 1
                        temp_0 = e[0] - 1
                        pts[temp_0], pts[temp_1] = pts[temp_1].copy(), pts[temp_0].copy()

                bbox = np.array(bounding_box_rotation(pts, trans_output_rot))

                bbox = np.clip(bbox, 0, output_res - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

                # Filter out GT if most of the keypoints are not visible (more than 4)
                visible_flag = True
                if ct_ori[0] >= width or ct_ori[0] < 0 or ct_ori[1] < 0 or ct_ori[1] >= height:
                    if pts[:, 2].sum() <= 12:
                        visible_flag = False

                if ((h > 0 and w > 0) or (rot != 0)) and visible_flag:
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))

                    if self.opt.center_3D == False:
                        # Todo: Need modification, bbox is not accurate enough as we do not have gt info
                        ct = np.array(
                            [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                        ct_int = ct.astype(np.int32)
                    else:
                        # Todo: Right row, do not consider objects whose center is out of the image
                        if flipped:
                            ct_ori[0] = width - ct_ori[0] - 1
                        ct = affine_transform(ct_ori, trans_output_rot)
                        ct_int = ct.astype(np.int32)
                        if ct_int[0] >= output_res or ct_int[1] >= output_res or ct_int[0] < 0 or ct_int[1] < 0:
                            continue

                    # Todo: Currently, normalized by y axis (up)
                    if self.opt.obj_scale:
                        if self.opt.use_absolute_scale:
                            scale[id_symmetry, k] = np.abs(ann['scale'])
                        else:
                            scale[id_symmetry, k] = np.abs(ann['scale']) / ann['scale'][1]

                        # Todo: Currently, use 0 as the std, not used yet
                        if self.opt.obj_scale_uncertainty:
                            scale_uncertainty[id_symmetry, k] = 0

                    wh[id_symmetry, k] = 1. * w, 1. * h
                    ind[id_symmetry, k] = ct_int[1] * output_res + ct_int[0]
                    reg[id_symmetry, k] = ct - ct_int
                    reg_mask[id_symmetry, k] = 1

                    # From CenterNet, not used in our case
                    num_kpts = pts[:, 2].sum()
                    if num_kpts == 0:
                        hm[id_symmetry, cls_id, ct_int[1], ct_int[0]] = 0.9999
                        reg_mask[id_symmetry, k] = 0

                    # Todo: Currently, hp_radius follows the same way as radius
                    hp_radius = radius
                    # hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    # hp_radius = self.opt.hm_gauss \
                    #     if self.opt.mse_loss else max(0, int(hp_radius))
                    for j in range(num_joints):

                        # Every point no matter if it is visible or not will be converted first
                        pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
                        if pts[j, 2] > 1:  # Check visibility
                            if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
                                    pts[j, 1] >= 0 and pts[j, 1] < output_res:
                                kps[id_symmetry, k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                                # Todo: Use the same std as what is used in heatmap, not used yet
                                kps_displacement_std[id_symmetry, k, j * 2: j * 2 + 2] = hp_radius

                                kps_mask[id_symmetry, k, j * 2: j * 2 + 2] = 1
                                pt_int = pts[j, :2].astype(np.int32)
                                hp_offset[id_symmetry, k * num_joints + j] = pts[j, :2] - pt_int
                                hp_ind[id_symmetry, k * num_joints + j] = pt_int[1] * output_res + pt_int[0]
                                hp_mask[id_symmetry, k * num_joints + j] = 1

                                if 'tracking_hp' in self.opt.heads:

                                    query_id = self.opt.c + ann['name'].split('_')[1]

                                    if query_id in track_ids:
                                        if None not in pts_pre_list[track_ids.index(query_id)][j] and not np.isnan(
                                                pts_pre_list[track_ids.index(query_id)][j]).any():
                                            # previous - current
                                            tracking_hp[id_symmetry, k, j * 2: j * 2 + 2] = \
                                                pts_pre_list[track_ids.index(query_id)][j] - pt_int
                                            tracking_hp_mask[id_symmetry, k, j * 2: j * 2 + 2] = \
                                                pts_mask_pre_list[track_ids.index(query_id)][j] & 1

                                if self.opt.dense_hp:
                                    # Must be before draw center hm gaussian
                                    draw_dense_reg(dense_kps[id_symmetry, j], hm[id_symmetry, cls_id], ct_int,
                                                   pts[j, :2] - ct_int, radius, is_offset=True)
                                    draw_gaussian(dense_kps_mask[id_symmetry, j], ct_int, radius)
                                draw_gaussian(hm_hp[id_symmetry, j], pt_int, hp_radius)

                    # For center point
                    draw_gaussian(hm[id_symmetry, cls_id], ct_int, radius)

                    if 'tracking' in self.opt.heads:

                        query_id = self.opt.c + ann['name'].split('_')[1]

                        if query_id in track_ids:
                            # previous - current
                            if cts_pre_list[track_ids.index(query_id)] is not None:
                                tracking[id_symmetry, k] = cts_pre_list[track_ids.index(query_id)] - ct_int
                                tracking_mask[id_symmetry, k] = 1

                    # 5+16+1+3+2+1+16+16=60
                    # top left (x,y), bottom right (x,y), confidence, keypoint, cls_id, scale,
                    # tracking, tracking_hp, tracking_hp_mask
                    if self.opt.center_3D == False:
                        gt_det_pad[id_symmetry][k] = [ct[0] - w / 2, ct[1] - h / 2,
                                                      ct[0] + w / 2, ct[1] + h / 2, 1] + pts[:, :2].reshape(
                            num_joints * 2).tolist() + [cls_id] + scale[id_symmetry, k].tolist() \
                                                     + tracking[id_symmetry, k].tolist() + [
                                                         tracking_mask[id_symmetry, k]] \
                                                     + tracking_hp[id_symmetry, k].tolist() + tracking_hp_mask[
                                                         id_symmetry, k].tolist()

                    else:
                        gt_det_pad[id_symmetry][k] = [bbox[0], bbox[1], bbox[2], bbox[3], 1] + pts[:, :2].reshape(
                            num_joints * 2).tolist() + [cls_id] + scale[id_symmetry, k].tolist() \
                                                     + tracking[id_symmetry, k].tolist() + [
                                                         tracking_mask[id_symmetry, k]] \
                                                     + tracking_hp[id_symmetry, k].tolist() + tracking_hp_mask[
                                                         id_symmetry, k].tolist()

        # </editor-fold>

        # <editor-fold desc="Update data record">
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind,
               'hps': kps, 'hps_mask': kps_mask}
        if self.opt.hps_uncertainty:
            ret.update({'hps_uncertainty': kps_displacement_std})

        if self.opt.pre_img:
            ret.update({'pre_img': img_pre})

        # Tracking stuff
        if self.opt.pre_hm:
            ret.update({'pre_hm': hm_pre})
        if self.opt.pre_hm_hp:
            ret.update({'pre_hm_hp': hm_hp_pre})
        if self.opt.tracking:
            ret.update({'tracking': tracking, 'tracking_mask': tracking_mask})
        if self.opt.tracking_hp:
            ret.update({'tracking_hp': tracking_hp, 'tracking_hp_mask': tracking_hp_mask})

        if self.opt.obj_scale:
            ret.update({'scale': scale})
            if self.opt.obj_scale_uncertainty:
                ret.update({'scale_uncertainty': scale_uncertainty})
        if self.opt.reg_bbox:
            ret.update({'wh': wh})
        if self.opt.dense_hp:
            dense_kps = dense_kps.reshape(num_symmetry, num_joints * 2, output_res, output_res)
            dense_kps_mask = dense_kps_mask.reshape(num_symmetry,
                                                    num_joints, 1, output_res, output_res)
            dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=2)
            dense_kps_mask = dense_kps_mask.reshape(num_symmetry,
                                                    num_joints * 2, output_res, output_res)
            ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
            del ret['hps'], ret['hps_mask']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.hm_hp:
            ret.update({'hm_hp': hm_hp})
        if self.opt.reg_hp_offset:
            ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
        if self.opt.debug > 0 or not self.split == 'train':
            meta = {'c': c, 's': s, 'gt_det': gt_det_pad, 'img_id': frame_id}

            ret['meta'] = meta
        # </editor-fold>

        return ret
