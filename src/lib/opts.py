# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Basic experiment setting
        self.parser.add_argument('--task', default='object_pose',
                                 help='object_pose')

        self.parser.add_argument('--dataset', default='objectron',
                                 help='objectron')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--test', action='store_true')  # run on validation dataset
        self.parser.add_argument('--debug', type=int, default=1,
                                 help='level of visualization.'
                                      '1: only show the final detection results'
                                      '2: show the network output features'
                                      '3: use matplotlib to display'  # useful when launching training with ipython notebook
                                      '4: save visualizations to disk and no display'
                                      '5: save visualizations to tensorboard and no display'
                                      '6: save visualizations to disk and no display (for evaluation, path is different from 4)')
        self.parser.add_argument('--demo', default='',
                                 help='path to image/ image folders/ video. '
                                      'or "webcam"')
        self.parser.add_argument('--show_axes', action='store_true',
                                 help='Whether to show axes in demo mode (which will requrire the OpenCV way return)')
        self.parser.add_argument('--demo_save', default='../demo/',
                                 help='path to save the results')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # System
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        # Log
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save models to disk without overriding')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                                 help='visualization threshold.')
        self.parser.add_argument('--debugger_theme', default='white',
                                 choices=['white', 'black'])

        self.parser.add_argument('--paper_display', action='store_true',
                                 help='whether to use paper display (blue edges & show axes)')

        # Model
        self.parser.add_argument('--arch', default='dla_34',
                                 help='model architecture. Currently tested'
                                      'res_101 | dla_34 | dlav1_34 ')
        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '64 for resnets and 256 for dla.')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')

        # Input
        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')

        # Train
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='90,120',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=20,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                 help='include validation in training and '
                                      'test on test set')
        # Test
        self.parser.add_argument('--test_scales', type=str, default='1',
                                 help='multi scale test augmentation.')
        self.parser.add_argument('--nms', action='store_true',
                                 help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=100,
                                 help='max number of output objects.')
        self.parser.add_argument('--not_prefetch_test', action='store_true',
                                 help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--fix_short', type=int, default=-1)
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')

        # Dataset generation
        self.parser.add_argument('--not_rand_crop', action='store_true',
                                 help='not use the random crop data augmentation'
                                      'from CornerNet.')
        self.parser.add_argument('--shift', type=float, default=0.05,
                                 help='when not using random crop'
                                      'apply shift augmentation.')
        self.parser.add_argument('--scale', type=float, default=0.4,
                                 help='when not using random crop'
                                      'apply scale augmentation.')
        self.parser.add_argument('--rotate', type=float, default=0,
                                 help='when not using random crop'
                                      'apply rotation augmentation.')
        self.parser.add_argument('--flip', type=float, default=0.5,
                                 help='probability of applying flip augmentation.')
        self.parser.add_argument('--no_color_aug', action='store_true',
                                 help='not use the color augmentation '
                                      'from CornerNet')

        self.parser.add_argument('--aug_rot', type=float, default=0,
                                 help='probability of applying '
                                      'rotation augmentation.')

        # Loss
        self.parser.add_argument('--mse_loss', action='store_true',
                                 help='use mse loss or focal loss to train '
                                      'keypoint heatmaps.')
        self.parser.add_argument('--reg_loss', default='l1',
                                 help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for object center heatmap.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for sub-pixel offsets (both center and keypoints).')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for 2D bounding box size.')
        self.parser.add_argument('--hp_weight', type=float, default=1,
                                 help='loss weight for keypoint displacements.')
        self.parser.add_argument('--hm_hp_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmap.')

        self.parser.add_argument('--not_reg_offset', action='store_true',
                                 help='not regress local offset.')

        # More option
        self.parser.add_argument('--center_thresh', type=float, default=0.3,
                                 help='threshold for centermap, just for debug.')
        self.parser.add_argument('--dense_hp', action='store_true',
                                 help='apply weighted pose regression near center '
                                      'or just apply regression on center point.')
        self.parser.add_argument('--not_hm_hp', action='store_true',
                                 help='not estimate keypoint heatmap, '
                                      'directly use the keypoint offset from center.')
        self.parser.add_argument('--not_reg_hp_offset', action='store_true',
                                 help='not regress local offset for '
                                      'keypoint heatmaps.')

        self.parser.add_argument('--not_reg_bbox', action='store_true',
                                 help='not regression bounding box size.')

        self.parser.add_argument('--c', default='chair',
                                 help='category in objectron dataset')
        self.parser.add_argument('--hps_uncertainty', action='store_true',
                                 help='Predict hps_uncertainty or not.')
        self.parser.add_argument('--obj_scale', action='store_true',
                                 help='Predict obj_scale or not.')
        self.parser.add_argument('--obj_scale_uncertainty', action='store_true',
                                 help='Predict obj_scale_uncertainty or not.')
        self.parser.add_argument('--obj_scale_weight', type=float, default=1,
                                 help='loss weight for relative cuboid dimensions.')
        self.parser.add_argument('--use_pnp', action='store_true',
                                 help='use pnp')
        self.parser.add_argument('--mug', action='store_true',
                                 help='target is a mug (only useful for training and evaluation)')
        self.parser.add_argument('--num_symmetry', default=12,
                                 help='the number of the symmetry for ground truths, idea comes from NOCS paper https://arxiv.org/abs/1901.02970')

        self.parser.add_argument('--cam_intrinsic', default=None, nargs='+', type=float,
                                 help='the camera intrinsics parameters. Input a list of 9 numbers following the column order.')

        self.parser.add_argument(
            '--rep_mode',
            default=1,
            type=int,
            help="0: 8 selective representation"
                 "1: 16 all representation"  # Tracking task will still involve gaussian fitting for heatmap
                 "2: Sampling representation"  # Involve gaussian fitting for heatmap
                 "3: Displacement only representation"
                 "4: Heatmap only representation"
        )
        self.parser.add_argument(
            '--data_generation_mode_ratio',
            default=0,
            type=float,
            help="the ratio of using centerpose detector mode, otherwise using noisy GT mode"
        )

        # Not used yet
        self.parser.add_argument('--center_3D', action='store_true',
                                 help='Use the 2D projection of the 3D centroid or not.')
        self.parser.add_argument('--use_residual', action='store_true',
                                 help='apply residual representation')
        self.parser.add_argument('--use_absolute_scale', action='store_true',
                                 help='apply absolute representation')
        self.parser.add_argument('--new_data_augmentation', action='store_true',
                                 help='apply some new data_augmentation steps')

        # Filtering process
        self.parser.add_argument('--balance_coefficient', default=
        dict({'bike': 2, 'book': 2, 'bottle': 2, 'camera': 2, 'cereal_box': 2, 'chair': 2,
              'cup': 2, 'mug': 2, 'laptop': 2, 'shoe': 2}))
        self.parser.add_argument('--conf_border', default=
        dict({'bike': [3, 9], 'book': [3, 9], 'bottle': [3, 9], 'camera': [3, 9], 'cereal_box': [3, 9], 'chair': [3, 9],
              'cup': [3, 9], 'mug': [3, 9], 'laptop': [3, 9], 'shoe': [3, 9]}))

        self.parser.add_argument('--R', type=float, default=20,
                                 help="fixed value for the uncertain of velocity in the Kalman Filter")

        self.parser.add_argument('--refined_Kalman', action='store_true', help="enable kalman filter for CenterPose")

        # Tracking
        self.parser.add_argument('--tracking_task', action='store_true')
        self.parser.add_argument('--tracking', action='store_true', help='track the center or not')
        self.parser.add_argument('--tracking_hp', action='store_true', help='track the corner keypoints or not')
        self.parser.add_argument('--pre_hm', action='store_true')
        self.parser.add_argument('--pre_hm_hp', action='store_true')
        self.parser.add_argument('--same_aug_pre', action='store_true')
        self.parser.add_argument('--hm_heat_random', action='store_true',
                                 help='activate random heat for hm. The value is related to hm_disturb')
        self.parser.add_argument('--hm_disturb', type=float, default=0)
        self.parser.add_argument('--lost_disturb', type=float, default=0)
        self.parser.add_argument('--fp_disturb', type=float, default=0)
        self.parser.add_argument('--hm_hp_heat_random', action='store_true', help='activate random heat for hm_hp')
        self.parser.add_argument('--hm_hp_disturb', type=float, default=0)
        self.parser.add_argument('--hp_lost_disturb', type=float, default=0)
        self.parser.add_argument('--hp_fp_disturb', type=float, default=0)

        self.parser.add_argument('--KL_scale_uncertainty', type=float, default=0.1)
        self.parser.add_argument('--KL_kps_uncertainty', type=float, default=0.1)

        self.parser.add_argument('--tracking_label_mode', default=1,
                                 help="0: CenterPose idea"
                                      "1: CenterPoseTrack idea "
                                 )
        self.parser.add_argument('--render_hm_mode', default=1,
                                 help="0: with conf=1"
                                      "1: with heat "
                                 )

        self.parser.add_argument('--render_hmhp_mode', default=2,
                                 help="0: with distance + heat"
                                      "1: with distance + conf=1 heat "
                                      "2: with pnp + heat"
                                      "3: with pnp + conf=1 heat"
                                 )

        self.parser.add_argument('--pre_thresh', type=float, default=-1,
                                 help='for heatmap rendering from last frame')
        self.parser.add_argument('--track_thresh', type=float, default=0.3,
                                 help='for changing pre_thresh/new_thresh/vis_thresh altogether')
        self.parser.add_argument('--new_thresh', type=float, default=0.3,
                                 help='for creating a new instance from last frame result')
        self.parser.add_argument('--max_frame_dist', type=int, default=3)
        self.parser.add_argument('--pre_img', action='store_true')
        self.parser.add_argument('--hungarian', action='store_true')
        self.parser.add_argument('--kalman', action='store_true',
                                 help='enable Kalman Filter for offline update')
        self.parser.add_argument('--scale_pool', action='store_true',
                                 help='enable bayesian fusion on scale for offline update')
        self.parser.add_argument('--max_age', type=int, default=5)

        self.parser.add_argument('--tracking_weight', type=float, default=1,
                                 help='weight for center track offset')
        self.parser.add_argument('--tracking_hp_weight', type=float, default=0.5,
                                 help='weight for keypoint track offset')

        self.parser.add_argument('--gt_pre_hm_hmhp', action='store_true',
                                 help='given ground truth heatmaps (center+keypoints from previous frame) for every frame')
        self.parser.add_argument('--gt_pre_hm_hmhp_first', action='store_true',
                                 help='given ground truth heatmaps only at first frame')
        self.parser.add_argument('--empty_pre_hm', action='store_true',
                                 help='only use empty heatmap')

        # Ground truth validation, borrowed from CenterNet, but not used yet
        self.parser.add_argument('--eval_oracle_hm', action='store_true',
                                 help='use ground center heatmap.')
        self.parser.add_argument('--eval_oracle_wh', action='store_true',
                                 help='use ground truth bounding box size.')
        self.parser.add_argument('--eval_oracle_offset', action='store_true',
                                 help='use ground truth local heatmap offset.')
        self.parser.add_argument('--eval_oracle_kps', action='store_true',
                                 help='use ground truth human pose offset.')
        self.parser.add_argument('--eval_oracle_hmhp', action='store_true',
                                 help='use ground truth human joint heatmaps.')
        self.parser.add_argument('--eval_oracle_hp_offset', action='store_true',
                                 help='use ground truth human joint local offset.')
        self.parser.add_argument('--eval_oracle_dep', action='store_true',
                                 help='use ground truth depth.')

    def parse(self, opt):
        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
        opt.reg_offset = not opt.not_reg_offset
        opt.reg_bbox = not opt.not_reg_bbox
        opt.hm_hp = not opt.not_hm_hp
        opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 64
        opt.pad = 127 if 'hourglass' in opt.arch else 31
        opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

        if opt.trainval:
            opt.val_intervals = 100000000

        if opt.debug > 0:
            opt.num_workers = 0
            opt.batch_size = 1
            opt.gpus = [opt.gpus[0]]
            opt.master_batch_size = -1

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        opt.data_dir = os.path.join(opt.root_dir, 'data')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        opt.flip_idx = dataset.flip_idx

        opt.heads = {'hm': opt.num_classes, 'wh': 2, 'hps': 16}

        if opt.hps_uncertainty:
            opt.heads.update({'hps_uncertainty': 16})
        if opt.reg_offset:
            opt.heads.update({'reg': 2})
        if opt.hm_hp:
            opt.heads.update({'hm_hp': 8})
        if opt.reg_hp_offset:
            opt.heads.update({'hp_offset': 2})
        if opt.obj_scale:
            opt.heads.update({'scale': 3})
            if opt.obj_scale_uncertainty:
                opt.heads.update({'scale_uncertainty': 3})

        # Not used yet
        if opt.use_residual:
            if opt.c == 'cup' and opt.mug:
                opt.dimension_ref = dataset.dimension_ref['mug']
            else:
                opt.dimension_ref = dataset.dimension_ref[opt.c]

            if opt.use_absolute_scale:
                opt.dimension_ref = opt.dimension_ref[0][0:3]
            else:
                # Relative scale
                opt.dimension_ref = [opt.dimension_ref[0][3], 1, opt.dimension_ref[0][4]]

        # Tracking stuff
        if opt.tracking == True:
            opt.heads.update({'tracking': 2})
        if opt.tracking_hp == True:
            opt.heads.update({'tracking_hp': 16})

        print('heads', opt.heads)
        return opt

    def init(self, opt):
        default_dataset_info = {
            'object_pose': {
                'default_resolution': [512, 512],
                'num_classes': 1,
                'mean': [0.408, 0.447, 0.470],
                'std': [0.289, 0.274, 0.278],
                'dataset': 'objectron',
                'num_joints': 8,
                'flip_idx': [[1, 5], [3, 7], [2, 6], [4, 8]],

                # Not used yet
                'dimension_ref': {
                    'bike': [[0.65320896, 1.021797894, 1.519635599, 0.6520559199, 1.506392621],
                             [0.1179380561, 0.176747817, 0.2981715678, 0.1667947895, 0.3830536275]],
                    'book': [[0.225618019, 0.03949624326, 0.1625821624, 7.021850281, 5.064694187],
                             [0.1687487664, 0.07391230822, 0.06436673199, 3.59629568, 2.723290812]],
                    'bottle': [
                        [0.07889784977450116, 0.24127451915330908, 0.0723714257114412, 0.33644069262302545,
                         0.3091134992864717, ],
                        [0.02984649578071775, 0.06381390122918497, 0.03088144838560917, 0.11052240441921059,
                         0.13327627592012867, ]],
                    'camera': [
                        [0.11989848700326843, 0.08226238775595619, 0.09871718158089632, 1.507216484439368,
                         1.1569407159290284, ],
                        [0.021177290310316968, 0.02158788017191602, 0.055673710278419844, 0.28789183678046854,
                         0.5342094080365904, ]],
                    'cereal_box': [
                        [0.19202754401417296, 0.2593114001714919, 0.07723794925413519, 0.7542602699204104,
                         0.29441151268928173, ],
                        [0.08481640897407464, 0.09999915952084068, 0.09495429981036707, 0.19829004029411457,
                         0.2744797990483879, ]],
                    'chair': [
                        [0.5740664085137888, 0.8434027515832329, 0.6051523831888338, 0.6949691013776601,
                         0.7326891354260606, ],
                        [0.12853104253707456, 0.14852086453095492, 0.13428881418587957, 0.16897092539619352,
                         0.18636134566748525, ]],
                    'cup': [
                        [0.08587637391801063, 0.12025228955138188, 0.08486836104868696, 0.7812126934904675,
                         0.7697895244331658, ],
                        [0.05886805978497525, 0.06794896438246326, 0.05875681990718713, 0.2887038681446475,
                         0.283821205157399, ]],
                    'mug': [
                        [0.14799136566553112, 0.09729087667918128, 0.08845449667169905, 1.3875694883045138,
                         1.0224997119392225, ],
                        [1.0488828523223728, 0.2552672927963539, 0.039095350310480705, 0.3947832854104711,
                         0.31089415283872546, ]],
                    'laptop': [
                        [0.33685059747485196, 0.1528068814247063, 0.2781020624738614, 35.920214652427696,
                         23.941173992376903, ],
                        [0.03529983948867832, 0.07017080198389423, 0.0665823136876069, 391.915687801732,
                         254.21325950495455, ]],
                    'shoe': [
                        [0.10308848289662519, 0.10932616184503478, 0.2611737789760352, 1.0301976264129833,
                         2.6157393112424328, ],
                        [0.02274768925924402, 0.044958380226590516, 0.04589720205423542, 0.3271000267177176,
                         0.8460337534776092, ]],
                },

            },

        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
