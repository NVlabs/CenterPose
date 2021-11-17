# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.utils.data
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import collate_fn_filtered
from lib.trains.train_factory import train_factory
import time
import numpy as np

from lib.datasets.dataset_combined import ObjectPoseDataset


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    Dataset = ObjectPoseDataset

    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    val_dataset = Dataset(opt, 'val')

    if opt.tracking_task == True:
        val_dataset_subset = torch.utils.data.Subset(val_dataset, range(0, len(val_dataset), 15))
    else:
        val_dataset_subset = val_dataset

    val_loader = torch.utils.data.DataLoader(
        val_dataset_subset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_filtered
    )

    print('Starting testing...')

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):

        # Reload model
        model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)

        new_model_path = os.path.join(os.path.dirname(opt.load_model), f'{opt.c}_{epoch}.pth')
        if not os.path.exists(new_model_path):
            continue

        model, _, _ = load_model(
            model, new_model_path, optimizer, opt.resume, opt.lr, opt.lr_step)
        trainer.model = model
        trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

        logger.write('epoch: {} | '.format(epoch))  # txt logging

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:

            with torch.no_grad():
                log_dict_val, preds, log_imgs = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('val_{} {:8f} | '.format(k, v))
            logger.img_summary('val', log_imgs, epoch)

        logger.write('\n\n')

    logger.close()


if __name__ == '__main__':

    # Default params with commandline input
    opt = opts()
    opt = opt.parser.parse_args()

    # Local configuration
    # Note: Assume multiple training weights are saved in the same folder, e.g., chair_20.pth
    # We will test on all the weights
    opt.load_model = "Path2weights"
    opt.c = 'bike'
    opt.arch = 'dlav1_34'
    opt.obj_scale = True
    opt.obj_scale_weight = 1
    opt.mug = False

    # Training param
    opt.exp_id = f'objectron_{opt.c}_{opt.arch}'
    opt.num_epochs = 140
    opt.val_intervals = 5
    opt.lr_step = '90,120'
    opt.batch_size = 8
    opt.lr = 6e-5
    opt.gpus = '0'
    opt.num_workers = 4
    opt.print_iter = 1
    opt.debug = 5
    opt.save_all = True

    # opt.tracking_task = True
    # Tracking related
    if opt.tracking_task == True:
        opt.obj_scale_uncertainty = True
        opt.hps_uncertainty = True
        opt.tracking_label_mode = 1
        opt.render_hm_mode = 1
        opt.render_hmhp_mode = 2

        opt.pre_img = True
        opt.pre_hm = True
        opt.tracking = True
        opt.pre_hm_hp = True
        opt.tracking_hp = True

        opt.shift = 0.05
        opt.scale = 0.05

        # For hm
        opt.hm_heat_random = True
        opt.hm_disturb = 0.05
        opt.lost_disturb = 0.2
        opt.fp_disturb = 0.1

        # For hm_hp
        opt.hm_hp_heat_random = True
        opt.hm_hp_disturb = 0.03
        opt.hp_lost_disturb = 0.1
        opt.hp_fp_disturb = 0.05

        opt.max_frame_dist = 3

        # Currently, CenterPose mode does not support symmetrical objects
        if opt.c in ['bottle', 'chair', 'cup']:
            opt.data_generation_mode_ratio = 0
        else:
            opt.data_generation_mode_ratio = 0.3

        print('Running tracking')

        opt.vis_thresh = max(opt.track_thresh, opt.vis_thresh)
        opt.pre_thresh = max(opt.track_thresh, opt.pre_thresh)
        opt.new_thresh = max(opt.track_thresh, opt.new_thresh)
        print('Using tracking threshold for out threshold!', opt.track_thresh)

    # Copy from parse function from opts.py
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

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    opt.save_dir = os.path.join(opt.exp_dir, f'{opt.exp_id}_{time_str}')
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)

    main(opt)
