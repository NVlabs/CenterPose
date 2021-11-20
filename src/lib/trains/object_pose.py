# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from lib.models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss, RegKLDKeyLoss, RegKLDScaleLoss
from lib.models.decode import object_pose_decode
from lib.models.utils import _sigmoid
from lib.utils.debugger import Debugger
from lib.utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

import math


class ObjectPoseLoss(torch.nn.Module):
    def __init__(self, opt):
        super(ObjectPoseLoss, self).__init__()
        self.crit = FocalLoss()

        # Todo: currently, do not support MSELoss
        self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()

        # Todo: currently, do not support opt.dense_hp for now
        self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else \
            torch.nn.L1Loss(reduction='sum')

        self.crit_kp_uncertainty = RegKLDKeyLoss()

        # Todo: currently, do not support RegLoss
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None

        self.crit_reg_uncertainty = RegKLDScaleLoss()

        self.opt = opt

    def forward(self, outputs, batch, phase):
        opt = self.opt

        # hm: object center heatmap
        # wh: 2D bounding box size
        # hps/hp: keypoint displacements
        # reg/off: sub-pixel offset for object center
        # hm_hp: keypoint heatmaps
        # hp_offset: sub-pixel offsets for keypoints
        # scale/obj_scale: relative cuboid dimensions

        hm_loss, wh_loss, hp_loss = 0, 0, 0
        off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0
        obj_scale_loss = 0
        tracking_loss = 0
        tracking_hp_loss = 0

        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = _sigmoid(output['hm'])
            if opt.hm_hp and not opt.mse_loss:
                output['hm_hp'] = _sigmoid(output['hm_hp'])

            if opt.eval_oracle_hmhp:
                output['hm_hp'] = batch['hm_hp']
            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_kps:
                if opt.dense_hp:
                    output['hps'] = batch['dense_hps']
                else:
                    output['hps'] = torch.from_numpy(gen_oracle_map(
                        batch['hps'].detach().cpu().numpy(),
                        batch['ind'].detach().cpu().numpy(),
                        opt.output_res, opt.output_res)).to(opt.device)
            if opt.eval_oracle_hp_offset:
                output['hp_offset'] = torch.from_numpy(gen_oracle_map(
                    batch['hp_offset'].detach().cpu().numpy(),
                    batch['hp_ind'].detach().cpu().numpy(),
                    opt.output_res, opt.output_res)).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.dense_hp:
                mask_weight = batch['dense_hps_mask'].sum() + 1e-4
                hp_loss += (self.crit_kp(output['hps'] * batch['dense_hps_mask'],
                                         batch['dense_hps'] * batch['dense_hps_mask']) /
                            mask_weight) / opt.num_stacks
            else:

                if not opt.hps_uncertainty or phase == 'val':

                    hp_loss += self.crit_kp(output['hps'], batch['hps_mask'],
                                            batch['ind'], batch['hps']) / opt.num_stacks
                else:
                    # KLD loss
                    hp_loss += self.crit_kp_uncertainty(output['hps'], output['hps_uncertainty'], batch['hps_mask'],
                                                        batch['ind'], batch['hps'], self.opt) / opt.num_stacks

            if opt.reg_bbox and opt.wh_weight > 0:
                wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                                         batch['ind'], batch['wh']) / opt.num_stacks

            # Add obj_scale
            if opt.obj_scale and opt.obj_scale_weight > 0:

                if phase == 'train':
                    if not opt.obj_scale_uncertainty:
                        if opt.use_residual:
                            obj_scale_loss += self.crit_reg(output['scale'], batch['reg_mask'],
                                                            batch['ind'], batch['scale'],
                                                            dimension_ref=opt.dimension_ref) / opt.num_stacks
                        else:
                            obj_scale_loss += self.crit_reg(output['scale'], batch['reg_mask'],
                                                            batch['ind'], batch['scale']) / opt.num_stacks

                    else:
                        # KLD loss
                        obj_scale_loss += self.crit_reg_uncertainty(output['scale'], output['scale_uncertainty'],
                                                                    batch['reg_mask'],
                                                                    batch['ind'], batch['scale'],
                                                                    self.opt) / opt.num_stacks
                else:
                    # Calculate relative loss only on validation phase
                    obj_scale_loss += self.crit_reg(output['scale'], batch['reg_mask'],
                                                    batch['ind'], batch['scale'], relative_loss=True) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks
            if opt.reg_hp_offset and opt.off_weight > 0:
                hp_offset_loss += self.crit_reg(
                    output['hp_offset'], batch['hp_mask'],
                    batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
            if opt.hm_hp and opt.hm_hp_weight > 0:
                hm_hp_loss += self.crit_hm_hp(
                    output['hm_hp'], batch['hm_hp']) / opt.num_stacks

            if opt.tracking and opt.tracking_weight > 0:
                tracking_loss += self.crit_reg(
                    output['tracking'], batch['tracking_mask'],
                    batch['ind'], batch['tracking']) / opt.num_stacks

            if opt.tracking_hp and opt.tracking_hp_weight > 0:
                tracking_hp_loss += self.crit_kp(
                    output['tracking_hp'], batch['tracking_hp_mask'],
                    batch['ind'], batch['tracking_hp']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss + opt.hp_weight * hp_loss + \
               opt.hm_hp_weight * hm_hp_loss + opt.off_weight * hp_offset_loss + \
               opt.obj_scale_weight * obj_scale_loss + \
               opt.tracking_weight * tracking_loss + \
               opt.tracking_hp_weight * tracking_hp_loss

        # Calculate the valid_mask where samples are valid
        valid_mask = torch.gt(batch['ind'].sum(dim=2), 0)
        pos_inf = torch.zeros_like(loss)
        pos_inf[~valid_mask] = math.inf

        # Argmin to choose the best matched gt
        choice_list = torch.argmin(loss * valid_mask.float() + pos_inf, dim=1)

        # Update all the losses according to the choice (7+2 in total for now)
        hm_loss = torch.stack([hm_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()
        hp_loss = torch.stack([hp_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()

        if opt.reg_bbox and opt.wh_weight > 0:
            wh_loss = torch.stack([wh_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()
        if opt.obj_scale and opt.obj_scale_weight > 0:
            obj_scale_loss = torch.stack([obj_scale_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()
        if opt.reg_offset and opt.off_weight > 0:
            off_loss = torch.stack([off_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()
        if opt.reg_hp_offset and opt.off_weight > 0:
            hp_offset_loss = torch.stack([hp_offset_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()
        if opt.hm_hp and opt.hm_hp_weight > 0:
            hm_hp_loss = torch.stack([hm_hp_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()
        if opt.tracking and opt.tracking_weight > 0:
            tracking_loss = torch.stack([tracking_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()
        if opt.tracking_hp and opt.tracking_hp_weight > 0:
            tracking_hp_loss = torch.stack(
                [tracking_hp_loss[idx][choice] for idx, choice in enumerate(choice_list)]).mean()

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss + opt.hp_weight * hp_loss + \
               opt.hm_hp_weight * hm_hp_loss + opt.off_weight * hp_offset_loss + \
               opt.obj_scale_weight * obj_scale_loss + \
               opt.tracking_weight * tracking_loss + \
               opt.tracking_hp_weight * tracking_hp_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'hp_loss': hp_loss,
                      'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'obj_scale_loss': obj_scale_loss,
                      'tracking_loss': tracking_loss,
                      'tracking_hp_loss': tracking_hp_loss,
                      }

        # Fix the bug in multi gpus
        for key in loss_stats:
            if isinstance(loss_stats[key], int):
                loss_stats[key] = torch.from_numpy(np.array(loss_stats[key])).type(torch.FloatTensor).to(
                    'cuda' if opt.gpus[0] >= 0 else 'cpu')
        return loss, loss_stats, choice_list


class ObjectPoseTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(ObjectPoseTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'hp_loss', 'hm_hp_loss',
                       'hp_offset_loss', 'wh_loss', 'off_loss', 'obj_scale_loss', 'tracking_loss', 'tracking_hp_loss']
        loss = ObjectPoseLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id, choice_list):
        opt = self.opt

        hps_uncertainty = output['hps_uncertainty'] if opt.hps_uncertainty else None
        reg = output['reg'] if opt.reg_offset else None
        hm_hp = output['hm_hp'] if opt.hm_hp else None
        hp_offset = output['hp_offset'] if opt.reg_hp_offset else None
        obj_scale = output['scale'] if opt.obj_scale else None
        obj_scale_uncertainty = output['scale_uncertainty'] if opt.obj_scale_uncertainty else None
        wh = output['wh'] if opt.reg_bbox else None
        tracking = output['tracking'] if 'tracking' in opt.heads else None
        tracking_hp = output['tracking_hp'] if 'tracking_hp' in opt.heads else None

        dets = object_pose_decode(
            output['hm'], output['hps'], wh=wh, kps_displacement_std=hps_uncertainty, obj_scale=obj_scale,
            obj_scale_uncertainty=obj_scale_uncertainty,
            reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, tracking=tracking, tracking_hp=tracking_hp, opt=self.opt)

        for k in dets:
            dets[k] = dets[k].detach().cpu().numpy()

        dets['bboxes'] *= opt.input_res / opt.output_res
        dets['kps'] *= opt.input_res / opt.output_res

        if 'tracking' in opt.heads:
            dets['tracking'] *= opt.input_res / opt.output_res

        if 'tracking_hp' in opt.heads:
            dets['tracking_hp'] *= opt.input_res / opt.output_res

        # Todo: Right now, only keep the best matched gt
        dets_gt = batch['meta']['gt_det']
        dets_gt = torch.stack([dets_gt[idx][choice] for idx, choice in enumerate(choice_list)])
        dets_gt = dets_gt.numpy()

        dets_gt[:, :, :4] *= opt.input_res / opt.output_res  # bbox
        dets_gt[:, :, 5:21] *= opt.input_res / opt.output_res  # kps
        dets_gt[:, :, 25:27] *= opt.input_res / opt.output_res  # tracking
        dets_gt[:, :, 28:44] *= opt.input_res / opt.output_res  # tracking_hp

        for i in range(1):  # We only care about the first sample in the batch
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)

            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i][choice_list[i]].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'out_hm_pred')
            debugger.add_blend_img(img, gt, 'out_hm_gt')

            if 'pre_img' in batch:
                pre_img = batch['pre_img'][i].detach().cpu().numpy().transpose(1, 2, 0)
                pre_img = np.clip(((pre_img * opt.std + opt.mean) * 255), 0, 255).astype(np.uint8)

                if 'pre_hm' in batch:
                    pre_hm = debugger.gen_colormap(
                        batch['pre_hm'][i].detach().cpu().numpy())
                    debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')

                if 'pre_hm_hp' in batch:
                    pre_hmhp = debugger.gen_colormap_hp(
                        batch['pre_hm_hp'][i].detach().cpu().numpy())
                    debugger.add_blend_img(pre_img, pre_hmhp, 'pre_hmhp')

            # Predictions
            debugger.add_img(img, img_id='out_img_pred')
            for k in range(len(dets['scores'][i])):
                if dets['scores'][i][k][0] > opt.center_thresh:

                    if self.opt.reg_bbox:
                        debugger.add_coco_bbox(dets['bboxes'][i][k], dets['clses'][i][k],
                                               dets['scores'][i][k][0], img_id='out_img_pred')
                    debugger.add_coco_hp(dets['kps'][i][k], img_id='out_img_pred')

                    if self.opt.obj_scale == True:
                        if self.opt.reg_bbox:
                            debugger.add_obj_scale(dets['bboxes'][i][k], dets['obj_scale'][i][k], img_id='out_img_pred')
                        else:
                            # Todo: A temporary location, need updates
                            debugger.add_obj_scale([20, 20, 0, 0], dets['obj_scale'][i][k], img_id='out_img_pred')

                    if 'tracking' in opt.heads:
                        debugger.add_arrow(
                            [(dets['bboxes'][i][k][0] + dets['bboxes'][i][k][2]) / 2,
                             (dets['bboxes'][i][k][1] + dets['bboxes'][i][k][3]) / 2, ],
                            dets['tracking'][i][k],
                            img_id='out_img_pred', c=(0, 255, 255))  # yellow
                        debugger.add_arrow(
                            [(dets['bboxes'][i][k][0] + dets['bboxes'][i][k][2]) / 2,
                             (dets['bboxes'][i][k][1] + dets['bboxes'][i][k][3]) / 2, ],
                            dets['tracking'][i][k],
                            img_id='pre_hm', c=(0, 255, 255))  # yellow

                    if 'tracking_hp' in opt.heads:

                        for idx in range(8):

                            if dets['kps'][i][k][idx * 2] == 0 and dets['kps'][i][k][idx * 2 + 1] == 0:
                                continue
                            debugger.add_arrow(
                                dets['kps'][i][k][idx * 2:idx * 2 + 2],
                                dets['tracking_hp'][i][k][idx * 2:idx * 2 + 2],
                                img_id='out_img_pred', c=(0, 0, 255))  # red
                            debugger.add_arrow(
                                dets['kps'][i][k][idx * 2:idx * 2 + 2],
                                dets['tracking_hp'][i][k][idx * 2:idx * 2 + 2],
                                img_id='pre_hmhp', c=(0, 0, 255))  # red

            if opt.hm_hp:
                pred = debugger.gen_colormap_hp(output['hm_hp'][i].detach().cpu().numpy())
                debugger.add_blend_img(img, pred, 'out_hmhp_pred')

            # Ground truth
            debugger.add_img(img, img_id='out_img_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    if self.opt.reg_bbox:
                        debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, 21],
                                               dets_gt[i, k, 4], img_id='out_img_gt')
                    debugger.add_coco_hp(dets_gt[i, k, 5:21], img_id='out_img_gt', pred_flag='gt')

                    if self.opt.obj_scale == True:
                        if self.opt.reg_bbox:
                            debugger.add_obj_scale(dets_gt[i, k, :4], dets_gt[i, k, 22:25], img_id='out_img_gt',
                                                   pred_flag='gt')
                        else:
                            # Todo: A temporary location, need updates
                            debugger.add_obj_scale([20, 20, 0, 0], dets_gt[i, k, 22:25], img_id='out_img_gt',
                                                   pred_flag='gt')

                    if 'tracking' in opt.heads:
                        # first param: current
                        # second param: previous - current
                        if dets_gt[i][k][27] == 1:
                            debugger.add_arrow(
                                [(dets_gt[i][k][0] + dets_gt[i][k][2]) / 2,
                                 (dets_gt[i][k][1] + dets_gt[i][k][3]) / 2, ],
                                [dets_gt[i][k][25], dets_gt[i][k][26]],
                                img_id='out_img_gt')  # cyan-blue
                            debugger.add_arrow(
                                [(dets_gt[i][k][0] + dets_gt[i][k][2]) / 2,
                                 (dets_gt[i][k][1] + dets_gt[i][k][3]) / 2, ],
                                [dets_gt[i][k][25], dets_gt[i][k][26]],
                                img_id='pre_hm')  # cyan-blue

                    if 'tracking_hp' in opt.heads:

                        for idx in range(8):

                            # tracking_hp_mask == 0 then continue
                            if dets_gt[i][k][44 + idx * 2] == 0 or dets_gt[i][k][44 + idx * 2 + 1] == 0:
                                continue

                            debugger.add_arrow(
                                dets_gt[i][k][5 + idx * 2:5 + idx * 2 + 2],
                                dets_gt[i][k][28 + idx * 2:28 + idx * 2 + 2],
                                img_id='out_img_gt', c=(0, 255, 0))  # green
                            debugger.add_arrow(
                                dets_gt[i][k][5 + idx * 2:5 + idx * 2 + 2],
                                dets_gt[i][k][28 + idx * 2:28 + idx * 2 + 2],
                                img_id='pre_hmhp', c=(0, 255, 0))  # green
            # Blended
            debugger.add_img(img, img_id='out_pred_gt_blend')
            for k in range(len(dets['scores'][i])):
                if dets['scores'][i][k][0] > opt.center_thresh:
                    debugger.add_coco_hp(dets['kps'][i][k], img_id='out_pred_gt_blend')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_hp(dets_gt[i, k, 5:21], img_id='out_pred_gt_blend', pred_flag='gt')

            if opt.hm_hp:
                gt = debugger.gen_colormap_hp(batch['hm_hp'][i][choice_list[i]].detach().cpu().numpy())
                debugger.add_blend_img(img, gt, 'out_hmhp_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            elif opt.debug == 5:  # return result, wait for further processing
                pass
            else:
                debugger.show_all_imgs(pause=True)

        return debugger.imgs
