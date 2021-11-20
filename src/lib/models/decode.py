# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _transpose_and_gather_feat
from lib.utils.gpfit import gaussian, moments, fitgaussian

import numpy as np


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

'''
# Slow for large number of categories
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
'''


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


# For training debug, Inference = False, while for demo, Inference = True
def object_pose_decode(
        heat, kps, wh=None, kps_displacement_std=None, obj_scale=None, obj_scale_uncertainty=None, reg=None, hm_hp=None,
        hp_offset=None, tracking=None, tracking_hp=None,
        opt=None, Inference=False):
    K = opt.K
    rep_mode = opt.rep_mode

    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2

    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)  # inds: index in a single heatmap

    kps = _transpose_and_gather_feat(kps, inds)  # 100*34
    kps = kps.view(batch, K, num_joints * 2)  # joint offset from the centroid loc
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)  # + centroid loc
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    if wh is not None:
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)

        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        if hm_hp is not None:

            if Inference == True:
                # Save a copy for future use
                hm_hp_copy = hm_hp.clone()

            hm_hp = _nms(hm_hp)
            thresh = 0.1
            kps = kps.view(batch, K, num_joints, 2).permute(
                0, 2, 1, 3).contiguous()  # b x J x K x 2

            mask_temp = torch.ones((batch, num_joints, K, 1)).to(kps.device)
            mask_temp = (mask_temp > 0).float().expand(batch, num_joints, K, 2)
            kps_displacement_mean = mask_temp * kps
            kps_displacement_mean = kps_displacement_mean.permute(0, 2, 1, 3).contiguous().view(
                batch, K, num_joints * 2)

            # Continue normal processing
            reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)  # b x J x K x K x 2
            hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
            if hp_offset is not None:
                hp_offset = _transpose_and_gather_feat(
                    hp_offset, hm_inds.view(batch, -1))
                hp_offset = hp_offset.view(batch, num_joints, K, 2)
                hm_xs = hm_xs + hp_offset[:, :, :, 0]
                hm_ys = hm_ys + hp_offset[:, :, :, 1]
            else:
                hm_xs = hm_xs + 0.5
                hm_ys = hm_ys + 0.5

            # Filter by thresh
            mask = (hm_score > thresh).float()
            hm_score = (1 - mask) * -1 + mask * hm_score  # -1 or hm_score
            hm_ys = (1 - mask) * (-10000) + mask * hm_ys  # -10000 or hm_ys
            hm_xs = (1 - mask) * (-10000) + mask * hm_xs

            # Find the nearest keypoint in the corresponding heatmap for each displacement representation
            hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
                2).expand(batch, num_joints, K, K, 2)
            dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)  # b x J x K x K
            min_dist, min_ind = dist.min(dim=3)  # b x J x K
            hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
            min_dist = min_dist.unsqueeze(-1)
            min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
                batch, num_joints, K, 1, 2)
            hm_kps = hm_kps.gather(3, min_ind)
            hm_kps = hm_kps.view(batch, num_joints, K, 2)

            l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                   (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
                   (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
            mask = (mask > 0).float().expand(batch, num_joints, K, 2)

            if rep_mode == 3:
                #  For displacement
                pass
            elif rep_mode == 4:
                kps = hm_kps
            else:
                kps = (1 - mask) * hm_kps + mask * kps

            kps = kps.permute(0, 2, 1, 3).contiguous().view(
                batch, K, num_joints * 2)

            if Inference == True:
                # Have to satisfy all the requirements: within an enlarged 2D bbox/
                # hm_score high enough/center_score high enough/not far away from the corresponding representation
                scores_copy = scores.unsqueeze(1).expand(batch, num_joints, K, 2)

                mask_2 = (hm_kps[..., 0:1] > 0.8 * l) + (hm_kps[..., 0:1] < 1.2 * r) + \
                         (hm_kps[..., 1:2] > 0.8 * t) + (hm_kps[..., 1:2] < 1.2 * b) + \
                         (hm_score > thresh) + (min_dist < (torch.max(b - t, r - l) * 0.5)) + \
                         (scores_copy > thresh)

                mask_2 = (mask_2 == 7).float().expand(batch, num_joints, K, 2)
                hm_kps_filtered = mask_2 * hm_kps + (1 - mask_2) * -10000

                hm_xs_filtered = hm_kps_filtered[:, :, :, 0].detach().cpu().numpy()
                hm_ys_filtered = hm_kps_filtered[:, :, :, 1].detach().cpu().numpy()

                # Todo: since the refinement idea is halted, we do not have to fit gaussian for all the modes
                if rep_mode == 0 or rep_mode == 3 or rep_mode == 4:
                    # Fake number, they are not used in these modes
                    kps_heatmap_mean = torch.ones([batch, K, num_joints * 2], dtype=torch.float32) * -10000
                    kps_heatmap_std = torch.ones([batch, K, num_joints * 2], dtype=torch.float32) * -10000
                    kps_heatmap_height = torch.ones([batch, K, num_joints], dtype=torch.float32) * -10000

                elif rep_mode == 1 or rep_mode == 2:
                    # Fit a 2D gaussian distribution on the heatmap
                    # Save a copy for further processing
                    kps_heatmap_mean = torch.ones([batch, K, num_joints * 2], dtype=torch.float32) * -10000
                    kps_heatmap_std = torch.ones([batch, K, num_joints * 2], dtype=torch.float32) * -10000
                    kps_heatmap_height = torch.ones([batch, K, num_joints], dtype=torch.float32) * -10000

                    # Need optimization
                    for idx_batch in range(batch):
                        for idx_joint in range(num_joints):
                            data = hm_hp_copy[idx_batch][idx_joint].detach().cpu().numpy()
                            for idx_K in range(K):
                                if hm_xs_filtered[idx_batch][idx_joint][idx_K] == -10000 or \
                                        hm_ys_filtered[idx_batch][idx_joint][idx_K] == -10000:
                                    continue
                                else:

                                    win = 11
                                    ran = win // 2

                                    # For the tracking task, both rep_mode 1 and 2 needs this step
                                    if opt.tracking_task or opt.refined_Kalman or rep_mode == 2:

                                        data_enlarged = np.zeros((data.shape[0] + 2 * ran, data.shape[1] + 2 * ran))
                                        data_enlarged[ran:data.shape[0] + ran, ran:data.shape[1] + ran] = data
                                        weights = data_enlarged[int(hm_ys_filtered[idx_batch][idx_joint][idx_K]):
                                                                int(hm_ys_filtered[idx_batch][idx_joint][
                                                                        idx_K] + 2 * ran + 1),
                                                  int(hm_xs_filtered[idx_batch][idx_joint][idx_K]):
                                                  int(hm_xs_filtered[idx_batch][idx_joint][idx_K] + 2 * ran + 1)
                                                  ]

                                        params = fitgaussian(weights)

                                        # mu will be slightly different from ran
                                        height, mu_x, mu_y, std_x, std_y = params
                                    elif rep_mode == 1:

                                        # For fair comparison, do not use fitted gaussian for correction
                                        mu_x = ran
                                        mu_y = ran
                                        height = data[int(hm_ys_filtered[idx_batch][idx_joint][idx_K]),
                                                      int(hm_xs_filtered[idx_batch][idx_joint][idx_K])]
                                        std_x = 1  # Just used as a mark
                                        std_y = 1  # Just used as a mark

                                    kps_heatmap_mean[idx_batch][idx_K][idx_joint * 2:idx_joint * 2 + 2] = \
                                        torch.FloatTensor([hm_xs_filtered[idx_batch][idx_joint][idx_K] + mu_x - ran,
                                                           hm_ys_filtered[idx_batch][idx_joint][idx_K] + mu_y - ran])
                                    kps_heatmap_std[idx_batch][idx_K][idx_joint * 2:idx_joint * 2 + 2] = \
                                        torch.FloatTensor([std_x, std_y])
                                    kps_heatmap_height[idx_batch][idx_K][idx_joint] = torch.from_numpy(np.array(height))

                kps_heatmap_mean = kps_heatmap_mean.to(kps_displacement_mean.device)
                kps_heatmap_std = kps_heatmap_std.to(kps_displacement_mean.device)
                kps_heatmap_height = kps_heatmap_height.to(kps_displacement_mean.device)

    else:

        if hm_hp is not None:
            hm_hp = _nms(hm_hp)
            thresh = 0.1
            kps = kps.view(batch, K, num_joints, 2).permute(
                0, 2, 1, 3).contiguous()  # b x J x K x 2
            reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)  # b x J x K x K x 2
            hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
            if hp_offset is not None:
                hp_offset = _transpose_and_gather_feat(
                    hp_offset, hm_inds.view(batch, -1))
                hp_offset = hp_offset.view(batch, num_joints, K, 2)
                hm_xs = hm_xs + hp_offset[:, :, :, 0]
                hm_ys = hm_ys + hp_offset[:, :, :, 1]
            else:
                hm_xs = hm_xs + 0.5
                hm_ys = hm_ys + 0.5

            mask = (hm_score > thresh).float()
            hm_score = (1 - mask) * -1 + mask * hm_score  # -1 or hm_score
            hm_ys = (1 - mask) * (-10000) + mask * hm_ys  # -10000 or hm_ys
            hm_xs = (1 - mask) * (-10000) + mask * hm_xs
            hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
                2).expand(batch, num_joints, K, K, 2)
            dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)  # b x J x K x K
            min_dist, min_ind = dist.min(dim=3)  # b x J x K
            hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1

            min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
                batch, num_joints, K, 1, 2)
            hm_kps = hm_kps.gather(3, min_ind)
            hm_kps = hm_kps.view(batch, num_joints, K, 2)

            mask = (hm_score < thresh)  # Not valid
            mask = (mask > 0).float().expand(batch, num_joints, K, 2)
            kps = (1 - mask) * hm_kps + mask * kps
            kps = kps.permute(0, 2, 1, 3).contiguous().view(
                batch, K, num_joints * 2)

        # Actually, it makes no sense here, just need something to fill the tensor
        bboxes = torch.cat([xs,
                            ys,
                            xs,
                            ys], dim=2)

    if kps_displacement_std is not None:
        kps_displacement_std = _transpose_and_gather_feat(kps_displacement_std, inds)

        # Since we pred log(var) while saving std, we need to convert it first
        kps_displacement_std = torch.sqrt(torch.exp(kps_displacement_std))
        kps_displacement_std = kps_displacement_std * opt.balance_coefficient[opt.c]
        kps_displacement_std = kps_displacement_std.view(batch, K, num_joints * 2)  # joint offset from the centroid loc
    else:
        kps_displacement_std = torch.zeros([batch, K, num_joints * 2], dtype=torch.float32)
        kps_displacement_std = kps_displacement_std.to(scores.device)

    if obj_scale is not None:
        obj_scale = _transpose_and_gather_feat(obj_scale, inds)
        obj_scale = obj_scale.view(batch, K, 3)
    else:
        obj_scale = torch.zeros([batch, K, 3], dtype=torch.float32)
        obj_scale = obj_scale.to(scores.device)

    if obj_scale_uncertainty is not None:
        obj_scale_uncertainty = _transpose_and_gather_feat(obj_scale_uncertainty, inds)

        # We predict log(var) while save std, so we need to convert it first
        obj_scale_uncertainty = torch.sqrt(torch.exp(obj_scale_uncertainty))

        obj_scale_uncertainty = obj_scale_uncertainty.view(batch, K, 3)
    else:
        obj_scale_uncertainty = torch.zeros([batch, K, 3], dtype=torch.float32)
        obj_scale_uncertainty = obj_scale_uncertainty.to(scores.device)

    if tracking is not None:
        tracking = _transpose_and_gather_feat(tracking, inds)
        tracking = tracking.view(batch, K, 2)
    else:
        tracking = torch.zeros([batch, K, 2], dtype=torch.float32)
        tracking = tracking.to(scores.device)

    if tracking_hp is not None:
        tracking_hp = _transpose_and_gather_feat(tracking_hp, inds)
        tracking_hp = tracking_hp.view(batch, K, num_joints * 2)  # joint offset from the centroid loc
    else:
        tracking_hp = torch.zeros([batch, K, num_joints * 2], dtype=torch.float32)
        tracking_hp = tracking_hp.to(scores.device)

    if Inference == True:
        detections = {'bboxes': bboxes,
                      'scores': scores,
                      'kps': kps,
                      'clses': clses,
                      'obj_scale': obj_scale,
                      'obj_scale_uncertainty': obj_scale_uncertainty,
                      'tracking': tracking,
                      'tracking_hp': tracking_hp,
                      'kps_displacement_mean': kps_displacement_mean,
                      'kps_displacement_std': kps_displacement_std,
                      'kps_heatmap_mean': kps_heatmap_mean,
                      'kps_heatmap_std': kps_heatmap_std,
                      'kps_heatmap_height': kps_heatmap_height,
                      }
    else:
        detections = {'bboxes': bboxes,
                      'scores': scores,
                      'kps': kps,
                      'clses': clses,
                      'obj_scale': obj_scale,
                      'obj_scale_uncertainty': obj_scale_uncertainty,
                      'tracking': tracking,
                      'tracking_hp': tracking_hp,
                      'kps_displacement_mean': kps_displacement_mean,
                      'kps_displacement_std': kps_displacement_std,
                      }

    return detections
