# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from numba import jit
import copy
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from lib.utils.pnp.cuboid_pnp_shell import pnp_shell


class Tracker(object):
    def __init__(self, opt):
        self.opt = opt
        self.meta = None  # Have not been inited yet
        self.reset()

    # Mainly to read external input
    def init_track(self, meta):

        # Not all the info will be updated along the time, e.g., id
        self.meta = meta

        if 'pre_dets' in self.meta:
            dets = self.meta['pre_dets']
            self.reset()
        else:
            dets = []
        for item in dets:
            if item['score'] > self.opt.new_thresh:
                self.id_count += 1
                # active is not used
                item['active'] = 1
                item['age'] = 1
                item['tracking_id'] = self.id_count
                if not ('ct' in item):
                    bbox = item['bbox']
                    item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

                if self.opt.kalman == True:
                    # Init Kalman Filter
                    item['kf'] = self.init_kf(item)
                if self.opt.scale_pool == True:
                    # Init scale pool with current prediction
                    item['scale_pool'] = [(item['obj_scale'], item['obj_scale_uncertainty'])]

                self.tracks.append(item)

    def reset(self):
        self.id_count = 0
        self.tracks = []

    def init_kf(self, det):
        # 4 (x,y,v_x,v_y) * 8 (8 keypoints), all of them are observable
        kf = KalmanFilter(dim_x=32, dim_z=32)
        kf.H = np.eye(32)
        for i in range(8):
            kf.F[4 * i, 4 * i + 2] = 1  # v_x
            kf.F[4 * i + 1, 4 * i + 2 + 1] = 1  # v_y

            kf.R[4 * i, 4 * i] *= det['kps_fusion_std'][i * 2] ** 2  # x
            kf.R[4 * i + 1, 4 * i + 1] *= det['kps_fusion_std'][i * 2 + 1] ** 2  # y

            # Note: Fixed value for v_x,v_y, need tuning
            kf.R[4 * i + 2:4 * i + 4, 4 * i + 2:4 * i + 4] *= self.opt.R

            # Initial P
            kf.P = kf.R

        # Read x,y,v_x, v_y, where -result['tracking_hp'] is from current - previous
        for i in range(8):
            kf.x[4 * i:4 * i + 4] = np.array([det['kps_fusion_mean'][2 * i],
                                              det['kps_fusion_mean'][2 * i + 1],
                                              -det['tracking_hp'][2 * i],
                                              -det['tracking_hp'][2 * i + 1]]).reshape(-1, 1)

        return kf

    def update_kf(self, det):

        # Read x,y,v_x, v_y, where -result['tracking_hp'] is from current - previous
        z = np.zeros(32)
        R = np.eye(32)
        for i in range(8):
            z[4 * i:4 * i + 4] = [det['kps_fusion_mean'][2 * i],
                                  det['kps_fusion_mean'][2 * i + 1],
                                  -det['tracking_hp'][2 * i],
                                  -det['tracking_hp'][2 * i + 1]]
            R[4 * i, 4 * i] *= det['kps_fusion_std'][i * 2] ** 2  # x
            R[4 * i + 1, 4 * i + 1] *= det['kps_fusion_std'][i * 2 + 1] ** 2  # y
            # Fixed value for v_x,v_y, need tuning
            R[4 * i + 2:4 * i + 4, 4 * i + 2:4 * i + 4] *= self.opt.R

        det['kf'].update(z, R=R)

    def update_scale_pool(self, det):

        # Bayesian fusion
        std = np.zeros(3)
        mean = np.zeros(3)
        for scale_data in det['scale_pool']:
            std_sample = np.array(scale_data[1]) ** -2
            std += std_sample
            mean += std_sample * np.array(scale_data[0])
        std = std ** -0.5
        mean *= (std ** 2)

        return mean, std

    def step(self, dets, boxes=[]):
        # Step 0:
        # Convert det
        if self.opt.use_pnp == True:
            if boxes:
                dets = []
                for idx, box in enumerate(boxes):
                    det = boxes[idx][4]
                    det['kps_pnp'] = boxes[idx][0]  # 9 keypoints normalized
                    det['kps_3d_cam'] = boxes[idx][1]
                    det['kps_ori'] = boxes[idx][3]  # 9 keypoints normalized from kps
                    dets.append(det)

        # Step 1:
        # Associate track and det
        N = len(dets)
        M = len(self.tracks)

        dets_center = np.array([det['ct'] + det['tracking'] for det in dets], np.float32)  # N x 2

        track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
                                (track['bbox'][3] - track['bbox'][1])) \
                               for track in self.tracks], np.float32)  # M
        track_cat = np.array([track['cls'] for track in self.tracks], np.int32)  # M

        item_size = np.array([((det['bbox'][2] - det['bbox'][0]) * \
                               (det['bbox'][3] - det['bbox'][1])) \
                              for det in dets], np.float32)  # N

        item_cat = np.array([det['cls'] for det in dets], np.int32)  # N

        tracks_center = np.array(
            [track['ct'] for track in self.tracks], np.float32)  # M x 2

        dist = (((tracks_center.reshape(1, -1, 2) - \
                  dets_center.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M

        invalid = ((dist > track_size.reshape(1, M)) + \
                   (dist > item_size.reshape(N, 1)) + \
                   (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
        dist = dist + invalid * 1e18

        if self.opt.hungarian:
            # item_score = np.array([det['score'] for det in dets], np.float32)  # N
            dist[dist > 1e18] = 1e18
            matched_indices = linear_assignment(dist)
        else:
            matched_indices = greedy_assignment(copy.deepcopy(dist))

        unmatched_dets = [d for d in range(dets_center.shape[0]) \
                          if not (d in matched_indices[:, 0])]
        unmatched_tracks = [d for d in range(tracks_center.shape[0]) \
                            if not (d in matched_indices[:, 1])]

        # Additional mandatory filter for hungarian algorithm
        if self.opt.hungarian:
            matches = []
            for m in matched_indices:
                if dist[m[0], m[1]] > 1e16:
                    unmatched_dets.append(m[0])
                    unmatched_tracks.append(m[1])
                else:
                    matches.append(m)
            matches = np.array(matches).reshape(-1, 2)
        else:
            matches = matched_indices

        # Step 2 Add matched ones & Update Kalman Filter:
        ret = []
        for m in matches:
            track = dets[m[0]]

            # Inherit some info from tracks
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']
            track['age'] = 1
            track['active'] = self.tracks[m[1]]['active'] + 1

            # Update kf with det
            if self.opt.kalman == True:
                track['kf'] = self.tracks[m[1]]['kf']
                track['kf'].predict()

                # Update kalman filter with the latest observation
                self.update_kf(track)
            if self.opt.scale_pool == True:
                track['scale_pool'] = self.tracks[m[1]]['scale_pool']
                track['scale_pool'].append((track['obj_scale'], track['obj_scale_uncertainty']))

            ret.append(track)

        # Step 3 Add unmatched ones:
        for i in unmatched_dets:
            track = dets[i]
            if track['score'] > self.opt.new_thresh:
                self.id_count += 1
                track['tracking_id'] = self.id_count
                track['age'] = 1
                track['active'] = 1

                if self.opt.kalman == True:
                    # Init Kalman Filter
                    track['kf'] = self.init_kf(track)
                if self.opt.scale_pool == True:
                    # Init scale pool with current prediction
                    track['scale_pool'] = [(track['obj_scale'], track['obj_scale_uncertainty'])]

                ret.append(track)

        # Step 4 Process instances in the unmatched record:
        for i in unmatched_tracks:
            track = self.tracks[i]
            if track['age'] < self.opt.max_age:
                track['age'] += 1
                track['active'] = 0

                # Todo: Update some modalities: bbox, ct, kps, kps_displacement_mean, kps_heatmap, tracking, tracking_hp
                # Right now, assume it does not move
                bbox = track['bbox']
                ct = track['ct']
                v = [0, 0]
                track['bbox'] = [
                    bbox[0] + v[0], bbox[1] + v[1],
                    bbox[2] + v[0], bbox[3] + v[1]]
                track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
                ret.append(track)

        if (self.opt.kalman == True or self.opt.scale_pool == True):

            if self.opt.use_pnp == True:
                boxes = []

            for idx, track in enumerate(ret):
                # Step 5 Get latest result from filters &:
                # Extract updated x,y state from Kalman Filter
                kps_mean_kf = track['kps']  # 8*2
                if self.opt.kalman == True:
                    ret[idx]['kps_mean_kf'] = np.array([track['kf'].x[4 * i:4 * i + 2] for i in range(8)])
                    kps_mean_kf = ret[idx]['kps_mean_kf']

                    # Todo: May apply some methods to filter some keypoints for pnp
                    kps_conf_kf = []
                    ret[idx]['kps_std_kf'] = []
                    for i in range(8):
                        ret[idx]['kps_std_kf'].append(np.sqrt(track['kf'].P[4 * i, 4 * i]))
                        ret[idx]['kps_std_kf'].append(np.sqrt(track['kf'].P[4 * i + 1, 4 * i + 1]))

                        std_combined = np.sqrt(track['kf'].P[4 * i, 4 * i] + track['kf'].P[4 * i + 1, 4 * i + 1])
                        kps_conf_kf.append(np.maximum(1 - np.exp(np.log(0.15) / (
                                self.opt.conf_border[self.opt.c][0] - self.opt.conf_border[self.opt.c][1])) ** (
                                                              std_combined - self.opt.conf_border[self.opt.c][1]),
                                                      0))
                        # Todo: According to kps_std_kf, may apply some filters?
                        if kps_conf_kf[i] < 0.15:
                            kps_mean_kf[i][0] = -10000
                            kps_mean_kf[i][1] = -10000

                scale_new = track['obj_scale']
                if self.opt.scale_pool == True:
                    # Get updated scale prediction
                    mean, std = self.update_scale_pool(track)
                    ret[idx]['obj_scale_kf'] = mean
                    ret[idx]['obj_scale_uncertainty_kf'] = std
                    scale_new = ret[idx]['obj_scale_kf']

                if self.opt.use_pnp == True:
                    # Step 6 Adopt pnp here to get new result, uncertainty coming from Kalman Filter

                    # Get new boxes & dict_out
                    ret_pnp = pnp_shell(self.opt, self.meta, track, kps_mean_kf, scale_new, OPENCV_RETURN=self.opt.show_axes)

                    # if ret_pnp is not None:
                    if ret_pnp is not None:

                        conf_avg = np.sum(kps_conf_kf) / 8
                        if conf_avg > 0.25:
                            boxes.append(ret_pnp)
                        # Todo: Sometimes, ret_pnp has no return
                        # Update some info, only used for rendering
                        ret[idx]['kps_pnp_kf'] = ret_pnp[0]  # 9 keypoints normalized (updated)
                        ret[idx]['kps_3d_cam_kf'] = ret_pnp[1]  # 9 keypoints (updated)
                        ret[idx]['kps_ori_kf'] = ret_pnp[3]  # 9 keypoints normalized from kps

            # Update tracks for next frame
            self.tracks = ret
            # Update dets, which is for plot; Update boxes here, which is for evaluation
            return ret, boxes
        else:
            # Update tracks for next frame
            self.tracks = ret
            # Update dets, which is for plot; Update boxes here, which is for evaluation
            return ret, boxes


def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)
