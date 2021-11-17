# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

"""
Save default parameters for the evaluation process.
"""

import argparse


class eval_opts(object):
    def __init__(self):
        # Param setting for opt_eval
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            '--report_file',
            default='report.txt',
            help="report file path"
        )
        self.parser.add_argument(
            '--outf',
            default='debug',
            help="folder path to save the debug files"
        )
        self.parser.add_argument(
            '--reportf',
            default='report',
            help="folder path to save the report files"
        )
        self.parser.add_argument(
            '--eval_max_num',
            default=100,
            type=int,
            help="the number of the test samples (1000 is a fair play)"
        )
        self.parser.add_argument(
            '--eval_resolution_ratio',
            default=2.4,
            type=float,
            help="ratio change from the original resolution (by default 1920*1440-> 800*600)"
        )
        self.parser.add_argument(
            '--eval_confidence_thresh',
            default=0.3,
            type=float,
            help="threshold for the confidence level of the detector"
        )
        self.parser.add_argument(
            '--eval_num_symmetry',
            default='1',
            type=int,
            help="number of symmetry, e.g., 180/360"
        )
        self.parser.add_argument(
            '--eval_gt_scale',
            action='store_true',
            help="apply the ground truth scale for pnp instead of the predicted one"
        )
        self.parser.add_argument(
            '--eval_MobilePose_postprocessing',
            action='store_true',
            help="apply the postprossing part used in MobilePose to lift 2D to 3D"
        )

        self.parser.add_argument(
            '--eval_kalman',
            action='store_true',
            help="apply kalman filter"
        )
        self.parser.add_argument(
            '--eval_scale_pool',
            action='store_true',
            help="enable scale pool"
        )
        self.parser.add_argument(
            '--eval_pre_hm',
            action='store_true',
            help="enable previous heatmap"
        )
        self.parser.add_argument(
            '--eval_pre_hm_hp',
            action='store_true',
            help="enable previous keypoint heatmap"
        )
        self.parser.add_argument(
            '--eval_gt_pre_hm_hmhp_first',
            action='store_true',
            help="given ground truth heatmaps only at first frame"
        )
        self.parser.add_argument(
            '--eval_refined_Kalman',
            action='store_true',
            help="enable kalman filter for CenterPose"
        )
        self.parser.add_argument(
            '--eval_use_absolute_scale',
            action='store_true',
            help="use the absolute scale"
        )
        self.parser.add_argument(
            '--eval_debug_save_thresh',
            default=1.0,
            type=float,
            help="the threshold of iou for saving the debug files"
        )
        self.parser.add_argument(
            '--eval_mug_symmetric',
            action='store_true',
            help="evaluate mug instances by a symmetric version"
        )
        self.parser.add_argument(
            '--eval_rep_mode',
            default=1,
            type=int,
            help="0: 8 selective representation,"
                 "1: 16 all representation"
                 "2: Sampling representation"
                 "3: Displacement only representation"
                 "4: Heatmap only representation"
        )

        self.parser.add_argument(
            '--eval_R',
            default=20,
            type=int,
            help="a fixed value for the uncertain of velocity in the Kalman Filter"
        )

        self.parser.add_argument(
            '--eval_c',
            default='bike',
            help="the category to be evaluated"
        )
        self.parser.add_argument(
            '--eval_arch',
            default='dla_34',
            help="the arch to be evaluated"
        )

        # Only for CenterPoseTrack
        self.parser.add_argument(
            '--eval_tracking_task',
            action='store_true',
            help="enable tracking task"
        )

        # Only for CenterPoseTrack
        self.parser.add_argument(
            '--eval_consistency_local_window',
            default=5,
            type=int,
            help="the length of the evaluated window for consistency score"
        )

        # Only for CenterPoseTrack
        self.parser.add_argument(
            '--eval_add_noise',
            action='store_true',
            help="add some gaussian noise to ground truth"
        )
        self.parser.add_argument(
            '--eval_noise_translation',
            default=0.03,
            type=float,
            help="the gaussian noise for translation"
        )
        self.parser.add_argument(
            '--eval_noise_rot',
            default=5,
            type=float,
            help="the gaussian noise for rotation (degree)"
        )
        self.parser.add_argument(
            '--eval_noise_scale',
            default=20,
            type=float,
            help="the gaussian noise for scale (percent)"
        )

        self.parser.add_argument(
            '--eval_CenterPose_initialization',
            action='store_true',
            help="use the CenterPose output as the ground truth"
        )

        self.parser.add_argument(
            '--eval_empty_pre_hm',
            action='store_true',
            help="only use empty heatmap"
        )

        # Debug related

        # Only for CenterPoseTrack
        self.parser.add_argument(
            '--eval_hard_case_list',
            default=[],
        )
        self.parser.add_argument(
            '--eval_hard_case',
            default=0,
            help="#0 Run on all cases"
                 "#1 Run on a subset of cases from hard_case.json"
                 "#2 Run on a subset of cases from eval_hard_case_list "
        )

        # Only for CenterPose
        self.parser.add_argument(
            '--eval_subset',
            action='store_true',
            help="only evaluate videos in a subset from selected_frames.json"
        )

        # Only for CenterPoseTrack
        self.parser.add_argument(
            '--eval_continue',
            action='store_true',
            help="skip the videos which have been evaluated"
        )

        self.parser.add_argument(
            '--eval_fake_output',
            action='store_true',
            help="using the prediction from the previous frame for the current frame, just for debug"
        )

        self.parser.add_argument(
            '--eval_skip',
            default=0,
            type=int,
            help="number of samples to skip"
        )
        self.parser.add_argument(
            '--eval_debug',
            action='store_true',
            help="save files for debug"
        )
        self.parser.add_argument(
            '--eval_debug_json',
            action='store_true',
            help="save json log"
        )
        self.parser.add_argument(
            '--eval_debug_clean',
            action='store_true',
            help="clean debug folder"
        )
        self.parser.add_argument(
            '--eval_debug_display',
            action='store_true',
            help="display the debug file "
        )

        # ID related

        # Only for CenterPose for now
        self.parser.add_argument(
            '--eval_exp_id',
            default=6,
            type=int,
            help="the id for the experiment group"
        )

        self.parser.add_argument(
            '--eval_save_id',
            default=0,
            type=int,
            help="the id for the saving folder"
        )

        self.parser.add_argument(
            '--eval_weight_id',
            default=140,
            type=int,
            help="the id for the training weights"
        )
