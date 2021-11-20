# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

import numpy as np
from .cuboid_objectron import Cuboid3d
from .cuboid_pnp_solver import CuboidPNPSolver
from scipy.spatial.transform import Rotation as R


def pnp_shell(opt, meta, bbox, points_filtered, scale, OPENCV_RETURN = False):
    cuboid3d = Cuboid3d(1 * np.array(scale) / scale[1])

    pnp_solver = \
        CuboidPNPSolver(
            opt.c,
            cuboid3d=cuboid3d
        )
    pnp_solver.set_camera_intrinsic_matrix(meta['camera_matrix'])

    location, quaternion, projected_points, reprojectionError = pnp_solver.solve_pnp(
        points_filtered, OPENCV_RETURN=OPENCV_RETURN)  # N * 2

    if location is not None:

        # Save to results
        bbox['location'] = location
        bbox['quaternion_xyzw'] = quaternion
        bbox['projected_cuboid'] = projected_points  # Just for debug # not normalized 16

        ori = R.from_quat(quaternion).as_matrix()
        pose_pred = np.identity(4)
        pose_pred[:3, :3] = ori
        pose_pred[:3, 3] = location
        point_3d_obj = cuboid3d.get_vertices()

        point_3d_cam = pose_pred @ np.hstack(
            (np.array(point_3d_obj), np.ones((np.array(point_3d_obj).shape[0], 1)))).T
        point_3d_cam = point_3d_cam[:3, :].T  # 8 * 3

        # Add the centroid
        point_3d_cam = np.insert(point_3d_cam, 0, np.mean(point_3d_cam, axis=0), axis=0)

        bbox['kps_3d_cam'] = point_3d_cam  # Just for debug

        # Add the center
        projected_points = np.insert(projected_points, 0, np.mean(projected_points, axis=0), axis=0)

        # Normalization
        projected_points[:, 0] = projected_points[:, 0] / meta['width']
        projected_points[:, 1] = projected_points[:, 1] / meta['height']

        # Debug only
        bbox['kps_pnp'] = projected_points # Normalized 9*2

        # Todo: Sometimes, the label in the dataset is missing if many keypoints are not visible,
        #  e.g., a camera on the corner

        if opt.c in ['bike', 'laptop', 'shoe']:
            pass
        else:
            if opt.c in ['book', 'chair', 'cereal_box']:
                num_not_visible_thresh = 6
            if opt.c in ['camera', 'bottle', 'cup']:
                num_not_visible_thresh = 3

            num_not_visible = 0
            for i in projected_points:
                if i[0] < 0 or i[0] > 1 or i[1] < 0 or i[1] > 1:
                    num_not_visible = num_not_visible + 1
            if num_not_visible >= num_not_visible_thresh:
                return

        def is_visible(point):
            """Determines if a 2D point is visible."""
            return point[0] > 0 and point[0] < 1 and point[1] > 0 and point[1] < 1

        if not is_visible(projected_points[0]):
            return

        points = [(x[0], x[1]) for x in np.array(bbox['kps']).reshape(-1, 2)]

        # Add the center
        points_ori = np.insert(points, 0, np.mean(points, axis=0), axis=0)

        # Normalization
        points_ori[:, 0] = points_ori[:, 0] / meta['width']
        points_ori[:, 1] = points_ori[:, 1] / meta['height']

        # keypoint_2d_pnp, keypoint_3d, predicted_scale, keypoint_2d_ori, result_ori for debug
        return projected_points, point_3d_cam, np.array(bbox['obj_scale']), points_ori, bbox

    return
