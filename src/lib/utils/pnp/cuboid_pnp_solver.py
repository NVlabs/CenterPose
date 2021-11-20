# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

import cv2
import numpy as np
from .cuboid_objectron import CuboidVertexType
from pyrr import Quaternion
from scipy.spatial.transform import Rotation as R
import sklearn


class CuboidPNPSolver(object):
    """
    This class is used to find the 6-DoF pose of a cuboid given its projected vertices.

    Runs perspective-n-point (PNP) algorithm.
    """

    # Class variables
    cv2version = cv2.__version__.split('.')
    cv2majorversion = int(cv2version[0])

    def __init__(self, object_name="", scaling_factor=1,
                 camera_intrinsic_matrix=None,
                 cuboid3d=None,
                 dist_coeffs=np.zeros((4, 1)),
                 min_required_points=4
                 ):

        self.object_name = object_name
        self.min_required_points = max(4, min_required_points)
        self.scaling_factor = scaling_factor

        if (not camera_intrinsic_matrix is None):
            self._camera_intrinsic_matrix = camera_intrinsic_matrix
        else:
            self._camera_intrinsic_matrix = np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ])
        self._cuboid3d = cuboid3d

        self._dist_coeffs = dist_coeffs

    def set_camera_intrinsic_matrix(self, new_intrinsic_matrix):
        '''Sets the camera intrinsic matrix'''
        self._camera_intrinsic_matrix = new_intrinsic_matrix

    def set_dist_coeffs(self, dist_coeffs):
        '''Sets the camera intrinsic matrix'''
        self._dist_coeffs = dist_coeffs

    def __check_pnp_result(self,
                           points,
                           projected_points,
                           fail_if_projected_diff_exceeds,
                           fail_if_projected_value_exceeds):
        """
        Check whether the output of PNP seems reasonable
        Inputs:
        - cuboid2d_points:  list of XY tuples
        - projected points:  np.ndarray of np.ndarrays
        """

        p1 = points
        p2 = projected_points.tolist()
        assert len(p1) == len(p2)

        # Compute max Euclidean 2D distance b/w points and projected points
        max_euclidean_dist = 0
        for i in range(len(p1)):
            if p1[i] is not None:
                dist = np.linalg.norm(np.array(p1[i]) - np.array(p2[i]))
                if dist > max_euclidean_dist:
                    max_euclidean_dist = dist

        # Compute max projected absolute value
        max_abs_value = 0
        for i in range(len(p2)):
            assert len(p2[i]) == 2
            for val in p2[i]:
                if val > max_abs_value:
                    max_abs_value = val

        # Return success (true) or failure (false)
        return max_euclidean_dist <= fail_if_projected_diff_exceeds \
               and max_abs_value <= fail_if_projected_value_exceeds

    def solve_pnp(self,
                  cuboid2d_points,
                  pnp_algorithm=None,
                  OPENCV_RETURN = False,
                  fail_if_projected_diff_exceeds=250,
                  fail_if_projected_value_exceeds=1e5,
                  verbose = False
                  ):
        """
        Detects the rotation and traslation 
        of a cuboid object from its vertexes' 
        2D location in the image

        Inputs:
        - cuboid2d_points:  list of XY tuples
          ...

        Outputs:
        - location in 3D
        - pose in 3D (as quaternion)
        - projected points:  np.ndarray of np.ndarrays

        """

        # Fallback to default PNP algorithm base on OpenCV version
        if pnp_algorithm is None:
            if CuboidPNPSolver.cv2majorversion == 2:
                pnp_algorithm = cv2.CV_ITERATIVE
            elif CuboidPNPSolver.cv2majorversion == 3:
                pnp_algorithm = cv2.SOLVEPNP_ITERATIVE
                # Alternative algorithms:
                # pnp_algorithm = SOLVE_PNP_P3P  
                # pnp_algorithm = SOLVE_PNP_EPNP    
            else:
                # pnp_algorithm = cv2.SOLVEPNP_EPNP
                pnp_algorithm = cv2.SOLVEPNP_ITERATIVE

        location = None
        quaternion = None

        location_new = None
        quaternion_new = None
        reprojectionError = None

        projected_points = cuboid2d_points

        cuboid3d_points = np.array(self._cuboid3d.get_vertices())
        obj_2d_points = []
        obj_3d_points = []

        # 8*n points
        for i in range(len(cuboid2d_points)):
            check_point_2d = cuboid2d_points[i]
            # Ignore invalid points
            if (check_point_2d is None or check_point_2d[0] < -5000 or check_point_2d[1] < -5000):
                continue
            obj_2d_points.append(check_point_2d)
            obj_3d_points.append(
                cuboid3d_points[int(i // (len(cuboid2d_points) / CuboidVertexType.TotalCornerVertexCount))])

        obj_2d_points = np.array(obj_2d_points, dtype=float)
        obj_3d_points = np.array(obj_3d_points, dtype=float)

        valid_point_count = len(obj_2d_points)

        # Can only do PNP if we have more than 3 valid points
        is_points_valid = valid_point_count >= self.min_required_points

        if is_points_valid:

            # Heatmap representation may have less than 6 points, in which case we have to use another pnp algorithm
            if valid_point_count < 6:
                pnp_algorithm = cv2.SOLVEPNP_EPNP
            # Usually, we use this one
            ret, rvec, tvec, reprojectionError = cv2.solvePnPGeneric(
                obj_3d_points,
                obj_2d_points,
                self._camera_intrinsic_matrix,
                self._dist_coeffs,
                flags=pnp_algorithm
            )

            if ret:

                rvec = np.array(rvec[0])
                tvec = np.array(tvec[0])
                reprojectionError = reprojectionError.flatten()[0]

                # Convert OpenCV coordinate system to OpenGL coordinate system
                transformation = np.identity(4)
                r = R.from_rotvec(rvec.reshape(1, 3))
                transformation[:3, :3] = r.as_matrix()
                transformation[:3, 3] = tvec.reshape(1, 3)
                M = np.zeros((4, 4))
                M[0, 1] = 1
                M[1, 0] = 1
                M[3, 3] = 1
                M[2, 2] = -1
                transformation = np.matmul(M, transformation)

                rvec_new = R.from_matrix(transformation[:3, :3]).as_rotvec()
                tvec_new = transformation[:3, 3]

                # OpenGL result, to be compared against GT
                location_new = list(x for x in tvec_new)
                quaternion_new = self.convert_rvec_to_quaternion(rvec_new)

                # OpenCV result
                location = list(x[0] for x in tvec)
                quaternion = self.convert_rvec_to_quaternion(rvec)

                # Still use OpenCV way to project 3D points
                projected_points, _ = cv2.projectPoints(cuboid3d_points, rvec, tvec, self._camera_intrinsic_matrix,
                                                        self._dist_coeffs)
                projected_points = np.squeeze(projected_points)

                # Todo: currently, we assume pnp fails if z<0
                x, y, z = location
                if z < 0:
                    # # Get the opposite location
                    # location = [-x, -y, -z]
                    #
                    # # Change the rotation by 180 degree
                    # rotate_angle = np.pi
                    # rotate_quaternion = Quaternion.from_axis_rotation(location, rotate_angle)
                    # quaternion = rotate_quaternion.cross(quaternion)
                    location = None
                    quaternion = None
                    location_new = None
                    quaternion_new = None

                    if verbose:
                        print("PNP solution is behind the camera (Z < 0) => Fail")
                else:
                    if verbose:
                        print("solvePNP found good results - location: {} - rotation: {} !!!".format(location, quaternion))
            else:
                if verbose:
                    print('Error:  solvePnP return false ****************************************')
        else:
            if verbose:
                print("Need at least 4 valid points in order to run PNP. Currently: {}".format(valid_point_count))

        if OPENCV_RETURN:
            # Return OpenCV result for demo
            return location, quaternion, projected_points, reprojectionError
        else:
            # Return OpenGL result for eval
            return location_new, quaternion_new, projected_points, reprojectionError

    def convert_rvec_to_quaternion(self, rvec):
        '''Convert rvec (which is log quaternion) to quaternion'''
        theta = np.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2])  # in radians
        raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

        # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
        return Quaternion.from_axis_rotation(raxis, theta)

        # Alternatively: pyquaternion
        # return Quaternion(axis=raxis, radians=theta)  # uses OpenCV's Quaternion (order is WXYZ)
