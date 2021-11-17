"""Parser for Objectron tf.examples."""

import math
import numpy as np
import tensorflow as tf
import cv2

import objectron.schema.features as features

import re
from scipy.spatial.transform import Rotation as R

# Label names.
LABEL_INSTANCE = '2d_instance'
LABEL_INSTANCE_3D = '3d_instance'
LABEL_INSTANCE_SCALE = 'scale_instance'
LABEL_INSTANCE_Mo2c = 'Mo2c_instance'
LABEL_INSTANCE_MugFlag = 'MugFlag_instance'  # Only for cup category

LABEL_INSTANCE_Image_ID = 'image_id'

VISIBILITY = 'visibility'

# Read three label lists
cup_list = []
with open('label/cup/cup_list.txt', 'r') as fp:
    for l in fp:
        line_list = re.split(r'\t+', l.strip())
        cup_list.append(line_list)

mug_left_list = []
with open('label/cup/mug_left_list.txt', 'r') as fp:
    for l in fp:
        line_list = re.split(r'\t+', l.strip())
        mug_left_list.append(line_list)

mug_right_list = []
with open('label/cup/mug_right_list.txt', 'r') as fp:
    for l in fp:
        line_list = re.split(r'\t+', l.strip())
        mug_right_list.append(line_list)

chair_symmetric_list = []
with open('label/chair/symmetric_list.txt', 'r') as fp:
    for l in fp:
        line_list = re.split(r'\t+', l.strip())
        chair_symmetric_list.append(line_list)

swap_list = [[1, 6], [2, 5], [3, 8], [4, 7]]

# Transform matrix
M = np.identity(3)
M[0, 0] = -1
M[2, 2] = -1


class ObjectronParser(object):
    """Parser using NumPy."""

    def __init__(self, c, height=640, width=480):
        self._in_height, self._in_width = height, width
        self._vis_thresh = 0.1
        self.c = c  # category

    def get_image_and_label(self, serialized):
        """Gets image and its label from a serialized tf.Example.

        Args:
          serialized: A string of serialized tf.Example.

        Returns:
          A tuple of image and its label.
        """
        example = tf.train.Example.FromString(serialized)
        return self.parse_example(example)

    def parse_example_basic(self, example):
        """Parses image and label from a tf.Example proto.

           Args:
             example: A tf.Example proto.

           Returns:
             video_id and image_id.
       """
        fm = example.features.feature
        filename = fm[features.FEATURE_NAMES['IMAGE_FILENAME']].bytes_list.value[0].decode("utf-8")
        video_id = filename.replace('/', '_')
        image_id = np.asarray(fm[features.FEATURE_NAMES['IMAGE_ID']].int64_list.value)[0]

        return video_id, image_id

    def parse_example(self, example):
        """Parses image and label from a tf.Example proto.

        Args:
          example: A tf.Example proto.

        Returns:
          A tuple of image and its label.
        """
        fm = example.features.feature
        image = self.get_image(
            fm[features.FEATURE_NAMES['IMAGE_ENCODED']], shape=(self._in_width, self._in_height))

        filename = fm[features.FEATURE_NAMES['IMAGE_FILENAME']].bytes_list.value[0].decode("utf-8")
        filename = filename.replace('/', '_')
        # Todo: We do not need this process for our algorithm
        # image = image / 255.
        # image = self._normalize_image(image)

        image_id = np.asarray(fm[features.FEATURE_NAMES['IMAGE_ID']].int64_list.value)[0]

        label = {}
        visibilities = fm[features.FEATURE_NAMES['VISIBILITY']].float_list.value
        visibilities = np.asarray(visibilities)
        # label[VISIBILITY] = visibilities # A bug not reported yet from Official Objectron dataset
        index = visibilities > self._vis_thresh

        if features.FEATURE_NAMES['POINT_2D'] in fm:
            points_2d = fm[features.FEATURE_NAMES['POINT_2D']].float_list.value
            points_2d = np.asarray(points_2d).reshape((-1, 9, 3))[..., :2]
            # label[LABEL_INSTANCE] = points_2d[index]

        if features.FEATURE_NAMES['POINT_3D'] in fm:
            points_3d = fm[features.FEATURE_NAMES['POINT_3D']].float_list.value
            points_3d = np.asarray(points_3d).reshape((-1, 9, 3))
            # label[LABEL_INSTANCE_3D] = points_3d[index]

        if features.FEATURE_NAMES['OBJECT_SCALE'] in fm:
            obj_scale = fm[features.FEATURE_NAMES['OBJECT_SCALE']].float_list.value
            obj_scale = np.asarray(obj_scale).reshape((-1, 3))
            # label[LABEL_INSTANCE_SCALE] = obj_scale[index]

        if features.FEATURE_NAMES['OBJECT_TRANSLATION'] in fm \
                and features.FEATURE_NAMES['OBJECT_ORIENTATION'] in fm:
            obj_trans = fm[features.FEATURE_NAMES['OBJECT_TRANSLATION']].float_list.value
            obj_trans = np.asarray(obj_trans).reshape((-1, 3))

            obj_ori = fm[features.FEATURE_NAMES['OBJECT_ORIENTATION']].float_list.value
            obj_ori = np.asarray(obj_ori).reshape((-1, 3, 3))

            # obj_trans = obj_trans[index]
            # obj_ori = obj_ori[index]
            obj_transformation = []
            for trans, ori in zip(obj_trans, obj_ori):
                # M_o2c, different from the raw dataset, this data has been processed
                transformation = np.identity(4)
                transformation[:3, :3] = ori
                transformation[:3, 3] = trans
                obj_transformation.append(transformation)
            obj_transformation = np.array(obj_transformation)

        # Update the gt labels according to the additional labels
        if self.c == 'cup':
            mug_flag_list = []
            for object_id in range(len(index)):
                if [filename, str(object_id)] not in cup_list:
                    mug_flag = True
                else:
                    mug_flag = False

                mug_flag_list.append(mug_flag)

                if mug_flag == True:
                    # Update projected_keypoints, keypoints_3d and quaternion_xyzw(M_o2c)
                    if [filename, str(object_id)] in mug_left_list:

                        def swapPositions(list, pos1, pos2):
                            list[pos1], list[pos2] = np.copy(list[pos2]), np.copy(list[pos1])
                            return list

                        for swap_pair in swap_list:
                            swapPositions(points_2d[object_id], swap_pair[0], swap_pair[1])
                            swapPositions(points_3d[object_id], swap_pair[0], swap_pair[1])

                        M_old = obj_transformation[object_id][:3, :3]
                        M_new = np.linalg.inv(M) @ M_old
                        obj_transformation[object_id][:3, :3] = M_new

            label[LABEL_INSTANCE_MugFlag] = np.array(mug_flag_list)[index]

        label[LABEL_INSTANCE] = points_2d[index]
        label[LABEL_INSTANCE_3D] = points_3d[index]
        label[LABEL_INSTANCE_SCALE] = obj_scale[index]
        label[LABEL_INSTANCE_Mo2c] = obj_transformation[index]
        label[LABEL_INSTANCE_Image_ID] = image_id

        label[VISIBILITY] = visibilities[index]

        label['ORI_INDEX'] = np.argwhere(index).flatten()  # corresponding to the current order
        label['ORI_NUM_INSTANCE'] = len(index)
        return image, label, filename

    def parse_camera(self, example):
        """Parses camera from a tensorflow example."""
        fm = example.features.feature
        if features.FEATURE_NAMES['PROJECTION_MATRIX'] in fm:
            proj = fm[features.FEATURE_NAMES['PROJECTION_MATRIX']].float_list.value
            proj = np.asarray(proj).reshape((4, 4))
        else:
            proj = None

        if features.FEATURE_NAMES['VIEW_MATRIX'] in fm:
            view = fm[features.FEATURE_NAMES['VIEW_MATRIX']].float_list.value
            view = np.asarray(view).reshape((4, 4))
        else:
            view = None

        if features.FEATURE_NAMES['INTRINSIC_MATRIX'] in fm:
            intrinsic = fm[features.FEATURE_NAMES['INTRINSIC_MATRIX']].float_list.value
            intrinsic = np.asarray(intrinsic).reshape((3, 3))
        else:
            intrinsic = None

        return proj, view, intrinsic

    def parse_plane(self, example):
        """Parses plane from a tensorflow example."""
        fm = example.features.feature
        if features.FEATURE_NAMES['PLANE_CENTER'] in fm and features.FEATURE_NAMES['PLANE_NORMAL'] in fm:
            center = fm[features.FEATURE_NAMES['PLANE_CENTER']].float_list.value
            center = np.asarray(center)
            normal = fm[features.FEATURE_NAMES['PLANE_NORMAL']].float_list.value
            normal = np.asarray(normal)
            return center, normal
        else:
            return None

    def get_image(self, feature, shape=None):
        image = cv2.imdecode(
            np.asarray(bytearray(feature.bytes_list.value[0]), dtype=np.uint8),
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if shape is not None:
            image = cv2.resize(image, shape)
        return image

    def _normalize_image(self, image):
        """Normalizes pixels of an image from [0, 1] to [-1, 1]."""
        return image * 2. - 1.
