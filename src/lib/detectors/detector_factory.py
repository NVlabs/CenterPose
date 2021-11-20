from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .object_pose import ObjectPoseDetector

detector_factory = {
    'object_pose': ObjectPoseDetector,
}
