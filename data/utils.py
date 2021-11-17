# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

import random
import math 
import simplejson as json
from scipy.spatial.transform import Rotation as R
import numpy as np

import cv2
import os

from objectron.dataset import graphics
import matplotlib.pyplot as plt

import re

from bbox_collision_detection import CheckLineBox

# Read three label lists
cup_list=[]
with open('label/cup/cup_list.txt','r') as fp:
    for l in fp:
        line_list=re.split(r'\t+', l.strip())
        cup_list.append(line_list)

mug_left_list=[]
with open('label/cup/mug_left_list.txt','r') as fp:
    for l in fp:
        line_list=re.split(r'\t+', l.strip())
        mug_left_list.append(line_list)

mug_right_list=[]
with open('label/cup/mug_right_list.txt','r') as fp:
    for l in fp:
        line_list=re.split(r'\t+', l.strip())
        mug_right_list.append(line_list)


chair_symmetric_list=[]
with open('label/chair/symmetric_list.txt','r') as fp:
    for l in fp:
        line_list=re.split(r'\t+', l.strip())
        chair_symmetric_list.append(line_list)


swap_list=[[1,6],[2,5],[3,8],[4,7]]

# Transform matrix
M = np.identity(3)
M[0,0] = -1
M[2,2] = -1


def swapPositions(list, pos1, pos2):
    list[pos1],list[pos2] = np.copy(list[pos2]), np.copy(list[pos1])
    return list

def export_to_ndds_file(
    frame=None,
    filename = "tmp.json", #this has to include path as well
    sequence=None,
    frame_id=None,
    opt=None,
    video_filename=None
    ):
    
    try:
        if sequence is None or frame_id is None or frame is None:
            return 1

        """Grab an annotated frame from the sequence."""
        data = sequence.frame_annotations[frame_id]

        ## Camera information

        # Get the camera for the current frame. We will use the camera to bring
        # the object from the world coordinate to the current camera coordinate.

        # M_c2w
        camera = np.array(data.camera.transform).reshape(4, 4)
        cam_world_quaternion = R.from_matrix(camera[:3, :3]).as_quat()

        # Numpy array width (img height) <- data.camera.image_resolution_width
        height,width = data.camera.image_resolution_width,data.camera.image_resolution_height

        height=int(height/opt.resolution_ratio)
        width=int(width/opt.resolution_ratio)

        cam_intrinsics=np.array(data.camera.intrinsics).reshape(3, 3)
        cam_intrinsics[:2,:3]=cam_intrinsics[:2,:3]/opt.resolution_ratio
        cam_projection_matrix=np.array(data.camera.projection_matrix).reshape(4, 4)
        cam_view_matrix=np.array(data.camera.view_matrix).reshape(4, 4)
        
        dict_out = {
                        "camera_data" : {
                            "width" : width,
                            'height' : height, # image height
                            'camera_view_matrix':cam_view_matrix.tolist(),
                            'camera_projection_matrix':cam_projection_matrix.tolist(),
                            'location_world':  # M_c2w 
                            [
                                camera[0][3],
                                camera[1][3],
                                camera[2][3],
                            ],
                            'quaternion_world_xyzw':[ # M_c2w
                                cam_world_quaternion[0],
                                cam_world_quaternion[1],
                                cam_world_quaternion[2],
                                cam_world_quaternion[3],
                            ],
                            'intrinsics':{
                                # The original order is not correct
                                'fx':cam_intrinsics[1][1],
                                'fy':cam_intrinsics[0][0],
                                'cx':cam_intrinsics[1][2],
                                'cy':cam_intrinsics[0][2]

                            }
                        }, 
                        "objects" : [],
                        "AR_data":{
                            'plane_center':[data.plane_center[0],
                                            data.plane_center[1],
                                            data.plane_center[2]],
                            'plane_normal':[data.plane_normal[0],
                                            data.plane_normal[1],
                                            data.plane_normal[2]]
                        }
                    }

        # Object information
        object_id = 0
        object_keypoints_2d = []
        object_keypoints_3d = []
        object_rotations = []
        object_translations = []
        object_scale = []
        keypoint_size_list = []
        object_categories = []

        for obj in sequence.objects:
            rotation = np.array(obj.rotation).reshape(3, 3)
            translation = np.array(obj.translation)
            object_scale.append(np.array(obj.scale))

            if 0 in np.array(obj.scale):
                return 1

            # M_o2w
            transformation = np.identity(4)
            transformation[:3, :3] = rotation
            transformation[:3, 3] = translation

            # M_w2c * M_o2w -> M_o2c
            obj_cam = np.matmul(cam_view_matrix, transformation) 
            object_translations.append(obj_cam[:3, 3])

            obj_cam_quaternion = R.from_matrix(obj_cam[:3, :3]).as_quat()
            object_rotations.append(obj_cam_quaternion)
            # object_rotations.append(obj_cam[:3, :3]) #  The official tutorial has a bug here
            object_categories.append(obj.category)

        for annotations in data.annotations:
            num_keypoints = len(annotations.keypoints)
            keypoint_size_list.append(num_keypoints)
            for keypoint_id in range(num_keypoints):
                keypoint = annotations.keypoints[keypoint_id]
                object_keypoints_2d.append(
                    (keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
                object_keypoints_3d.append(
                    (keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
            object_id += 1


        keypoints_2d = np.split(object_keypoints_2d, np.array(np.cumsum(keypoint_size_list)))
        keypoints_2d = [points.reshape(-1, 3) for points in keypoints_2d]
        keypoints_2d = [np.multiply(keypoint, np.asarray([width, height, 1.], np.float32)).astype(int) for keypoint in keypoints_2d]

        keypoints_3d = np.split(object_keypoints_3d, np.array(np.cumsum(keypoint_size_list)))
        keypoints_3d = [points.reshape(-1, 3) for points in keypoints_3d]

        num_objects = len(keypoint_size_list)
        for object_id in range(num_objects):
            
            projected_keypoints=keypoints_2d[object_id][:,0:2]
            
            video_name=os.path.split(os.path.split(filename)[0])[1]

            if [video_name,str(object_id)] in opt.bug_list:
                continue
            
            # For cup (mug/cup)
            if 'cup' in filename:
                if [video_name,str(object_id)] not in cup_list:
                    mug_flag = True
                else:
                    mug_flag = False

                # # Debug only
                # projected_keypoints=keypoints_2d[object_id]

                # Not used in the paper
                if mug_flag == True:
                    # Update projected_keypoints, keypoints_3d and quaternion_xyzw(M_o2c)
                    if [video_name,str(object_id)] in mug_left_list:
                        for swap_pair in swap_list:
                            swapPositions(projected_keypoints, swap_pair[0], swap_pair[1])
                            swapPositions(keypoints_3d[object_id], swap_pair[0], swap_pair[1])

                        M_old = R.from_quat(object_rotations[object_id]).as_matrix()
                        M_new = np.linalg.inv(M) @ M_old
                        object_rotations[object_id] = R.from_matrix(M_new).as_quat()

                        M_o2c=np.identity(4)
                        M_o2c[:3,:3]=M_new
                        M_o2c[:3,3]=object_translations[object_id]

                    else:
                        M_o2c=np.identity(4)
                        M_o2c[:3,:3]=R.from_quat(object_rotations[object_id]).as_matrix()
                        M_o2c[:3,3]=object_translations[object_id]


                    keypoint_B1=np.linalg.inv(M_o2c) @ np.vstack((keypoints_3d[object_id][1].reshape(3, -1),1))

                    keypoint_handle=(keypoints_3d[object_id][5]+keypoints_3d[object_id][8])/2

                    # Project keypoint_handle to the image plane via the Official OpenGL way 
                    k_3d=np.array([keypoint_handle[0],keypoint_handle[1],keypoint_handle[2],1])
                    pp2=cam_projection_matrix @ k_3d.reshape(4, 1)
                    pp2=(pp2/pp2[3])[:3]
                    viewport_point = (pp2 + 1.0)/2.0 * np.array([height, width, 1.0]).reshape(3,1)

                    # -> opencv x, opencv y
                    viewport_point=[viewport_point[1],viewport_point[0]]

                    # Todo: Give some tolerance since the keypoint_handle is the extreme value
                    if (viewport_point[0] < width * 1.2 and viewport_point[0] > - width*0.2 and viewport_point[1] > - height * 0.2 and viewport_point[1] < height * 1.2):
                        # Handle in the object frame
                        keypoint_L1=np.linalg.inv(M_o2c) @ np.vstack((keypoint_handle.reshape(3, -1),1))

                        keypoint_outmost=keypoints_3d[object_id][4]+(keypoints_3d[object_id][8]-keypoints_3d[object_id][4])*2/3
                        
                        keypoint_B2=np.linalg.inv(M_o2c) @ np.vstack((keypoint_outmost.reshape(3, -1),1))

                        # Camera in the object frame
                        keypoint_L2=np.linalg.inv(M_o2c) @ np.array([[0],[0],[0],[1]])

                        # Todo: Check three possible keypoints for the handle, not 100% correct
                        diff=(keypoint_B2-keypoint_B1)[1]/6
                        checker=CheckLineBox( keypoint_B1, keypoint_B2,  keypoint_L1+diff,  keypoint_L2)
                        visible_flag_1 = not checker.check()

                        checker=CheckLineBox( keypoint_B1, keypoint_B2,  keypoint_L1-diff,  keypoint_L2)
                        visible_flag_2 = not checker.check()

                        checker=CheckLineBox( keypoint_B1, keypoint_B2,  keypoint_L1,  keypoint_L2)
                        visible_flag_3= not checker.check()

                        visible_flag = visible_flag_1 | visible_flag_2 | visible_flag_3
                    else:
                        visible_flag = False

            # Save info to .json
            dict_obj={
                'class':object_categories[object_id],
                'name':object_categories[object_id]+'_'+str(object_id),
                'provenance':'objectron',
                'location': [  # M_o2c in the OpenGL cam frame
                    object_translations[object_id][0],
                    object_translations[object_id][1],
                    object_translations[object_id][2],
                ],
                'quaternion_xyzw':[ # M_o2c in the OpenGL cam frame
                    object_rotations[object_id][0],
                    object_rotations[object_id][1],
                    object_rotations[object_id][2],
                    object_rotations[object_id][3],
                ],
                'projected_cuboid':projected_keypoints.tolist(), # 2d keypoints in the opencv image frame
                'scale':object_scale[object_id].tolist(),
                'keypoints_3d':keypoints_3d[object_id].tolist(), # 3d keypoints in the OpenGL cam frame
            }

            if 'cup' in filename:
                dict_obj.update({'mug': mug_flag})
                if mug_flag == True:
                    dict_obj.update({'mug_handle_visible': visible_flag})
            
            if 'chair' in filename:
                if [video_name,str(object_id)] in chair_symmetric_list:
                    dict_obj.update({'symmetric': True})
                else:
                    dict_obj.update({'symmetric': False})
                    
            # Final export
            dict_out['objects'].append(dict_obj)
            
        with open(filename, 'w+') as fp:
            json.dump(dict_out, fp, indent=4, sort_keys=True)

        return 0
    except:
        return 1




