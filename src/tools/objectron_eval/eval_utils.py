
"""
Save some utility functions for the evaluation process, including drawing and parser.
"""

import numpy as np
import cv2
import json

# Draw pose axes
def draw_axes(image_debug, box, projection_matrix, height, width, c=None):

    # length of the axes, have to be adjusted for different categories
    if c in ['bike']:
        N =0.4
    elif c in ['book', 'bottle', 'camera', 'cereal_box']:
        N = 0.05
    elif c in ['chair']:
        N = 0.2
    elif c in ['laptop','shoe']:
        N = 0.1
    else:
        N = 0.05

    # Centroid, top, front, right
    axes_point_list = [0, box[3] - box[1], box[2] - box[1], box[5] - box[1]]
    viewport_point_list = []
    for axes_point in axes_point_list:
        vector = axes_point
        vector = vector / np.linalg.norm(vector) * N if np.linalg.norm(vector) != 0 else 0
        vector = vector + box[0]
        vector = vector.reshape(-1, 3)
        up_2d = projection_matrix @ np.hstack((vector, np.ones((vector.shape[0], 1)))).T

        up_2d = (up_2d / up_2d[3])[:3]
        viewport_point = (up_2d + 1.0) / 2.0 * np.array([height, width, 1.0]).reshape(3, 1)
        viewport_point_list.append((int(viewport_point[1]), int(viewport_point[0])))

    # RGB space
    cv2.line(image_debug, viewport_point_list[0], viewport_point_list[1], (0, 255, 0), 5)  # y-> green
    cv2.line(image_debug, viewport_point_list[0], viewport_point_list[2], (0, 0, 255), 5)  # z-> blue
    cv2.line(image_debug, viewport_point_list[0], viewport_point_list[3], (255, 0, 0), 5)  # x-> red

