# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

from enum import IntEnum

import cv2
import numpy as np


# Related to the object's local coordinate system
# @unique
class CuboidVertexType(IntEnum):
    FrontTopRight = 0
    FrontTopLeft = 1
    FrontBottomLeft = 2
    FrontBottomRight = 3
    RearTopRight = 4
    RearTopLeft = 5
    RearBottomLeft = 6
    RearBottomRight = 7
    Center = 8
    TotalCornerVertexCount = 8  # Corner vertexes doesn't include the center point
    TotalVertexCount = 9


# List of the vertex indexes in each line edges of the cuboid
CuboidLineIndexes = [
    # Front face
    [CuboidVertexType.FrontTopLeft, CuboidVertexType.FrontTopRight],
    [CuboidVertexType.FrontTopRight, CuboidVertexType.FrontBottomRight],
    [CuboidVertexType.FrontBottomRight, CuboidVertexType.FrontBottomLeft],
    [CuboidVertexType.FrontBottomLeft, CuboidVertexType.FrontTopLeft],
    # Back face
    [CuboidVertexType.RearTopLeft, CuboidVertexType.RearTopRight],
    [CuboidVertexType.RearTopRight, CuboidVertexType.RearBottomRight],
    [CuboidVertexType.RearBottomRight, CuboidVertexType.RearBottomLeft],
    [CuboidVertexType.RearBottomLeft, CuboidVertexType.RearTopLeft],
    # Left face
    [CuboidVertexType.FrontBottomLeft, CuboidVertexType.RearBottomLeft],
    [CuboidVertexType.FrontTopLeft, CuboidVertexType.RearTopLeft],
    # Right face
    [CuboidVertexType.FrontBottomRight, CuboidVertexType.RearBottomRight],
    [CuboidVertexType.FrontTopRight, CuboidVertexType.RearTopRight],
]


# ========================= Cuboid3d =========================
class Cuboid3d():
    '''This class contains a 3D cuboid.'''

    # Create a box with a certain size
    def __init__(self, size3d=[1.0, 1.0, 1.0],
                 coord_system=None, parent_object=None):
        # NOTE: This local coordinate system is similar
        # to the intrinsic transform matrix of a 3d object
        self.center_location = [0, 0, 0]
        # self.center_location = [size3d[0]/2,size3d[1]/2,size3d[2]/2]
        self.coord_system = coord_system
        self.size3d = size3d
        self._vertices = [0, 0, 0] * CuboidVertexType.TotalCornerVertexCount
        # self._vertices = [0, 0, 0] * CuboidVertexType.TotalVertexCount

        self.generate_vertexes()

    def get_vertex(self, vertex_type):
        """Returns the location of a vertex.

        Args:
            vertex_type: enum of type CuboidVertexType

        Returns:
            Numpy array(3) - Location of the vertex type in the cuboid
        """
        return self._vertices[vertex_type]

    def get_vertices(self):
        return self._vertices

    def generate_vertexes(self):
        width, height, depth = self.size3d

        # By default just use the normal OpenCV coordinate system
        if (self.coord_system is None):
            cx, cy, cz = self.center_location
            # X axis point to the right
            right = cx + width / 2.0
            left = cx - width / 2.0
            # Y axis point upward
            top = cy + height / 2.0
            bottom = cy - height / 2.0
            # Z axis point forward
            front = cz + depth / 2.0
            rear = cz - depth / 2.0

            # List of 8 vertices of the box
            self._vertices = [
                # self.center_location,   # Center
                [left, bottom, rear],  # Rear Bottom Left
                [left, bottom, front],  # Front Bottom Left
                [left, top, rear],  # Rear Top Left
                [left, top, front],  # Front Top Left

                [right, bottom, rear],  # Rear Bottom Right
                [right, bottom, front],  # Front Bottom Right
                [right, top, rear],  # Rear Top Right
                [right, top, front],  # Front Top Right

            ]

