from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import os
import math

class Debugger(object):
    def __init__(self, ipynb=False, theme='black',
                 num_classes=-1, dataset=None, down_ratio=4):
        self.ipynb = ipynb
        if not self.ipynb:
            import matplotlib.pyplot as plt
            self.plt = plt
        self.imgs = {}
        self.theme = theme
        colors = [(color_list[_]).astype(np.uint8) \
                  for _ in range(len(color_list))]
        self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
        if self.theme == 'white':
            self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
            self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
        self.dim_scale = 1

        self.names = ['op']
        self.num_class = 1
        self.num_joints = 8
        self.edges = [[2, 4], [2, 6], [6, 8], [4, 8],
                      [1, 2], [3, 4], [5, 6], [7, 8],
                      [1, 3], [1, 5], [3, 7], [5, 7]]

        self.top_cross = [[3, 8], [4, 7]]
        self.front_cross = [[2, 8], [4, 6]]
        # edge color

        # BGR for prediction
        self.ec = [(255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
                   (255, 0, 0), (255, 0, 0), (255, 0, 0),
                   (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0)]

        # To show the diff rather than all blue
        self.colors_hp = [(0, 0, 255), (0, 165, 255), (0, 255, 255),
                          (0, 128, 0), (255, 0, 0), (130, 0, 75), (238, 130, 238),
                          (0, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0)]

        # self.colors_hp = [(255, 0, 0), (255, 0, 0), (255, 0, 0),
        #                   (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
        #                   (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0)]

        num_classes = len(self.names)
        self.down_ratio = down_ratio
        # for bird view
        self.world_size = 64
        self.out_size = 384

    def add_img(self, img, img_id='default', revert_color=False):
        if revert_color:
            img = 255 - img
        self.imgs[img_id] = img.copy()

    def add_blend_img(self, back, fore, img_id='blend', trans=0.6):
        if self.theme == 'white':
            fore = 255 - fore
        if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
            fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
        if len(fore.shape) == 2:
            fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
        self.imgs[img_id] = (back * (1. - trans) + fore * trans)
        self.imgs[img_id][self.imgs[img_id] > 255] = 255
        self.imgs[img_id][self.imgs[img_id] < 0] = 0
        self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

    def gen_colormap(self, img, output_res=None, color=None):
        img = img.copy()
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        if output_res is None:
            output_res = (h * self.down_ratio, w * self.down_ratio)
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)

        if color == None:
            colors = np.array(
                self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
        else:
            colors = np.array(color)

        if self.theme == 'white':
            colors = 255 - colors

        # # For paper
        # colors = np.array([255, 255, 0]).reshape(1, 1, 1, 3)

        color_map = (img * colors).max(axis=2).astype(np.uint8)
        color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
        return color_map

    '''
    # slow
    def gen_colormap_hp(self, img, output_res=None):
      # num_classes = len(self.colors)
      # img[img < 0] = 0
      h, w = img.shape[1], img.shape[2]
      if output_res is None:
        output_res = (h * self.down_ratio, w * self.down_ratio)
      color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
      for i in range(img.shape[0]):
        resized = cv2.resize(img[i], (output_res[1], output_res[0]))
        resized = resized.reshape(output_res[0], output_res[1], 1)
        cl =  self.colors_hp[i] if not (self.theme == 'white') else \
          (255 - np.array(self.colors_hp[i]))
        color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
      return color_map
    '''

    def gen_colormap_hp(self, img, output_res=None):
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        if output_res is None:
            output_res = (h * self.down_ratio, w * self.down_ratio)
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        colors = np.array(
            self.colors_hp, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
        if self.theme == 'white':
            colors = 255 - colors

        color_map = (img * colors).max(axis=2).astype(np.uint8)
        color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
        return color_map

    # Draw a 2D bounding box
    def add_coco_bbox(self, bbox, cat, conf=1, id=None, show_txt=True, img_id='default'):
        bbox = np.array(bbox, dtype=np.int32)

        # Todo: Clip for better visualization, maybe unnecessary
        height, width = self.imgs[img_id].shape[0], self.imgs[img_id].shape[1]
        bbox[0] = np.clip(bbox[0], 0, width)
        bbox[1] = np.clip(bbox[1], 0, height)
        bbox[2] = np.clip(bbox[2], 0, width)
        bbox[3] = np.clip(bbox[3], 0, height)

        cat = int(cat)
        c = self.colors[cat][0][0].tolist()
        if self.theme == 'white':
            c = (255 - np.array(c)).tolist()
        if id is None:
            txt = '{}{:.1f}'.format(self.names[cat], conf)
        else:
            txt = '{}{:.1f} ID:{}'.format(self.names[cat], conf, id)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]

        # Todo: For simplicity, comment it to not display it
        cv2.rectangle(
            self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)

        if show_txt:
            cv2.rectangle(self.imgs[img_id],
                          (bbox[0], bbox[1] - cat_size[1] - 2),
                          (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
            cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2),
                        font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    # Draw the numbers in a small region
    def add_obj_scale(self, bbox, scale, img_id='default', pred_flag='pred'):
        bbox = np.array(bbox, dtype=np.int32)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if pred_flag == 'pred':
            txt = 'Pred:{:.3f}/{:.3f}/{:.3f}'.format(scale[0], scale[1], scale[2])
            cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
            cv2.rectangle(self.imgs[img_id],
                          (bbox[0], bbox[1] + cat_size[1] + 2),
                          (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + cat_size[1] + 4), (0, 0, 0), -1)
            cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] + cat_size[1] + cat_size[1]),
                        font, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        elif pred_flag == 'gt':
            txt = 'GT:{:.3f}/{:.3f}/{:.3f}'.format(scale[0], scale[1], scale[2])
            cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
            cv2.rectangle(self.imgs[img_id],
                          (bbox[0], bbox[1] + 2),
                          (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2), (0, 0, 0), -1)
            cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] + cat_size[1]),
                        font, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        elif pred_flag == 'pnp':
            txt = 'PnP:{:.3f}/{:.3f}/{:.3f}'.format(scale[0], scale[1], scale[2])
            cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
            cv2.rectangle(self.imgs[img_id],
                          (bbox[0], bbox[1] + 2),
                          (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 6), (0, 0, 0), -1)
            cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] + cat_size[1]),
                        font, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Connect center to corners
    def add_coco_hp_paper(self, bbox, points, std=None, img_id='default'):

        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.int32)

        points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
        if std is not None:
            std = np.array(std, dtype=np.int32).reshape(self.num_joints, 2)
        for j in range(self.num_joints):
            if std is None:
                cv2.circle(self.imgs[img_id],
                           (points[j, 0], points[j, 1]), 5, self.colors_hp[j], -1)
            else:
                cv2.ellipse(self.imgs[img_id],
                            (points[j, 0], points[j, 1]), (std[j, 0], std[j, 1]), 0, 0, 360, self.colors_hp[j], 2)

            cv2.line(self.imgs[img_id],
                     (points[j, 0], points[j, 1]), (center[0], center[1]), self.colors_hp[j], 1, lineType=cv2.LINE_AA)

    # Draw a 3D bounding box with two crosses
    def add_coco_hp(self, points, img_id='default', pred_flag='pred', PAPER_DISPLAY=False):
        points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)

        # Draw corner points
        for j in range(self.num_joints):

            if not PAPER_DISPLAY:
                cv2.circle(self.imgs[img_id],
                           (points[j, 0], points[j, 1]), 5, self.colors_hp[j], -1)
            else:
                cv2.circle(self.imgs[img_id],
                           (points[j, 0], points[j, 1]), 10, (255, 255, 0), -1)


        # Draw edges
        for j, e in enumerate(self.edges):
            temp = [e[0] - 1, e[1] - 1]
            if pred_flag == 'pred':
                edge_color = (0, 0, 255)  # bgr
            elif pred_flag == 'gt':
                edge_color = (0, 255, 0)
            elif pred_flag == 'pnp':
                edge_color = (0, 0, 255)  # bgr
            elif pred_flag == 'extra':
                edge_color = (0, 165, 255)
            if points[temp[1], 0] <= -10000 or points[temp[1], 1] <= -10000 or points[temp[0], 0] <= -10000 or \
                    points[temp[0], 1] <= -10000:
                continue
            else:
                cv2.line(self.imgs[img_id], (points[temp[0], 0], points[temp[0], 1]),
                         (points[temp[1], 0], points[temp[1], 1]), edge_color, 2)

        if pred_flag == 'pred':
            edge_color = (0, 0, 255)
        elif pred_flag == 'gt':
            edge_color = (255, 255, 255)
        elif pred_flag == 'pnp':
            edge_color = (0, 0, 0)

        # def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
        #     px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
        #                 (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        #     py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
        #                 (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        #
        #     # Todo: for simplicity
        #     if math.isnan(px) or math.isnan(py) or math.isinf(px) or math.isinf(py):
        #         return 0,0
        #     else:
        #         return (int(px), int(py))
        #
        # # Draw center
        # cv2.circle(self.imgs[img_id],
        #            findIntersection(points[1, 0], points[1, 1], points[7, 0], points[7, 1], \
        #                             points[3, 0], points[3, 1], points[5, 0], points[5, 1]), 5, edge_color, -1)
        #
        # cv2.circle(self.imgs[img_id],
        #            findIntersection(points[2, 0], points[2, 1], points[7, 0], points[7, 1], \
        #                             points[3, 0], points[3, 1], points[6, 0], points[6, 1]), 5, edge_color, -1)

        if not PAPER_DISPLAY:
            # Draw cross
            for j, e in enumerate(self.front_cross):
                temp = [e[0] - 1, e[1] - 1]

                if points[temp[1], 0] <= -10000 or points[temp[1], 1] <= -10000 or points[temp[0], 0] <= -10000 or \
                        points[temp[0], 1] <= -10000:
                    continue
                else:
                    cv2.line(self.imgs[img_id], (points[temp[0], 0], points[temp[0], 1]),
                             (points[temp[1], 0], points[temp[1], 1]), edge_color, 2,
                             lineType=cv2.LINE_AA)

                for j, e in enumerate(self.top_cross):
                    temp = [e[0] - 1, e[1] - 1]

                    if points[temp[1], 0] <= -10000 or points[temp[1], 1] <= -10000 or points[temp[0], 0] <= -10000 or \
                            points[temp[0], 1] <= -10000:
                        continue
                    else:
                        cv2.line(self.imgs[img_id], (points[temp[0], 0], points[temp[0], 1]),
                                 (points[temp[1], 0], points[temp[1], 1]), edge_color, 2,
                                 lineType=cv2.LINE_AA)


    def add_axes(self, box, cam_intrinsic, img_id='default'):
        # box 9x3 array
        # OpenCV way
        N = 0.5
        # Centroid, top, front, right
        axes_point_list = [0, box[3] - box[1], box[2] - box[1], box[5] - box[1]]
        viewport_point_list = []
        for axes_point in axes_point_list:
            vector = axes_point
            vector = vector / np.linalg.norm(vector) * N if np.linalg.norm(vector) != 0 else 0
            vector = vector + box[0]
            vector = vector.flatten()

            k_3d = np.array([vector[0], vector[1], vector[2]])
            pp = np.matmul(cam_intrinsic, k_3d.reshape(3, 1))
            viewport_point = [pp[0] / pp[2], pp[1] / pp[2]]
            viewport_point_list.append((int(viewport_point[0]), int(viewport_point[1])))

        # BGR space
        cv2.line(self.imgs[img_id], viewport_point_list[0], viewport_point_list[1], (0, 255, 0), 5)  # y-> green
        cv2.line(self.imgs[img_id], viewport_point_list[0], viewport_point_list[2], (255, 0, 0), 5)  # z-> blue
        cv2.line(self.imgs[img_id], viewport_point_list[0], viewport_point_list[3], (0, 0, 255), 5)  # x-> red


    # Draw an arrow for track offsets
    def add_arrow(self, st, ed, img_id, c=(255, 0, 255), w=2):
        # st is current
        # ed is previous-current

        # A vector from previous to current
        cv2.arrowedLine(
            self.imgs[img_id],
            (int(ed[0] + st[0]), int(ed[1] + st[1])), (int(st[0]), int(st[1])), c, w,
            line_type=cv2.LINE_AA, tipLength=0.3)

    # To show the images for debug
    def show_all_imgs(self, pause=False, time=0):
        if not self.ipynb:
            for i, v in self.imgs.items():
                cv2.imshow('{}'.format(i), v)
            if cv2.waitKey(0 if pause else 1) == 27:
                import sys
                sys.exit(0)
        else:
            self.ax = None
            nImgs = len(self.imgs)
            fig = self.plt.figure(figsize=(nImgs * 10, 10))
            nCols = nImgs
            nRows = nImgs // nCols
            for i, (k, v) in enumerate(self.imgs.items()):
                fig.add_subplot(1, nImgs, i + 1)
                if len(v.shape) == 3:
                    self.plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
                else:
                    self.plt.imshow(v)
            self.plt.show()

    def save_all_imgs(self, path='./cache/debug/', prefix=''):
        for i, v in self.imgs.items():
            cv2.imwrite(path + '/{}{}.png'.format(prefix, i), v)

    # save_all_imgs_eval and save_all_imgs_demo to save images in different places
    def save_all_imgs_eval(self, src_path, path='./cache/debug/', video_layout=False):
        # E.g., 'book_batch-9_5_00000' for video source, 'chair_batch-40_36_1' for official source

        file_id_name = src_path[src_path.rfind('_') + 1:]
        folder_name = src_path[:src_path.rfind('_')]

        txt = f'Frame:{int(file_id_name)}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]

        for i, v in self.imgs.items():
            cv2.putText(v, txt, (5, v.shape[0] - cat_size[1]), font, 0.5, (255, 255, 255), thickness=1,
                        lineType=cv2.LINE_AA)
            if video_layout:
                cv2.imwrite(path + '/{}_{}.png'.format(file_id_name, i), v)
            else:
                cv2.imwrite(path + '/{}_{}_{}.png'.format(folder_name, file_id_name, i), v)

    def save_all_imgs_demo(self, src_path, path='./cache/debug/'):

        # We assume the images should have a name like '00000.png', otherwise ignore the frame stamp
        file_id_name = os.path.splitext(os.path.basename(src_path))[0]  # file id name

        if os.path.isdir(src_path):
            # Folder case
            folder_name = src_path.split('/')[-2]  # folder name
            if file_id_name.isnumeric():
                txt = f'Frame: {int(file_id_name)}'
                font = cv2.FONT_HERSHEY_SIMPLEX
                cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
            for i, v in self.imgs.items():
                if file_id_name.isnumeric():
                    cv2.putText(v, txt, (5, v.shape[0] - cat_size[1]), font, 0.5, (255, 255, 255), thickness=1,
                                lineType=cv2.LINE_AA)
                cv2.imwrite(path + '/{}_{}_{}.png'.format(folder_name, file_id_name, i), v)
        else:
            # Video/image case
            for i, v in self.imgs.items():
                cv2.imwrite(path + '/{}_{}.png'.format(file_id_name, i), v)

color_list = np.array(
    [
        1.000, 1.000, 1.000,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
