import cv2
import numpy as np


def random_range(a=0, b=1):
    return np.random.rand() * (b - a) + a


def change_color_space(image, current_space="BGR", to_space="BGR"):
    if not ((current_space.lower() in {'bgr', 'rgb', 'hsv'}) 
            and (to_space.lower() in {'bgr', 'rgb', 'hsv', 'gray'})):
        raise NotImplementedError
    if current_space.lower() == 'bgr' and to_space.lower() == 'rgb':
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if current_space.lower() == 'rgb' and to_space.lower() == 'bgr':
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif current_space.lower() == 'bgr' and to_space.lower() == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image


def coordinates_converter(bboxes, conversion=None):
    if conversion == "centroids2corners":
        boxes_xy = bboxes[..., 0:2] - 0.5 * bboxes[..., 2:4]
        boxes_wh = bboxes[..., 0:2] + 0.5 * bboxes[..., 2:4]
        bboxes[..., 0:2] = boxes_xy
        bboxes[..., 2:4] = boxes_wh
    elif conversion == "corners2centroids":
        boxes_xy  = (bboxes[..., 0:2] + bboxes[..., 2:4]) // 2
        boxes_wh = bboxes[..., 2:4] - bboxes[..., 0:2]
        bboxes[..., 0:2] = boxes_xy
        bboxes[..., 2:4] = boxes_wh
    return bboxes
