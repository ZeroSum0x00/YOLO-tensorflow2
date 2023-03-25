import os
import cv2
import xml
import numpy as np

from utils.files import extract_zip, verify_folder, get_files
from data_utils import ParseVOC, ParseCOCO, ParseYOLO, ParseTXT
from configs import general_config as cfg


def extract_data_folder(data_dir, dst_dir=None):
    ACCEPTABLE_EXTRACT_FORMATS = ['.zip', '.rar', '.tar']
    if (os.path.isfile(data_dir)) and os.path.splitext(data_dir)[-1] in ACCEPTABLE_EXTRACT_FORMATS:
        if dst_dir is not None:
            data_destination = dst_dir
        else:
            data_destination = '/'.join(data_dir.split('/')[: -1])

        folder_name = data_dir.split('/')[-1]
        folder_name = os.path.splitext(folder_name)[0]
        data_destination = verify_folder(data_destination) + folder_name 

        if not os.path.isdir(data_destination):
            extract_zip(data_dir, data_destination)
        
        return data_destination
    else:
        return data_dir


def get_data(data_dir       = cfg.DATA_PATH, 
             annotation_dir = cfg.DATA_ANNOTATION_PATH,
             classes        = cfg.YOLO_CLASSES, 
             data_type      = cfg.DATA_TYPE,
             phase          = 'train', 
             check_data     = cfg.CHECK_DATA,
             load_memory    = cfg.DATA_LOAD_MEMORY,
             *args, **kwargs):
    data_dir = verify_folder(data_dir) + phase
    data_extraction = []
    
    if data_type.lower() == "voc" or data_type.lower() == 'pascal':
        xml_files = sorted(get_files(data_dir, extensions='xml'))
        parser = ParseVOC(data_dir, annotation_dir, classes, load_memory, check_data=check_data, *args, **kwargs)
        data_extraction = parser(xml_files)
    elif data_type.lower() == "coco":
        annotation_file = verify_folder(annotation_dir) + f'instances_{phase}.json'
        parser = ParseCOCO(data_dir, annotation_file, load_memory, check_data=check_data, *args, **kwargs)
        data_extraction = parser()
    elif data_type.lower() == "yolo":
        txt_files = sorted(get_files(data_dir, extensions='txt'))
        parser = ParseYOLO(data_dir, annotation_dir, classes, load_memory, check_data=check_data, *args, **kwargs)
        data_extraction = parser(txt_files)
    elif data_type.lower() == "txt" or data_type.lower() == "text":
        annotation_file = verify_folder(annotation_dir) + f'instances_{phase}.txt'
        parser = ParseTXT(data_dir, annotation_file, load_memory, check_data=check_data, *args, **kwargs)
        data_extraction = parser()
    dict_data = {
        'data_path': verify_folder(data_dir),
        'data_extractor': data_extraction
    }
    return dict_data


class Normalizer():
    def __init__(self, max_bboxes=cfg.YOLO_MAX_BBOXES, mode=cfg.DATA_NORMALIZER):
        self.mode = mode
        self.max_bboxes = max_bboxes

    @classmethod
    def __get_standard_deviation(cls, img, mean=None, std=None):
        if mean is not None:
            for i in range(img.shape[-1]):
                if isinstance(mean, float) or isinstance(mean, int):
                    img[..., i] -= mean
                else:
                    img[..., i] -= mean[i]

                if std is not None:
                    for i in range(img.shape[-1]):
                        if isinstance(std, float) or isinstance(std, int):
                            img[..., i] /= (std + 1e-20)
                        else:
                            img[..., i] /= (std[i] + 1e-20)
        return img

    @classmethod
    def __resize_basic_mode(cls, image, bboxes, target_size, max_bboxes, interpolation=None):
        h, w, _ = image.shape
        image_resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)

        box_data = np.zeros((max_bboxes, 5))
        if len(bboxes) > 0:
            for index, box in enumerate(bboxes):
                box[0] *= (target_size[1] / w)
                box[1] *= (target_size[0] / h)
                box[2] *= (target_size[1] / w)
                box[3] *= (target_size[0] / h) 
                bboxes[index] = box
            if len(bboxes) > max_bboxes: bboxes = bboxes[:max_bboxes]
            box_data[:len(bboxes)] = bboxes
        return image_resized, box_data

    def _sub_divide(self, image, bboxes=None, mean=None, std=None, target_size=None, interpolation=None):
        if target_size and image.shape[0] != target_size[0] and image.shape[1] != target_size[1]:
            image, bboxes = self.__resize_basic_mode(image, bboxes, target_size, self.max_bboxes, interpolation)
        image = image.astype(np.float32)
        image = image / 127.5 - 1
        if mean or std:
            image = self.__get_standard_deviation(image, mean, std)
        image = np.clip(image, -1, 1)
        return image, bboxes

    def _divide(self, image, bboxes=None, mean=None, std=None, target_size=None, interpolation=None):
        if target_size and image.shape[0] != target_size[0] and image.shape[1] != target_size[1]:
            image, bboxes = self.__resize_basic_mode(image, bboxes, target_size, self.max_bboxes, interpolation)
        image = image.astype(np.float32)
        image = image / 255.0
        if mean or std:
            image = self.__get_standard_deviation(image, mean, std)
        image = np.clip(image, 0, 1)
        return image, bboxes

    def _basic(self, image, bboxes=None, mean=None, std=None, target_size=None, interpolation=None):
        if target_size and image.shape[0] != target_size[0] and image.shape[1] != target_size[1]:
            image, bboxes = self.__resize_basic_mode(image, bboxes, target_size, self.max_bboxes, interpolation)
        image = image.astype(np.uint8)
        if mean or std:
            image = self.__get_standard_deviation(image, mean, std)
        image = np.clip(image, 0, 255)
        return image, bboxes

    def __call__(self, input, *args, **kargs):
        if self.mode == "divide":
            return self._divide(input, *args, **kargs)
        elif self.mode == "sub_divide":
            return self._sub_divide(input, *args, **kargs)
        else:
            return self._basic(input, *args, **kargs)

            
def preprocess_true_boxes(true_boxes, input_shape, anchors, anchors_mask, num_classes, strides, coords="corners"):
    true_boxes  = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')

    num_layers  = len(anchors_mask)

    batch       = true_boxes.shape[0]
    grid_shapes = [input_shape[:-1] // np.array(strides[i]) for i in range(len(strides))]
    y_true = [np.zeros((batch, grid_shapes[i][0], grid_shapes[i][1], len(anchors_mask[i]), 5 + num_classes), dtype='float32') for i in range(num_layers)]

    if coords == 'centroids':
        boxes_xy   = true_boxes[..., 0:2].copy()
        boxes_wh   = true_boxes[..., 2:4].copy()
    else:
        if coords == "corners":
            boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2       # Tính toán x_center, y_center
            boxes_wh =  true_boxes[..., 2:4] - true_boxes[..., 0:2]             # Tính toán w, h
        elif coords == "minmax":
            boxes_x = (true_boxes[..., 0] + true_boxes[..., 1]) // 2            # Tính toán x_center
            boxes_y = (true_boxes[..., 2] + true_boxes[..., 3]) // 2            # Tính toán y_center
            boxes_xy = np.stack([boxes_x, boxes_y], axis=-1)

            boxes_w = true_boxes[..., 1] - true_boxes[..., 0]                   # Tính toán w
            boxes_h = true_boxes[..., 3] - true_boxes[..., 2]                   # Tính toán h
            boxes_wh =  np.stack([boxes_w, boxes_h], axis=-1)

    true_boxes[..., 0:2] = boxes_xy / input_shape[:-1][::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[:-1][::-1]
    
    anchors         = np.expand_dims(anchors, axis=0)
    anchor_maxes    = anchors / 2.
    anchor_mins     = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(batch):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: 
            continue

        wh          = np.expand_dims(wh, -2)
        box_maxes   = wh / 2.
        box_mins    = -box_maxes
        intersect_mins  = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area    = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):

            for l in range(num_layers):
                if n in anchors_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    if coords == 'centroids':
                        if i == grid_shapes[l][1]:
                            i -= 1
                        elif j == grid_shapes[l][0]:
                            j -= 1
                    k = anchors_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')

                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1
    return y_true
