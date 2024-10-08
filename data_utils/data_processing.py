import os
import cv2
import types
import numpy as np

from utils.files import extract_zip, get_files
from data_utils import ParseVOC, ParseCOCO, ParseYOLO, ParseTXT


def extract_data_folder(data_dir, dst_dir=None):
    ACCEPTABLE_EXTRACT_FORMATS = ['.zip', '.rar', '.tar']
    if (os.path.isfile(data_dir)) and os.path.splitext(data_dir)[-1] in ACCEPTABLE_EXTRACT_FORMATS:
        if dst_dir is not None:
            data_destination = dst_dir
        else:
            data_destination = '/'.join(data_dir.split('/')[: -1])

        folder_name = data_dir.split('/')[-1]
        folder_name = os.path.splitext(folder_name)[0]
        data_destination = os.path.join(data_destination, folder_name)

        if not os.path.isdir(data_destination):
            extract_zip(data_dir, data_destination)
        
        return data_destination
    else:
        return data_dir


def get_data(data_dir, 
             annotation_dir = None,
             classes        = 80, 
             data_type      = 'coco',
             phase          = 'train', 
             check_data     = False,
             load_memory    = False,
             *args, **kwargs):
    data_dir = os.path.join(data_dir, phase)
    data_extraction = []
    
    if data_type.lower() == "voc" or data_type.lower() == 'pascal':
        xml_files = sorted(get_files(data_dir, extensions='xml'))
        parser = ParseVOC(data_dir, annotation_dir, classes, load_memory, check_data=check_data, *args, **kwargs)
        data_extraction = parser(xml_files)
    elif data_type.lower() == "coco":
        annotation_file = os.path.join(annotation_dir, f'instances_{phase}.json')
        parser = ParseCOCO(data_dir, annotation_file, load_memory, check_data=check_data, *args, **kwargs)
        data_extraction = parser()
    elif data_type.lower() == "yolo":
        txt_files = sorted(get_files(data_dir, extensions='txt'))
        parser = ParseYOLO(data_dir, annotation_dir, classes, load_memory, check_data=check_data, *args, **kwargs)
        data_extraction = parser(txt_files)
    elif data_type.lower() == "txt" or data_type.lower() == "text":
        annotation_file = os.path.join(annotation_dir, f'instances_{phase}.txt')
        parser = ParseTXT(data_dir, annotation_file, load_memory, check_data=check_data, *args, **kwargs)
        data_extraction = parser()
    dict_data = {
        'data_path': data_dir,
        'data_extractor': data_extraction
    }
    return dict_data


class Normalizer():
    def __init__(self, norm_type='divide', mean=None, std=None, max_bboxes=100):
        self.norm_type  = norm_type
        self.mean       = mean
        self.std        = std
        self.max_bboxes = max_bboxes

    def __get_standard_deviation(self, img):
        if self.mean is not None:
            for i in range(img.shape[-1]):
                if isinstance(self.mean, float) or isinstance(self.mean, int):
                    img[..., i] -= self.mean
                else:
                    img[..., i] -= self.mean[i]

        if self.std is not None:
            for i in range(img.shape[-1]):
                if isinstance(self.std, float) or isinstance(self.std, int):
                    img[..., i] /= (self.std + 1e-20)
                else:
                    img[..., i] /= (self.std[i] + 1e-20)
        return img

    @classmethod
    def __resize_basic_mode(cls, image, bboxes, target_size, max_bboxes, interpolation=None):
        h, w, _ = image.shape
        image_resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)

        box_data = np.zeros((max_bboxes, 5))
        box_data[:, -1] = -1
        
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

    @classmethod
    def __remove_jam_box(cls, bboxes, target_size):
        for idx in range(len(bboxes)):
            xmin, ymin, xmax, ymax, cls = bboxes[idx]
            if cls == -1:
                bboxes[idx] = [0, 0, 0, 0, -1]
            elif (xmax - xmin < target_size[1] * 0.025) or (ymax - ymin < target_size[0] * 0.025):
                bboxes[idx] = [0, 0, 0, 0, -1]
        return bboxes
        
    def _sub_divide(self, image, bboxes=None, target_size=None, interpolation=None):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image, bboxes = self.__resize_basic_mode(image, bboxes, target_size, self.max_bboxes, interpolation)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image = image.astype(np.float32)
        image = image / 127.5 - 1
        image = self.__get_standard_deviation(image)
        image = np.clip(image, -1, 1)
        bboxes = self.__remove_jam_box(bboxes, target_size)
        return image, bboxes

    def _divide(self, image, bboxes=None, target_size=None, interpolation=None):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image, bboxes = self.__resize_basic_mode(image, bboxes, target_size, self.max_bboxes, interpolation)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image = image.astype(np.float32)
        image = image / 255.0
        image = self.__get_standard_deviation(image)
        image = np.clip(image, 0, 1)
        bboxes = self.__remove_jam_box(bboxes, target_size)
        return image, bboxes

    def _basic(self, image, bboxes=None, target_size=None, interpolation=None):
        if target_size and (image.shape[0] != target_size[0] or image.shape[1] != target_size[1]):
            image, bboxes = self.__resize_basic_mode(image, bboxes, target_size, self.max_bboxes, interpolation)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image = image.astype(np.uint8)
        image = self.__get_standard_deviation(image)
        image = np.clip(image, 0, 255)
        bboxes = self.__remove_jam_box(bboxes, target_size)
        return image, bboxes

    def __call__(self, input, *args, **kargs):
        if isinstance(self.norm_type, str):
            if self.norm_type == "divide":
                return self._divide(input, *args, **kargs)
            elif self.norm_type == "sub_divide":
                return self._sub_divide(input, *args, **kargs)
        elif isinstance(self.norm_type, types.FunctionType):
            return self._func_calc(input, self.norm_type, *args, **kargs)
        else:
            return self._basic(input, *args, **kargs)

            
def preprocess_true_boxes(true_boxes, input_shape, anchors, anchors_mask, num_classes, strides, coords="corners"):
    true_boxes  = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape[:-1], dtype='int32')

    num_layers  = len(anchors_mask)

    batch       = true_boxes.shape[0]
    grid_shapes = [input_shape // np.array(strides * 2**(num_layers - 1 - i)) for i in range(num_layers)]
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

    true_boxes[..., 0:2] = boxes_xy / input_shape[..., ::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[..., ::-1]
    
    anchors         = np.expand_dims(anchors, axis=0)
    anchor_maxes    = anchors / 2.
    anchor_mins     = -anchor_maxes
    valid_mask   = np.equal(boxes_wh[..., 0] > 0, boxes_wh[..., 1] > 0)

    for b in range(batch):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: 
            continue

        wh          = np.expand_dims(wh, -2)
        box_maxes   = wh / 2.
        box_mins    = -box_maxes

        # Calculate Intersection area of bbox and anchor
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
                    # if coords == 'centroids':
                    #     if i == grid_shapes[l][1]:
                    #         i -= 1
                    #     elif j == grid_shapes[l][0]:
                    #         j -= 1
                    k = anchors_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')

                    if c != -1 and i < grid_shapes[l][1] and j < grid_shapes[l][0]:
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1
    return y_true
