import cv2
import math
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from data_utils.data_processing import extract_data_folder, get_data, Normalizer, preprocess_true_boxes
from utils.logger import logger
from configs import base_config as cfg
from data_utils.data_augmentation import Augmentor

def preprocess_input(image):
    image /= 255.0
    return image

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

# def get_train_test_data(data_zipfile=cfg.DATA_PATH, 
#                         dst_dir=cfg.DESTINATION_PATH,
#                         classes=cfg.OBJECT_CLASSES, 
#                         target_size=cfg.TRAIN_TARGET_SIZE, 
#                         batch_size=cfg.BATCH_SIZE, 
#                         yolo_strides=cfg.YOLO_STRIDES,
#                         yolo_anchors=cfg.YOLO_ANCHORS,
#                         anchors_mask=cfg.YOLO_ANCHORS_MASK,
#                         max_bboxes_per_scale=cfg.YOLO_MAX_BBOX_PER_SCALE,
#                         augmentor=cfg.AUGMENTATION,
#                         normalizer=cfg.NORMALIZER,
#                         data_type=cfg.DATA_TYPE,
#                         check_data=cfg.CHECK_DATA, 
#                         load_memory=cfg.LOAD_MEMORY,
#                         exclude_difficult=cfg.DATA_EXCLUDE_DIFFICULT,
#                         exclude_truncated=cfg.DATA_EXCLUDE_TRUNCATED,
#                         *args, **kwargs):
                        
#     data_folder = extract_data_folder(data_zipfile, dst_dir)
#     data_train = get_data(data_folder,
#                           classes=classes,
#                           data_type=data_type,
#                           phase='train', 
#                           check_data=check_data,
#                           load_memory=load_memory,
#                           exclude_difficult=exclude_difficult,
#                           exclude_truncated=exclude_truncated)
#     train_generator = YoloDatasets(data_train, target_size, yolo_anchors, anchors_mask, yolo_strides, batch_size, classes, train=True)

#     data_valid = get_data(data_folder,
#                           classes=classes,
#                           data_type=data_type,
#                           phase='validation', 
#                           check_data=check_data,
#                           load_memory=load_memory,
#                           exclude_difficult=exclude_difficult,
#                           exclude_truncated=exclude_truncated)
#     valid_generator = YoloDatasets(data_valid, target_size, yolo_anchors, anchors_mask, yolo_strides, batch_size, classes, train=False)
#     logger.info('Load data successfully')
#     return train_generator, valid_generator
    
class YoloDatasets(tf.keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, anchors, anchors_mask, strides, batch_size, num_classes, train):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        
        self.input_shape        = input_shape
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.strides            = strides
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.train              = train

        self.augmentor = Augmentor(target_size=input_shape, max_bboxes_per_scale=100, aug_mode=train)
        self.normalizer = Normalizer(max_bboxes_per_scale=100, mode='divide')
            
    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        image_data  = []
        box_data    = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.length
            
            sample = self.annotation_lines[i].split()

            images = cv2.imread(sample[0])
            bboxes     = np.array([np.array(list(map(int, box.split(',')))) for box in sample[1:]])

            images, bboxes = self.augmentor(images, bboxes)
            images, bboxes = self.normalizer(images, 
                                             bboxes=bboxes,
                                             target_size=self.input_shape,
                                             interpolation=cv2.INTER_NEAREST)
            image_data.append(images)
            box_data.append(bboxes)

        image_data  = np.array(image_data)
        box_data    = np.array(box_data)

        y_true      = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.anchors_mask, self.num_classes, self.strides)
        return image_data, [y_true[0], y_true[1], y_true[2]]

    def on_epoch_end(self):
        shuffle(self.annotation_lines)
        
# class YoloDatasets(tf.keras.utils.Sequence):
#     def __init__(self, dataset, input_shape, anchors, anchors_mask, strides, batch_size, classes, train):
#         self.dataset            = dataset
#         self.data_path = dataset['data_path']
#         self.dataset = dataset['data_extractor']
#         self.dataset = shuffle(self.dataset)
#         self.N = self.n = len(self.dataset)
        
#         self.input_shape        = input_shape
#         self.anchors            = anchors
#         self.anchors_mask       = anchors_mask
#         self.strides            = strides
#         self.batch_size         = batch_size
#         self.num_classes        = len(classes)
#         self.train              = train

#     def __len__(self):
#         return int(np.ceil(self.N / float(self.batch_size)))

#     def __getitem__(self, index):
#         image_data  = []
#         box_data    = []
#         for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
#             i           = i % self.N
            
#             sample = self.dataset[i]
            
#             img_path = self.data_path + sample['filename']
#             image = cv2.imread(img_path)
            
#             bboxes = np.array(sample['bboxes'])

#             image, bboxes  = augmentor(image, bboxes, self.input_shape, random=self.train)
#             processed_image = preprocess_input(np.array(image, np.float32))

#             image_data.append(processed_image)
#             box_data.append(bboxes)

#         image_data  = np.array(image_data)
#         box_data    = np.array(box_data)

#         y_true      = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.anchors_mask, self.num_classes, self.strides)
#         return image_data, [y_true[0], y_true[1], y_true[2]]

#     def on_epoch_end(self):
#         self.dataset = shuffle(self.dataset)




