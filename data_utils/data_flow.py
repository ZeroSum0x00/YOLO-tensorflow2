import cv2
import math
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle

from data_utils.data_processing import extract_data_folder, get_data, Normalizer, preprocess_true_boxes
from data_utils.data_augmentation import Augmentor
from utils.logger import logger
from configs import base_config as cfg


def get_train_test_data(data_zipfile      = cfg.DATA_PATH, 
                        dst_dir           = cfg.DATA_DESTINATION_PATH,
                        classes           = cfg.OBJECT_CLASSES, 
                        target_size       = cfg.YOLO_TARGET_SIZE, 
                        batch_size        = cfg.TRAIN_BATCH_SIZE, 
                        yolo_strides      = cfg.YOLO_STRIDES,
                        yolo_anchors      = cfg.YOLO_ANCHORS,
                        yolo_anchors_mask = cfg.YOLO_ANCHORS_MASK,
                        max_bboxes        = cfg.YOLO_MAX_BBOXES,
                        augmentor         = cfg.DATA_AUGMENTATION,
                        normalizer        = cfg.DATA_NORMALIZER,
                        data_type         = cfg.DATA_TYPE,
                        check_data        = cfg.CHECK_DATA, 
                        load_memory       = cfg.DATA_LOAD_MEMORY,
                        exclude_difficult = cfg.DATA_EXCLUDE_DIFFICULT,
                        exclude_truncated = cfg.DATA_EXCLUDE_TRUNCATED,
                        *args, **kwargs):
                        
    data_folder = extract_data_folder(data_zipfile, dst_dir)
    data_train = get_data(data_folder,
                          classes           = classes,
                          data_type         = data_type,
                          phase             = 'train', 
                          check_data        = check_data,
                          load_memory       = load_memory,
                          exclude_difficult = exclude_difficult,
                          exclude_truncated = exclude_truncated)
    train_generator = Train_Data_Sequence(data_train, 
                                          target_size       = target_size, 
                                          batch_size        = batch_size, 
                                          yolo_strides      = yolo_strides,
                                          classes           = classes,
                                          yolo_anchors      = yolo_anchors,
                                          yolo_anchors_mask = yolo_anchors_mask,
                                          max_bboxes        = max_bboxes,
                                          augmentor='train',
                                          normalizer=normalizer,
                                          *args, **kwargs)

    data_valid = get_data(data_folder,
                          classes           = classes,
                          data_type         = data_type,
                          phase             = 'validation', 
                          check_data        = check_data,
                          load_memory       = load_memory,
                          exclude_difficult = exclude_difficult,
                          exclude_truncated = exclude_truncated)
    valid_generator = Valid_Data_Sequence(data_valid, 
                                          target_size       = target_size, 
                                          batch_size        = batch_size, 
                                          yolo_strides      = yolo_strides,
                                          classes           = classes,
                                          yolo_anchors      = yolo_anchors,
                                          yolo_anchors_mask = yolo_anchors_mask,
                                          max_bboxes        = max_bboxes,
                                          augmentor         = 'validation',
                                          normalizer        = normalizer,
                                          *args, **kwargs)
    
#     data_test = get_data(data_folder,
#                           classes=classes,
#                           data_type=data_type,
#                           phase='test', 
#                           check_data=check_data,
#                           load_memory=load_memory,
#                           exclude_difficult=exclude_difficult,
#                           exclude_truncated=exclude_truncated)
#     test_generator = Test_Data_Sequence(data_test, 
#                                         target_size=target_size, 
#                                         batch_size=batch_size, 
#                                         yolo_strides=yolo_strides,
#                                         classes=classes,
#                                         yolo_anchors=yolo_anchors,
#                                         yolo_anchors_mask=yolo_anchors_mask,
#                                         max_bboxes_per_scale=max_bboxes_per_scale,
#                                         augmentor=augmentor['test'],
#                                         normalizer=normalizer,
#                                         *args, **kwargs)

    logger.info('Load data successfully')
    return train_generator, valid_generator


class Train_Data_Sequence(Sequence):
    def __init__(self, 
                 dataset, 
                 target_size       = cfg.YOLO_TARGET_SIZE, 
                 batch_size        = cfg.TRAIN_BATCH_SIZE, 
                 yolo_strides      = cfg.YOLO_STRIDES,
                 classes           = cfg.OBJECT_CLASSES,
                 yolo_anchors      = cfg.YOLO_ANCHORS,
                 yolo_anchors_mask = cfg.YOLO_ANCHORS_MASK,
                 max_bboxes        = cfg.YOLO_MAX_BBOXES,
                 augmentor         = cfg.DATA_AUGMENTATION['train'],
                 normalizer        = cfg.DATA_NORMALIZER):
        
        self.data_path = dataset['data_path']
        self.dataset = dataset['data_extractor']
        
        if isinstance(augmentor, str):
            self.augmentor = Augmentor(target_size=target_size, max_bboxes=max_bboxes, aug_mode=augmentor)
        else:
            self.augmentor = augmentor

        self.target_size = target_size
        self.batch_size = batch_size

        self.dataset = shuffle(self.dataset)

        self.N = self.n = len(self.dataset)
        self.normalizer = Normalizer(max_bboxes=max_bboxes, mode=normalizer)

        self.yolo_strides = np.array(yolo_strides)
        self.num_classes = len(classes)
        self.anchors = np.array(yolo_anchors)
        self.yolo_anchors_mask = yolo_anchors_mask

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, index):
        batch_image = np.zeros((self.batch_size, *self.target_size), dtype=np.float32)
#         batch_image    = []
        batch_label    = []

        for i in range(self.batch_size):          
            index = min((index * self.batch_size) + i, self.N)
            if index < self.N:
                sample = self.dataset[index]
            else:
                sample = random.choice(self.dataset)
#         for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
#             i           = i % self.N
#             sample = self.dataset[i]
            img_path = self.data_path + sample['filename']
            image = cv2.imread(img_path)
            bboxes = np.array(sample['bboxes'])

            if self.augmentor is not None:
                image, bboxes = self.augmentor(image, bboxes)

            image, bboxes = self.normalizer(image, 
                                            bboxes=bboxes,
                                            target_size=self.target_size,
                                            interpolation=cv2.INTER_NEAREST)
            batch_image[i] = image
#             batch_image.append(image)
            batch_label.append(bboxes)
        
#         batch_image = np.array(batch_image)
        batch_label = np.array(batch_label)
        batch_label = preprocess_true_boxes(batch_label, self.target_size, self.anchors, self.yolo_anchors_mask, self.num_classes, self.yolo_strides)
        return batch_image, batch_label
    
    def on_epoch_end(self):
        self.dataset = shuffle(self.dataset)
    
    
class Valid_Data_Sequence(Sequence):
    def __init__(self, 
                 dataset, 
                 target_size       = cfg.YOLO_TARGET_SIZE, 
                 batch_size        = cfg.TRAIN_BATCH_SIZE, 
                 yolo_strides      = cfg.YOLO_STRIDES,
                 classes           = cfg.OBJECT_CLASSES,
                 yolo_anchors      = cfg.YOLO_ANCHORS,
                 yolo_anchors_mask = cfg.YOLO_ANCHORS_MASK,
                 max_bboxes        = cfg.YOLO_MAX_BBOXES,
                 augmentor         = cfg.DATA_AUGMENTATION['validation'],
                 normalizer        = cfg.DATA_NORMALIZER):
        
        self.data_path = dataset['data_path']
        self.dataset = dataset['data_extractor']
        
        if isinstance(augmentor, str):
            self.augmentor = Augmentor(max_bboxes=max_bboxes, aug_mode=augmentor)
        else:
            self.augmentor = augmentor

        self.target_size = target_size
        self.batch_size = batch_size

        self.N = self.n = len(self.dataset)
        self.normalizer = Normalizer(max_bboxes=max_bboxes, mode=normalizer)

        self.yolo_strides = np.array(yolo_strides)
        self.num_classes = len(classes)
        self.anchors = np.array(yolo_anchors)
        self.yolo_anchors_mask = yolo_anchors_mask

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, index):
        batch_image = np.zeros((self.batch_size, *self.target_size), dtype=np.float32)
#         batch_image    = []
        batch_label    = []

        for i in range(self.batch_size):          
            index = min((index * self.batch_size) + i, self.N)
            if index < self.N:
                sample = self.dataset[index]
            else:
                sample = random.choice(self.dataset)
#         for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
#             i           = i % self.N
#             sample = self.dataset[i]
            img_path = self.data_path + sample['filename']
            image = cv2.imread(img_path)
            bboxes = np.array(sample['bboxes'])

            if self.augmentor is not None:
                image, bboxes = self.augmentor(image, bboxes)

            image, bboxes = self.normalizer(image, 
                                            bboxes=bboxes,
                                            target_size=self.target_size,
                                            interpolation=cv2.INTER_NEAREST)
            batch_image[i] = image
#             batch_image.append(image)
            batch_label.append(bboxes)

#         batch_image  = np.array(batch_image)
        batch_label = np.array(batch_label)
        batch_label = preprocess_true_boxes(batch_label, self.target_size, self.anchors, self.yolo_anchors_mask, self.num_classes, self.yolo_strides)
        return batch_image, batch_label
    
    def on_epoch_end(self):
        self.dataset = shuffle(self.dataset)


class Test_Data_Sequence(Sequence):
    def __init__(self, 
                 dataset, 
                 target_size       = cfg.YOLO_TARGET_SIZE, 
                 batch_size        = cfg.TRAIN_BATCH_SIZE, 
                 yolo_strides      = cfg.YOLO_STRIDES,
                 classes           = cfg.OBJECT_CLASSES,
                 yolo_anchors      = cfg.YOLO_ANCHORS,
                 yolo_anchors_mask = cfg.YOLO_ANCHORS_MASK,
                 max_bboxes        = cfg.YOLO_MAX_BBOXES,
                 augmentor         = cfg.DATA_AUGMENTATION['test'],
                 normalizer        = cfg.DATA_NORMALIZER):
        
        self.data_path = dataset['data_path']
        self.dataset = dataset['data_extractor']
        
        if isinstance(augmentor, str):
            self.augmentor = Augmentor(max_bboxes=max_bboxes, aug_mode=augmentor)
        else:
            self.augmentor = augmentor

        self.target_size = target_size
        self.batch_size = batch_size

        self.N = self.n = len(self.dataset)
        self.normalizer = Normalizer(max_bboxes=max_bboxes, mode=normalizer)

        self.yolo_strides = np.array(yolo_strides)
        self.num_classes = len(classes)
        self.anchors = np.array(yolo_anchors)
        self.yolo_anchors_mask = yolo_anchors_mask

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_image = np.zeros((self.batch_size, *self.target_size), dtype=np.float32)
        batch_label    = []

        for i in range(self.batch_size):          
            index = min((idx * self.batch_size) + i, self.N)
            if index < self.N:
                sample = self.dataset[index]
            else:
                sample = random.choice(self.dataset)

            img_path = self.data_path + sample['filename']
            image = cv2.imread(img_path)
            bboxes = np.array(sample['bboxes'])

            if self.augmentor is not None:
                image, bboxes = self.augmentor(image, bboxes)

            image, bboxes = self.normalizer(image, 
                                            bboxes=bboxes,
                                            target_size=self.target_size,
                                            interpolation=cv2.INTER_NEAREST)
            batch_image[i] = image
            batch_label.append(bboxes)

        batch_label = np.array(batch_label)
        batch_label = preprocess_true_boxes(batch_label, self.target_size, self.anchors, self.yolo_anchors_mask, self.num_classes, self.yolo_strides)
        return batch_image, batch_label

    def on_epoch_end(self):
        self.dataset = shuffle(self.dataset)
