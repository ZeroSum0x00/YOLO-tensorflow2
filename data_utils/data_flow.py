import cv2
import math
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle

from data_utils.data_processing import extract_data_folder, get_data, Normalizer, preprocess_true_boxes
from data_utils.data_augmentation import Augmentor, EndemicAugmentor
from utils.auxiliary_processing import random_range, change_color_space
from utils.logger import logger
from configs import general_config as cfg


def get_train_test_data(data_zipfile            = cfg.DATA_PATH, 
                        annotation_dir          = cfg.DATA_ANNOTATION_PATH,
                        dst_dir                 = cfg.DATA_DESTINATION_PATH,
                        classes                 = cfg.YOLO_CLASSES, 
                        target_size             = cfg.YOLO_TARGET_SIZE, 
                        batch_size              = cfg.TRAIN_BATCH_SIZE, 
                        yolo_strides            = cfg.YOLO_STRIDES,
                        yolo_anchors            = cfg.YOLO_ANCHORS,
                        yolo_anchors_mask       = cfg.YOLO_ANCHORS_MASK,
                        max_bboxes              = cfg.YOLO_MAX_BBOXES,
                        init_epoch              = cfg.TRAIN_EPOCH_INIT,
                        end_epoch               = cfg.TRAIN_EPOCH_END,
                        color_space             = cfg.DATA_COLOR_SPACE,
                        augmentor               = cfg.DATA_AUGMENTATION,
                        endemic_augmentor       = cfg.DATA_ENDEMIC_AUGMENTATION,
                        endemic_augmentor_proba = cfg.DATA_ENDEMIC_AUGMENTATION_PROBA,
                        endemic_augmentor_ratio = cfg.DATA_ENDEMIC_AUGMENTATION_RATIO,
                        coordinate              = cfg.DATA_COORDINATE,
                        normalizer              = cfg.DATA_NORMALIZER,
                        data_type               = cfg.DATA_TYPE,
                        check_data              = cfg.CHECK_DATA, 
                        load_memory             = cfg.DATA_LOAD_MEMORY,
                        exclude_difficult       = cfg.DATA_EXCLUDE_DIFFICULT,
                        exclude_truncated       = cfg.DATA_EXCLUDE_TRUNCATED,
                        *args, **kwargs):
                        
    data_folder = extract_data_folder(data_zipfile, dst_dir)
    data_train = get_data(data_folder,
                          annotation_dir    = annotation_dir,
                          classes           = classes,
                          data_type         = data_type,
                          phase             = 'train', 
                          check_data        = check_data,
                          load_memory       = load_memory,
                          exclude_difficult = exclude_difficult,
                          exclude_truncated = exclude_truncated)
    train_generator = Data_Sequence(data_train, 
                                  target_size             = target_size, 
                                  batch_size              = batch_size, 
                                  yolo_strides            = yolo_strides,
                                  classes                 = classes,
                                  yolo_anchors            = yolo_anchors,
                                  yolo_anchors_mask       = yolo_anchors_mask,
                                  max_bboxes              = max_bboxes,
                                  color_space             = color_space,
                                  augmentor               = augmentor,
                                  endemic_augmentor       = endemic_augmentor,
                                  endemic_augmentor_proba = endemic_augmentor_proba,
                                  endemic_augmentor_ratio = endemic_augmentor_ratio,
                                  init_epoch              = init_epoch,
                                  end_epoch               = end_epoch,
                                  coordinate              = coordinate,
                                  normalizer              = normalizer,
                                  phase                   = "train",
                                  *args, **kwargs)

    data_valid = get_data(data_folder,
                          annotation_dir    = annotation_dir,
                          classes           = classes,
                          data_type         = data_type,
                          phase             = 'validation', 
                          check_data        = check_data,
                          load_memory       = load_memory,
                          exclude_difficult = exclude_difficult,
                          exclude_truncated = exclude_truncated)
    valid_generator = Data_Sequence(data_valid, 
                                  target_size             = target_size, 
                                  batch_size              = batch_size, 
                                  yolo_strides            = yolo_strides,
                                  classes                 = classes,
                                  yolo_anchors            = yolo_anchors,
                                  yolo_anchors_mask       = yolo_anchors_mask,
                                  max_bboxes              = max_bboxes,
                                  color_space             = color_space,
                                  augmentor               = augmentor,
                                  endemic_augmentor       = endemic_augmentor,
                                  endemic_augmentor_proba = endemic_augmentor_proba,
                                  endemic_augmentor_ratio = endemic_augmentor_ratio,
                                  init_epoch              = init_epoch,
                                  end_epoch               = end_epoch,
                                  coordinate              = coordinate,
                                  normalizer              = normalizer,
                                  phase                   = "valid",
                                  *args, **kwargs)

    logger.info('Load data successfully')
    return train_generator, valid_generator


class Data_Sequence(Sequence):
    def __init__(self, dataset, target_size, batch_size, yolo_strides,
                 classes, yolo_anchors, yolo_anchors_mask, max_bboxes,
                 color_space, coordinate, normalizer, augmentor,
                 endemic_augmentor, endemic_augmentor_proba,
                 endemic_augmentor_ratio, init_epoch, end_epoch,
                 phase, debug_mode=False):
        
        self.data_path = dataset['data_path']
        self.dataset   = dataset['data_extractor']
        if phase == "train":
            self.dataset   = shuffle(self.dataset)
        self.phase     = phase
        
        if isinstance(augmentor[phase], dict):
            self.use_augment_auxiliary = augmentor[phase]["auxiliary"]
            self.augmentor = Augmentor(augment_objects=augmentor[phase], 
                                       target_size=target_size, 
                                       max_bboxes=max_bboxes)
        else:
            self.use_augment_auxiliary = None
            self.augmentor = augmentor
        if isinstance(endemic_augmentor[phase], dict):
            self.use_endemic_auxiliary = endemic_augmentor[phase]["auxiliary"]
            self.endemic_augmentor = EndemicAugmentor(augment_objects=endemic_augmentor[phase], 
                                                      target_size=target_size, 
                                                      max_bboxes=max_bboxes)
        else:
            self.use_endemic_auxiliary = None
            try:
                self.endemic_augmentor = endemic_augmentor[phase]
            except:
                self.endemic_augmentor = endemic_augmentor

        self.target_size = target_size
        self.batch_size = batch_size

        self.N = self.n = len(self.dataset)
        self.normalizer = Normalizer(max_bboxes=max_bboxes, mode=normalizer)

        self.coordinate = coordinate
        self.yolo_strides = np.array(yolo_strides)
        self.num_classes = len(classes)
        self.anchors = np.array(yolo_anchors)
        self.yolo_anchors_mask = yolo_anchors_mask

        self.endemic_augmentor_proba = endemic_augmentor_proba
        self.endemic_augmentor_ratio = endemic_augmentor_ratio
        self.current_epoch = init_epoch
        self.end_epoch = end_epoch
        self.color_space = color_space
        self.debug_mode = debug_mode
        
    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def get_samples(self, sample):
        img_path = self.data_path + sample['filename']
        image = cv2.imread(img_path)
        image = change_color_space(image, 'bgr', self.color_space)
        box   = np.array(sample['bboxes'])
        return image, box
    
    def __getitem__(self, index):
        batch_image    = []
        batch_label    = []

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.N

            if self.endemic_augmentor and random_range() < self.endemic_augmentor_proba and self.current_epoch < self.end_epoch * self.endemic_augmentor_ratio:
                batch_sample = random.sample(self.dataset, 3)
                batch_sample.append(self.dataset[i])
                shuffle(batch_sample)
                images = []
                bboxes = []
                for sample in batch_sample:
                    image, box = self.get_samples(sample)
                    images.append(image)
                    bboxes.append(box)

                if self.use_endemic_auxiliary and random_range() < self.endemic_augmentor_proba:
                    auxiliary_sample = random.sample(self.dataset, 1)[0]
                    auxiliary_image, auxiliary_bboxes = self.get_samples(auxiliary_sample)
                    images, bboxes  = self.endemic_augmentor(images, bboxes, auxiliary_image, auxiliary_bboxes)
                else:
                    images, bboxes  = self.endemic_augmentor(images, bboxes)
            else:
                sample = self.dataset[i]
                img_path = self.data_path + sample['filename']
                images = cv2.imread(img_path)
                images = change_color_space(images, 'bgr', self.color_space)
                bboxes = np.array(sample['bboxes'])

            if self.use_augment_auxiliary and self.augmentor and random_range() < self.endemic_augmentor_proba and self.current_epoch < self.end_epoch * self.endemic_augmentor_ratio:
                auxiliary_sample = random.sample(self.dataset, 1)[0]
                auxiliary_image, auxiliary_bboxes = self.get_samples(auxiliary_sample)
                images, bboxes  = self.augmentor(images, bboxes, auxiliary_image, auxiliary_bboxes)
            else:
                images, bboxes = self.augmentor(images, bboxes)

            images, bboxes = self.normalizer(images, 
                                            bboxes=bboxes,
                                            target_size=self.target_size,
                                            interpolation=cv2.INTER_NEAREST)
            
            batch_image.append(images)
            batch_label.append(bboxes)
        
        batch_image = np.array(batch_image)
        batch_label = debug_boxes = np.array(batch_label)
        batch_label = preprocess_true_boxes(batch_label, self.target_size, self.anchors, self.yolo_anchors_mask, self.num_classes, self.yolo_strides, self.coordinate)
        if self.debug_mode:
            return batch_image, batch_label, debug_boxes
        else:
            return batch_image, batch_label

    def on_epoch_end(self):
        self.current_epoch += 1
        if self.phase:
            self.dataset = shuffle(self.dataset)
