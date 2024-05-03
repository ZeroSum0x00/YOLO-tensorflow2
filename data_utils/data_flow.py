import cv2
import random
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from augmenter import build_augmenter
from data_utils.data_processing import get_data, Normalizer, preprocess_true_boxes
from data_utils.data_augmentation import Augmentor, EndemicAugmentor
from utils.post_processing import get_labels
from utils.auxiliary_processing import random_range, change_color_space
from utils.constant import *
from utils.logger import logger



def get_train_test_data(data_dirs, 
                        annotation_dir          = None,
                        classes                 = 80, 
                        target_size             = [416, 416, 3], 
                        batch_size              = 4, 
                        yolo_strides            = yolo_strides,
                        yolo_anchors            = yolo_anchors,
                        yolo_anchors_mask       = yolo_anchor_masks,
                        max_bboxes              = 100,
                        init_epoch              = 0,
                        end_epoch               = 300,
                        color_space             = 'RGB',
                        augmentor               = None,
                        endemic_augmentor       = None,
                        endemic_augmentor_proba = 0.5,
                        endemic_augmentor_ratio = 0.5,
                        coordinate              = 'corners',
                        normalizer              = 'divide',
                        mean_norm               = None,
                        std_norm                = None,
                        data_type               = 'voc',
                        check_data              = False, 
                        load_memory             = False,
                        exclude_difficult       = True,
                        exclude_truncated       = False,
                        dataloader_mode         = 0,
                        *args, **kwargs):
                            
    """
        dataloader_mode = 0:   train - validation - test
        dataloader_mode = 1:   train - validation
        dataloader_mode = 2:   train
    """
                            
    if isinstance(classes, str):
        classes, _ = get_labels(classes)

    data_train = get_data(data_dirs,
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
                                    coordinate              = coordinate,
                                    normalizer              = normalizer,
                                    mean_norm               = mean_norm,
                                    std_norm                = std_norm,
                                    augmentor               = augmentor,
                                    endemic_augmentor       = endemic_augmentor,
                                    endemic_augmentor_proba = endemic_augmentor_proba,
                                    endemic_augmentor_ratio = endemic_augmentor_ratio,
                                    init_epoch              = init_epoch,
                                    end_epoch               = end_epoch,
                                    phase                   = "train",
                                    *args, **kwargs)
                            
    if dataloader_mode != 2:
        data_valid = get_data(data_dirs,
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
                                        coordinate              = coordinate,
                                        normalizer              = normalizer,
                                        mean_norm               = mean_norm,
                                        std_norm                = std_norm,
                                        augmentor               = augmentor,
                                        endemic_augmentor       = endemic_augmentor,
                                        endemic_augmentor_proba = endemic_augmentor_proba,
                                        endemic_augmentor_ratio = endemic_augmentor_ratio,
                                        init_epoch              = init_epoch,
                                        end_epoch               = end_epoch,
                                        phase                   = "valid",
                                        *args, **kwargs)
    else:
        valid_generator = None

    if dataloader_mode == 1:
        data_test  = get_data(data_dirs,
                              annotation_dir    = annotation_dir,
                              classes           = classes,
                              data_type         = data_type,
                              phase             = 'test', 
                              check_data        = check_data,
                              load_memory       = load_memory,
                              exclude_difficult = exclude_difficult,
                              exclude_truncated = exclude_truncated)
        
        test_generator  = Data_Sequence(data_valid, 
                                        target_size             = target_size, 
                                        batch_size              = batch_size, 
                                        yolo_strides            = yolo_strides,
                                        classes                 = classes,
                                        yolo_anchors            = yolo_anchors,
                                        yolo_anchors_mask       = yolo_anchors_mask,
                                        max_bboxes              = max_bboxes,
                                        color_space             = color_space,
                                        coordinate              = coordinate,
                                        normalizer              = normalizer,
                                        mean_norm               = mean_norm,
                                        std_norm                = std_norm,
                                        augmentor               = augmentor,
                                        endemic_augmentor       = endemic_augmentor,
                                        endemic_augmentor_proba = endemic_augmentor_proba,
                                        endemic_augmentor_ratio = endemic_augmentor_ratio,
                                        init_epoch              = init_epoch,
                                        end_epoch               = end_epoch,
                                        phase                   = "test",
                                        *args, **kwargs)
    else:
        test_generator = None
        
    logger.info('Load data successfully')
    return train_generator, valid_generator, test_generator


class Data_Sequence(tf.keras.utils.Sequence):
    def __init__(self, 
                 dataset,
                 target_size,
                 batch_size,
                 yolo_strides,
                 classes,
                 yolo_anchors,
                 yolo_anchors_mask,
                 max_bboxes,
                 color_space='RGB',
                 coordinate="corners",
                 normalizer=None,
                 mean_norm=None, 
                 std_norm=None,
                 augmentor=None,
                 endemic_augmentor=None, 
                 endemic_augmentor_proba=0.,
                 endemic_augmentor_ratio=0.,
                 init_epoch=0,
                 end_epoch=100,
                 phase='train',
                 debug_mode=False):
        
        self.data_path = dataset['data_path']
        self.dataset   = dataset['data_extractor']
        if phase == "train":
            self.dataset   = shuffle(self.dataset)
        self.phase     = phase

        additional_config = {'ResizePadded': {'target_size': target_size, 'coords': coordinate, 'max_bboxes': max_bboxes},
                             'RandomFlip': {'coords': coordinate},
                             'LightIntensityChange': {'color_space': color_space},
                             'Mosaic': {'target_size': target_size, 'coords': coordinate, 'max_bboxes': max_bboxes}}

        if isinstance(augmentor[phase], dict):
            self.use_augment_auxiliary = augmentor[phase]["auxiliary"]
            self.augmentor = Augmentor(augment_objects=build_augmenter(augmentor[phase],
                                                                       additional_config=additional_config))
        else:
            self.use_augment_auxiliary = None
            self.augmentor = augmentor[phase]

        if isinstance(endemic_augmentor[phase], dict):
            self.use_endemic_auxiliary = endemic_augmentor[phase]["auxiliary"]
            self.endemic_augmentor = EndemicAugmentor(augment_objects=build_augmenter(endemic_augmentor[phase],
                                                                                      additional_config=additional_config))
        else:
            self.use_endemic_auxiliary = None
            self.endemic_augmentor = endemic_augmentor[phase]

        self.target_size = target_size
        self.batch_size = batch_size

        self.N = self.n = len(self.dataset)
        self.normalizer = Normalizer(normalizer,
                                     mean=mean_norm, 
                                     std=std_norm, 
                                     max_bboxes=max_bboxes)

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

            if self.augmentor is not None:
                if self.use_augment_auxiliary and random_range() < self.endemic_augmentor_proba and self.current_epoch < self.end_epoch * self.endemic_augmentor_ratio:
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
        # if self.phase == "train":
        self.dataset = shuffle(self.dataset)
