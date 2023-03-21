import os
import cv2
import numpy as np
import tensorflow as tf

from models.architectures.darknet53 import DarkNet53
from models.yolov3 import YOLOv3
from models.yolo import YOLO
from utils.post_processing import get_labels
from utils.post_processing import detect_image
from configs import general_config as cfg


if __name__ == "__main__":
    classes, num_classes = cfg.YOLO_CLASSES, cfg.NUM_CLASSES

    backbone = DarkNet53(input_shape   = cfg.YOLO_TARGET_SIZE, 
                         activation    = cfg.YOLO_BACKBONE_ACTIVATION, 
                         norm_layer    = cfg.YOLO_BACKBONE_NORMALIZATION)

    architecture = YOLOv3(backbone     = backbone,
                          num_classes  = num_classes,
                          anchors      = cfg.YOLO_ANCHORS,
                          anchor_mask  = cfg.YOLO_ANCHORS_MASK,
                          activation   = cfg.YOLO_BACKBONE_ACTIVATION, 
                          norm_layer   = cfg.YOLO_BACKBONE_NORMALIZATION,
                          max_boxes    = cfg.YOLO_MAX_BBOXES,
                          confidence   = cfg.TEST_CONFIDENCE_THRESHOLD,
                          nms_iou      = cfg.TEST_IOU_THRESHOLD,
                          gray_padding = True)
        
    model = YOLO(architecture, image_size=cfg.YOLO_TARGET_SIZE)

    load_type                          = "weights"

    weight_objects                    = [        
                                        {
                                            'path': './saved_weights/best_weights/best_weights_mAP',
                                            'stage': 'full',
                                            'custom_objects': None
                                        }
                                    ]

    if load_type and weight_objects:
        if load_type == "weights":
            model.load_weights(weight_objects)
        elif load_type == "models":
            model.load_models(weight_objects)

    image = "/home/vbpo/Desktop/TuNIT/working/Datasets/yolo_data/TH/272793944_1935029086682137_5306500212476524044_n.jpg"
    img = detect_image(image, 
                       model, 
                       cfg.YOLO_TARGET_SIZE, 
                       classes, 
                       crop=False, 
                       count=False, 
                       letterbox_image=True)
