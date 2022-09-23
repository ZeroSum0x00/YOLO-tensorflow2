import os
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from models.yolov3 import YOLOv3Encoder, YOLOv3Decoder
from models.yolo import YOLO
from losses.yolo_loss import YOLOv3Loss
from data_utils.data_flow import get_train_test_data
from callbacks.warmup_lr import AdvanceWarmUpLearningRate
from callbacks.loss_history import LossHistory
from callbacks.evaluate_map import mAPEvaluate
from utils.files import verify_folder
from utils.logger import logger
from configs import base_config as cfg


classes = cfg.OBJECT_CLASSES
yolo_anchors                        = [[ 10,  13],
                                       [ 16,  30],
                                       [ 33,  23],
                                       [ 30,  61],
                                       [ 62,  45],
                                       [ 59, 119],
                                       [116,  90],
                                       [156, 198],
                                       [373, 326]]
anchors_mask                        = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
yolo_strides                        = [8, 16, 32]
max_bboxes_per_scale                = 100
input_shape                         = (416, 416, 3)
momentum                            = 0.937

saved_path                            = './saved_weights/'

load_type                             = None
weight_objects                    = [        
                                    {
                                        'path': './saved_weights/20220912-222333/best_weights',
                                        'stage': 'full',
                                        'custom_objects': None
                                    }
                                ]

show_frequency                      = 10

batch_size = 8
learning_rate = 1e-2
lr_init = 1e-2
lr_end = lr_init * 0.01

epochs = 300


def create_folder_weights(saved_dir):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    TRAINING_TIME_PATH = saved_dir + current_time
    access_rights = 0o755
    try:
        os.makedirs(TRAINING_TIME_PATH, access_rights)
        logger.info("Successfully created the directory %s" % TRAINING_TIME_PATH)
        return verify_folder(TRAINING_TIME_PATH)
    except: 
        logger.error("Creation of the directory %s failed" % TRAINING_TIME_PATH)


def train(input_shape, yolo_anchors, batch_size, classes, anchors_mask, 
          yolo_strides, learning_rate, lr_init, lr_end, momentum, epochs, max_bboxes_per_scale, load_type, weight_objects, saved_path, show_frequency):
    TRAINING_TIME_PATH = create_folder_weights(saved_path)

    train_generator, val_generator = get_train_test_data(data_zipfile=cfg.DATA_PATH, 
                                                                         dst_dir=cfg.DESTINATION_PATH,
                                                                         classes=classes, 
                                                                         target_size=input_shape, 
                                                                         batch_size=batch_size, 
                                                                         yolo_strides=yolo_strides,
                                                                         yolo_anchors=yolo_anchors,
                                                                         anchors_mask=anchors_mask,
                                                                         max_bboxes_per_scale=max_bboxes_per_scale,
                                                                         augmentor=cfg.AUGMENTATION,
                                                                         normalizer='divide',
                                                                         data_type='voc',
                                                                         check_data=False, 
                                                                         load_memory=False,
                                                                         exclude_difficult=True,
                                                                         exclude_truncated=False)
    num_classes = len(classes)

    encoder = YOLOv3Encoder(num_classes, num_anchor=3, darknet_weight="/home/vbpo/Desktop/TuNIT/working/Yolo/yolo3-tf2/model_data/yolov3.weights")
    decoder = YOLOv3Decoder(yolo_anchors,
                            num_classes,
                            input_shape,
                            anchors_mask,
                            max_boxes=max_bboxes_per_scale,
                            confidence=0.05,
                            nms_iou=0.5,
                            letterbox_image=True)

    model = YOLO(encoder, decoder)

    if load_type and weight_objects:
        if load_type == "weights":
            model.load_weights(weight_objects)
        elif load_type == "models":
            model.load_models(weight_objects)


    loss = YOLOv3Loss(input_shape, yolo_anchors, anchors_mask, num_classes, 
                      balance     = [0.4, 1.0, 4],
                      box_ratio   = 0.05, 
                      obj_ratio   = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2),
                      cls_ratio   = 1 * (num_classes / 80))
    
    nbs             = 64
    lr_limit_max    = 5e-2 
    lr_limit_min    = 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * lr_init, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * lr_end, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    warmup_lr = AdvanceWarmUpLearningRate(lr_init=Init_lr_fit, lr_end=Min_lr_fit, epochs=epochs, result_path=TRAINING_TIME_PATH)
    eval_callback = mAPEvaluate(val_generator, 
                                input_shape=input_shape, 
                                classes=classes, 
                                result_path=TRAINING_TIME_PATH, 
                                max_bboxes_per_scale=max_bboxes_per_scale, 
                                minoverlap=0.5,
                                saved_best_map=True,
                                show_frequency=show_frequency)
    history = LossHistory(result_path=TRAINING_TIME_PATH)
    callbacks = [warmup_lr, eval_callback, history]
    

    optimizer = SGD(learning_rate=Init_lr_fit, momentum=momentum, nesterov=True)

    model.compile(optimizer=optimizer, loss=loss)

    model.fit(train_generator,
              steps_per_epoch     = train_generator.n // batch_size,
              validation_data     = val_generator,
              validation_steps    = val_generator.n // batch_size,
              epochs              = epochs,
              callbacks           = callbacks)
    model.save_weights(TRAINING_TIME_PATH + 'best_weights', save_format="tf")


if __name__ == '__main__':
    train(input_shape, yolo_anchors, batch_size, classes, 
      anchors_mask, yolo_strides, learning_rate, lr_init, lr_end, momentum,
      epochs, max_bboxes_per_scale, load_type, weight_objects, saved_path, show_frequency)
