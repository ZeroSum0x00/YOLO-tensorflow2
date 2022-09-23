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
# from data_utils.data_flow2 import get_train_test_data
from data_utils.data_flow2 import YoloDatasets

from callbacks.warmup_lr import AdvanceWarmUpLearningRate
from callbacks.loss_history import LossHistory
from callbacks.evaluate_map2 import mAPEvaluate
from utils.files import verify_folder
from utils.logger import logger
from configs import base_config as cfg


classes = cfg.OBJECT_CLASSES
anchors                             = cfg.YOLO_ANCHORS
anchors_mask                        = cfg.YOLO_ANCHORS_MASK

strides = [8, 16, 32]


input_shape                         = cfg.TRAIN_TARGET_SIZE

Init_Epoch                          = 0
Freeze_Epoch                        = 50
Freeze_batch_size                   = 16

UnFreeze_Epoch                      = 300
Unfreeze_batch_size                 = 8


Init_lr                             = 1e-2
Min_lr                              = Init_lr * 0.01

optimizer_type                      = "sgd"
momentum                            = 0.937
weight_decay                        = 5e-4

lr_decay_type                       = 'cos'

save_period                         = 10

saved_path                            = './saved_weights/'

load_type                             = None

weight_objects                    = [        
                                    {
                                        'path': '/home/vbpo/Desktop/TuNIT/working/Yolo/yolo-project2/saved_weights/checkpoints/last_epoch_weights',
                                        'stage': 'full',
                                        'custom_objects': None
                                    }
                                ]


eval_period                         = 5

num_workers                         = 1

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


def train(input_shape, anchors, batch_size, classes, anchors_mask, 
          strides, learning_rate, lr_init, lr_end, epochs, load_type, weight_objects, saved_path):
    TRAINING_TIME_PATH = create_folder_weights(saved_path)
    
    train_annotation_path               = '/home/vbpo/Desktop/TuNIT/working/Yolo/yolo-project/datasets/VOC2017/2007_train.txt'
    val_annotation_path                 = '/home/vbpo/Desktop/TuNIT/working/Yolo/yolo-project/datasets/VOC2017/2007_val.txt'
#     train_annotation_path               = '/home/vbpo/Desktop/TuNIT/working/Yolo/yolo-project2/datasets/VOCTiny/voc_tiny/2007_train.txt'
#     val_annotation_path                 = '/home/vbpo/Desktop/TuNIT/working/Yolo/yolo-project2/datasets/VOCTiny/voc_tiny/2007_val.txt'

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    num_classes = len(classes)
    train_generator    = YoloDatasets(train_lines, input_shape, anchors, anchors_mask, strides, batch_size, num_classes, train='train')
    val_generator      = YoloDatasets(val_lines, input_shape, anchors, anchors_mask, strides, batch_size, num_classes, train='test')

    
#     train_generator, val_generator = get_train_test_data(data_zipfile=cfg.DATA_PATH, 
#                                                                          dst_dir=cfg.DESTINATION_PATH,
#                                                                          classes=cfg.OBJECT_CLASSES, 
#                                                                          target_size=cfg.TRAIN_TARGET_SIZE, 
#                                                                          batch_size=cfg.BATCH_SIZE, 
#                                                                          yolo_strides=cfg.YOLO_STRIDES,
#                                                                          yolo_anchors=cfg.YOLO_ANCHORS,
#                                                                          anchors_mask=cfg.YOLO_ANCHORS_MASK,
#                                                                          max_bboxes_per_scale=cfg.YOLO_MAX_BBOX_PER_SCALE,
#                                                                          augmentor=cfg.AUGMENTATION,
#                                                                          normalizer=cfg.NORMALIZER,
#                                                                          data_type=cfg.DATA_TYPE,
#                                                                          check_data=cfg.CHECK_DATA, 
#                                                                          load_memory=cfg.LOAD_MEMORY,
#                                                                          exclude_difficult=cfg.DATA_EXCLUDE_DIFFICULT,
#                                                                          exclude_truncated=cfg.DATA_EXCLUDE_TRUNCATED)
    num_classes = len(classes)

    encoder = YOLOv3Encoder(num_classes, num_anchor=3, darknet_weight="/home/vbpo/Desktop/TuNIT/working/Yolo/yolo3-tf2/model_data/yolov3.weights")
    decoder = YOLOv3Decoder(anchors,
                            num_classes,
                            input_shape,
                            anchors_mask,
                            max_boxes=100,
                            confidence=0.05,
                            nms_iou=0.5,
                            letterbox_image=True)

    model = YOLO(encoder, decoder)

    if load_type and weight_objects:
        if load_type == "weights":
            model.load_weights(weight_objects)
        elif load_type == "models":
            model.load_models(weight_objects)


    loss = YOLOv3Loss(input_shape, anchors, anchors_mask, num_classes, 
                      balance     = [0.4, 1.0, 4],
                      box_ratio   = 0.05, 
                      obj_ratio   = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2),
                      cls_ratio   = 1 * (num_classes / 80))
    
    nbs             = 64
    lr_limit_max    = 5e-2 
    lr_limit_min    = 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    warmup_lr = AdvanceWarmUpLearningRate(lr_init=Init_lr_fit, lr_end=Min_lr_fit, epochs=epochs, result_path=TRAINING_TIME_PATH)
#     eval_callback = mAPEvaluate(val_generator, 
#                                 input_shape=cfg.TRAIN_TARGET_SIZE, 
#                                 classes=cfg.OBJECT_CLASSES, 
#                                 result_path=TRAINING_TIME_PATH, 
#                                 normalizer=cfg.NORMALIZER, 
#                                 max_bboxes_per_scale=cfg.YOLO_MAX_BBOX_PER_SCALE, 
#                                 minoverlap=0.5,
#                                 saved_best_map=True,
#                                 show_frequency=10)
    eval_callback = mAPEvaluate(input_shape, cfg.OBJECT_CLASSES, val_lines, TRAINING_TIME_PATH, eval_flag=True, period=10)
    history = LossHistory(result_path=TRAINING_TIME_PATH)
    callbacks = [warmup_lr, eval_callback, history]
    

    optimizer = SGD(learning_rate=Init_lr_fit, momentum=momentum, nesterov=True)

    model.compile(optimizer=optimizer, loss=loss)

    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size

    model.fit(train_generator,
              steps_per_epoch     = epoch_step,
              validation_data     = val_generator,
              validation_steps    = epoch_step_val,
              epochs              = epochs,
              callbacks           = callbacks)
    model.save_weights(TRAINING_TIME_PATH + 'best_weights', save_format="tf")


if __name__ == '__main__':
    train(input_shape, anchors, batch_size, classes, 
      anchors_mask, strides, learning_rate, lr_init, lr_end, 
      epochs, load_type, weight_objects, saved_path)
