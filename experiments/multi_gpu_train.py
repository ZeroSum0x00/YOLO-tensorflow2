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
from data_utils.data_flow import YoloDatasets
from callbacks.warmup_lr import AdvanceWarmUpLearningRate
from callbacks.evaluate_map import mAPEvaluate
from utils.files import verify_folder
from utils.logger import logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

train_gpu                           = [0, 1]

classes_path                        = './saved_weights/voc_classes.txt'

train_annotation_path               = '2007_train.txt'
val_annotation_path                 = '2007_val.txt'
anchors_path                        = './configs/yolo_anchors.txt'
anchors_mask                        = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

strides = [8, 16, 32]


input_shape                         = [416, 416, 3]

Init_Epoch                          = 0
Freeze_Epoch                        = 50
Freeze_batch_size                   = 16

UnFreeze_Epoch                      = 300
Unfreeze_batch_size                 = 8

Freeze_Train                        = False

Init_lr                             = 1e-2
Min_lr                              = Init_lr * 0.01

optimizer_type                      = "sgd"
momentum                            = 0.937
weight_decay                        = 5e-4

lr_decay_type                       = 'cos'

save_period                         = 10

saved_path                            = './saved_weights/'

load_type                          = None

weight_objects                    = [        
                                    {
                                        'path': './saved_weights/20220912-222333/best_weights',
                                        'stage': 'full',
                                        'custom_objects': None
                                    }
                                ]

eval_flag                           = True
eval_period                         = 5

num_workers                         = 16

batch_size = 8
learning_rate = 1e-2
lr_init = 1e-2
lr_end = lr_init * 0.01

epochs = UnFreeze_Epoch

class_names, num_classes            = get_classes(classes_path)
anchors, num_anchors                = get_anchors(anchors_path)


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


def train(input_shape, anchors, batch_size, num_classes, anchors_mask, 
          strides, learning_rate, lr_init, lr_end, epochs, train_gpu, num_workers, load_type, weight_objects, saved_path):
    TRAINING_TIME_PATH = create_folder_weights(saved_path)
    
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print('Number of devices: {}'.format(ngpus_per_node))
    
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    train_dataloader    = YoloDatasets(train_lines, input_shape, anchors, anchors_mask, strides, batch_size, num_classes, train=True)
    val_dataloader      = YoloDatasets(val_lines, input_shape, anchors, anchors_mask, strides, batch_size, num_classes, train=False)
    if ngpus_per_node > 1:
        with strategy.scope():
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

    warmup_lr = AdvanceWarmUpLearningRate(lr_init=Init_lr_fit, lr_end=Min_lr_fit, epochs=epochs)
    eval_callback = mAPEvaluate(input_shape, class_names, val_lines, TRAINING_TIME_PATH, eval_flag=True, period=10)
    callbacks = [warmup_lr, eval_callback]

    optimizer = SGD(learning_rate=Init_lr_fit, momentum=momentum, nesterov=True)
    
    if ngpus_per_node > 1:
        with strategy.scope():
            model.compile(optimizer=optimizer, loss=loss)
    

    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size

    model.fit(train_dataloader,
              steps_per_epoch     = epoch_step,
              validation_data     = val_dataloader,
              validation_steps    = epoch_step_val,
              epochs              = epochs,
              use_multiprocessing = True if num_workers > 1 else False,
              workers             = num_workers,
              callbacks           = callbacks)
    model.save_weights(TRAINING_TIME_PATH + 'best_weights', save_format="tf")


if __name__ == '__main__':
    train(input_shape, anchors, batch_size, num_classes, 
      anchors_mask, strides, learning_rate, lr_init, lr_end, 
      epochs, train_gpu, num_workers, load_type, weight_objects, saved_path)
