import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger

from models.yolov3 import YOLOv3Encoder, YOLOv3Decoder
from models.architectures.darknet53 import DarkNet53
from models.yolo import YOLO
from losses.yolo_loss import YOLOLoss
from data_utils.data_flow import get_train_test_data
from callbacks.warmup_lr import AdvanceWarmUpLearningRate
from callbacks.loss_history import LossHistory
from callbacks.evaluate_map import mAPEvaluate
from utils.logger import logger
from utils.train_processing import create_folder_weights
from configs import base_config as cfg



def train(data_path                   = cfg.DATA_PATH,
          data_anno_path              = cfg.DATA_ANNOTATION_PATH,
          data_dst_path               = cfg.DATA_DESTINATION_PATH,
          data_normalizer             = cfg.DATA_NORMALIZER,
          data_augmentation           = cfg.DATA_AUGMENTATION,
          data_endemic_augmentor      = cfg.DATA_ENDEMIC_AUGMENTATION,
          endemic_augmentor_proba     = cfg.DATA_ENDEMIC_AUGMENTATION_PROBA,
          endemic_augmentor_ratio     = cfg.DATA_ENDEMIC_AUGMENTATION_RATIO,
          data_type                   = cfg.DATA_TYPE,
          check_data                  = cfg.CHECK_DATA,
          load_memory                 = cfg.DATA_LOAD_MEMORY,
          exclude_difficult           = cfg.DATA_EXCLUDE_DIFFICULT,
          exclude_truncated           = cfg.DATA_EXCLUDE_TRUNCATED,
          classes                     = cfg.YOLO_CLASSES,
          yolo_activation             = cfg.YOLO_ACTIVATION,
          yolo_normalization          = cfg.YOLO_NORMALIZATION,
          yolo_backbone_activation    = cfg.YOLO_BACKBONE_ACTIVATION,
          yolo_backbone_normalization = cfg.YOLO_BACKBONE_NORMALIZATION,
          yolo_backbone_weight        = cfg.YOLO_BACKBONE_WEIGHTS,
          input_shape                 = cfg.YOLO_TARGET_SIZE,
          yolo_anchors                = cfg.YOLO_ANCHORS,
          yolo_anchors_mask           = cfg.YOLO_ANCHORS_MASK,
          yolo_strides                = cfg.YOLO_STRIDES,
          max_bboxes                  = cfg.YOLO_MAX_BBOXES,
          ignore_threshold            = cfg.YOLO_IGNORE_THRESHOLD,
          balance_loss                = cfg.YOLO_BALANCE_LOSS,
          box_ratio_loss              = cfg.YOLO_BOX_RATIO_LOSS,
          obj_ratio_loss              = cfg.YOLO_OBJ_RATIO_LOSS,
          cls_ratio_loss              = cfg.YOLO_CLS_RATIO_LOSS,
          label_smoothing             = cfg.YOLO_LABEL_SMOOTHING,
          iou_method                  = cfg.YOLO_IOU_METHOD,
          focal_loss                  = cfg.YOLO_FOCAL_LOSS,
          focal_loss_ratio            = cfg.YOLO_FOCAL_LOSS_RATIO,
          focal_alpha_ratio           = cfg.YOLO_FOCAL_ALPHA_RATIO,
          focal_gamma_ratio           = cfg.YOLO_FOCAL_GAMMA_RATIO,
          batch_size                  = cfg.TRAIN_BATCH_SIZE,
          init_epoch                  = cfg.TRAIN_EPOCH_INIT,
          end_epoch                   = cfg.TRAIN_EPOCH_END,
          momentum                    = cfg.TRAIN_MOMENTUM,
          nesterov                    = cfg.TRAIN_NESTEROV,
          lr_init                     = cfg.TRAIN_LR_INIT,
          lr_end                      = cfg.TRAIN_LR_END,
          weight_type                 = cfg.TRAIN_WEIGHT_TYPE,
          weight_objects              = cfg.TRAIN_WEIGHT_OBJECTS,
          show_frequency              = cfg.TRAIN_SHOW_FREQUENCY,
          saved_weight_frequency      = cfg.TRAIN_SAVE_WEIGHT_FREQUENCY,
          saved_path                  = cfg.TRAIN_SAVED_PATH,
          confidence_threshold        = cfg.TEST_CONFIDENCE_THRESHOLD,
          iou_threshold               = cfg.TEST_IOU_THRESHOLD,
          min_overlap                 = cfg.TEST_MIN_OVERLAP):
     
    TRAINING_TIME_PATH = create_folder_weights(saved_path)
    
    num_classes = len(classes)
    
    train_generator, val_generator = get_train_test_data(data_zipfile            = data_path, 
                                                         dst_dir                 = data_dst_path,
                                                         classes                 = classes, 
                                                         target_size             = input_shape, 
                                                         batch_size              = batch_size, 
                                                         yolo_strides            = yolo_strides,
                                                         yolo_anchors            = yolo_anchors,
                                                         yolo_anchors_mask       = yolo_anchors_mask,
                                                         max_bboxes              = max_bboxes,
                                                         init_epoch              = init_epoch,
                                                         end_epoch               = end_epoch,
                                                         augmentor               = data_augmentation,
                                                         endemic_augmentor       = data_endemic_augmentor,
                                                         endemic_augmentor_proba = endemic_augmentor_proba,
                                                         endemic_augmentor_ratio = endemic_augmentor_ratio,
                                                         normalizer              = data_normalizer,
                                                         data_type               = data_type,
                                                         check_data              = check_data, 
                                                         load_memory             = load_memory,
                                                         exclude_difficult       = exclude_difficult,
                                                         exclude_truncated       = exclude_truncated)
    
    backbone = DarkNet53(input_shape   = input_shape, 
                         activation    = yolo_backbone_activation, 
                         norm_layer    = yolo_backbone_normalization, 
                         model_weights = yolo_backbone_weight)
    
    encoder = YOLOv3Encoder(backbone    = backbone,
                            num_classes = num_classes, 
                            num_anchor  = 3,
                            activation  = yolo_activation,
                            norm_layer  = yolo_normalization)
    
    decoder = YOLOv3Decoder(anchors     = yolo_anchors,
                            num_classes = num_classes,
                            input_size  = input_shape,
                            anchor_mask = yolo_anchors_mask,
                            max_boxes   = max_bboxes,
                            confidence  = confidence_threshold,
                            nms_iou     = iou_threshold,
                            letterbox_image=True)

    model = YOLO(encoder, decoder)

    if weight_type and weight_objects:
        if weight_type == "weights":
            model.load_weights(weight_objects)
        elif weight_type == "models":
            model.load_models(weight_objects)


    loss = YOLOLoss(input_shape       = input_shape, 
                    anchors           = yolo_anchors, 
                    anchors_mask      = yolo_anchors_mask, 
                    num_classes       = num_classes, 
                    ignore_threshold  = ignore_threshold,
                    balance           = balance_loss,
                    box_ratio         = box_ratio_loss, 
                    obj_ratio         = obj_ratio_loss,
                    cls_ratio         = cls_ratio_loss,
                    label_smoothing   = label_smoothing,
                    iou_method        = iou_method,
                    focal_loss        = focal_loss,
                    focal_loss_ratio  = focal_loss_ratio,
                    focal_alpha_ratio = focal_alpha_ratio,
                    focal_gamma_ratio = focal_gamma_ratio)
    
    nbs             = 64
    lr_limit_max    = 5e-2 
    lr_limit_min    = 5e-4
    Init_lr_fit     = min(max(batch_size / nbs * lr_init, lr_limit_min), lr_limit_max)
    Min_lr_fit      = min(max(batch_size / nbs * lr_end, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    eval_callback = mAPEvaluate(val_generator, 
                                input_shape    = input_shape, 
                                classes        = classes, 
                                result_path    = TRAINING_TIME_PATH, 
                                max_bboxes     = max_bboxes, 
                                minoverlap     = min_overlap,
                                saved_best_map = True,
                                show_frequency = show_frequency)
    
    history = LossHistory(result_path=TRAINING_TIME_PATH)
    
    checkpoint = ModelCheckpoint(TRAINING_TIME_PATH + 'checkpoint_{epoch:04d}/saved_yolo_weights', 
                                 monitor='val_total_loss',
                                 verbose=1, 
                                 save_weights_only=True,
                                 save_freq="epoch",
                                 period=saved_weight_frequency)
    
    logger = CSVLogger(TRAINING_TIME_PATH + 'train_history.csv', separator=",", append=True)

    warmup_lr = AdvanceWarmUpLearningRate(lr_init=Init_lr_fit, lr_end=Min_lr_fit, epochs=end_epoch, result_path=TRAINING_TIME_PATH)
    
    callbacks = [eval_callback, history, checkpoint, logger, warmup_lr]
    

    optimizer = SGD(learning_rate=Init_lr_fit, momentum=momentum, nesterov=nesterov)

    model.compile(optimizer=optimizer, loss=loss)

    model.fit(train_generator,
              steps_per_epoch     = train_generator.n // batch_size,
              validation_data     = val_generator,
              validation_steps    = val_generator.n // batch_size,
              epochs              = end_epoch,
              initial_epoch       = init_epoch,
              callbacks           = callbacks)
    model.save_weights(TRAINING_TIME_PATH + 'best_weights', save_format="tf")
          

if __name__ == '__main__':
    train()
