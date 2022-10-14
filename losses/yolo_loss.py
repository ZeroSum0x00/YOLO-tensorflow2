import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from utils.bboxes import get_anchors_and_decode
from utils.iou import box_iou, iou_loss
from configs import base_config as cfg


class YOLOLoss(tf.keras.losses.Loss):
    def __init__(self, 
                 input_shape       = cfg.YOLO_TARGET_SIZE,
                 anchors           = cfg.YOLO_ANCHORS,
                 anchors_mask      = cfg.YOLO_ANCHORS_MASK,
                 num_classes       = len(cfg.OBJECT_CLASSES),
                 ignore_threshold  = cfg.YOLO_IGNORE_THRESHOLD,
                 balance           = cfg.YOLO_BALANCE_LOSS,
                 box_ratio         = cfg.YOLO_BOX_RATIO_LOSS,
                 obj_ratio         = cfg.YOLO_OBJ_RATIO_LOSS,
                 cls_ratio         = cfg.YOLO_CLS_RATIO_LOSS,
                 label_smoothing   = cfg.YOLO_LABEL_SMOOTHING,
                 iou_method        = cfg.YOLO_IOU_METHOD, 
                 focal_loss        = cfg.YOLO_FOCAL_LOSS,
                 focal_loss_ratio  = cfg.YOLO_FOCAL_LOSS_RATIO,
                 focal_alpha_ratio = cfg.YOLO_FOCAL_ALPHA_RATIO,
                 focal_gamma_ratio = cfg.YOLO_FOCAL_GAMMA_RATIO,
                 name="YOLOLoss", **kwargs):
        super(YOLOLoss, self).__init__(name=name, **kwargs)
        self.input_shape       = input_shape
        self.anchors           = np.array(anchors)
        self.anchors_mask      = np.array(anchors_mask)
        self.num_classes       = num_classes
        self.ignore_threshold  = ignore_threshold
        self.balance           = balance
        self.box_ratio         = box_ratio
        self.obj_ratio         = obj_ratio
        self.cls_ratio         = cls_ratio
        self.label_smoothing   = label_smoothing
        self.iou_method        = iou_method
        self.focal_loss        = focal_loss
        self.focal_loss_ratio  = focal_loss_ratio
        self.focal_alpha_ratio = focal_alpha_ratio
        self.focal_gamma_ratio = focal_gamma_ratio

    def _smooth_labels(self, y_true, label_smoothing):
        label_smoothing = K.constant(label_smoothing, dtype=K.float32)
        return y_true * (1.0 - label_smoothing) + label_smoothing / self.num_classes
    
    def __call__(self, y_true, y_pred):
        num_layers      = len(self.anchors_mask)

        input_shape = K.cast(self.input_shape[:-1], K.dtype(y_true[0]))

        grid_shapes = [K.cast(K.shape(y_pred[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers) if not self.iou_method]

        m = K.shape(y_pred[0])[0]
        loss    = 0

        for l in range(num_layers):
            object_mask         = y_true[l][..., 4:5]
            true_class_probs    = y_true[l][..., 5:]
            
            if self.label_smoothing:
                true_class_probs = self._smooth_labels(true_class_probs, self.label_smoothing)
                
            grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(y_pred[l], self.anchors[self.anchors_mask[l]], self.num_classes, input_shape, calc_loss=True)

            pred_box = K.concatenate([pred_xy, pred_wh])

            ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            object_mask_bool = K.cast(object_mask, 'bool')

            def loop_body(b, ignore_mask):
                true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b,...,0])
                iou = box_iou(pred_box[b], true_box)
                best_iou = K.max(iou, axis=-1)
                ignore_mask = ignore_mask.write(b, K.cast(best_iou < self.ignore_threshold, K.dtype(true_box)))
                return b+1, ignore_mask

            _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

            ignore_mask = ignore_mask.stack()

            ignore_mask = K.expand_dims(ignore_mask, -1)

            box_loss_scale  = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]
            
            if self.iou_method:
                raw_true_box    = y_true[l][..., 0:4]
                iou_value       = iou_loss(pred_box, raw_true_box, iou_method=self.iou_method)
                iou_value       = object_mask * (1 - iou_value)
                location_loss   = K.sum(iou_value)
            else:
                raw_true_xy     = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
                raw_true_wh     = K.log(y_true[l][..., 2:4] / self.anchors[self.anchors_mask[l]] * input_shape[::-1])

                raw_true_wh     = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))

                xy_loss         = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)

                wh_loss         = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[...,2:4])
                location_loss   = (K.sum(xy_loss) + K.sum(wh_loss)) * 0.1
            
            if self.focal_loss:
                confidence_loss = (object_mask * (tf.ones_like(raw_pred[...,4:5]) - tf.sigmoid(raw_pred[...,4:5])) ** self.focal_gamma_ratio * self.focal_alpha_ratio * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) + (1 - object_mask) * ignore_mask * tf.sigmoid(raw_pred[...,4:5]) ** self.focal_gamma_ratio * (1 - self.focal_alpha_ratio) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)) * self.focal_loss_ratio
            else:
                confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) + \
                            (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
                
            class_loss      = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

            num_pos         = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
            num_neg         = tf.maximum(K.sum(K.cast((1 - object_mask) * ignore_mask, tf.float32)), 1)

            location_loss   = location_loss * self.box_ratio / num_pos
            confidence_loss = K.sum(confidence_loss) * self.balance[l] * self.obj_ratio / (num_pos + num_neg)
            class_loss      = K.sum(class_loss) * self.cls_ratio / num_pos / self.num_classes

            loss            += location_loss + confidence_loss + class_loss
        return loss
