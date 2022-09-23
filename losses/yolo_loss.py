import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from utils.bboxes import get_anchors_and_decode
from utils.iou import box_iou, box_ciou


class YOLOv3Loss(tf.keras.losses.Loss):
    def __init__(self, 
                 input_shape, 
                 anchors, 
                 anchors_mask, 
                 num_classes,
                 ignore_thresh   = 0.5,
                 balance         = [0.4, 1.0, 4], 
                 box_ratio       = 0.05, 
                 obj_ratio       = 1, 
                 cls_ratio       = 0.5 / 4, 
                 ciou_flag       = True, 
                 print_loss      = False,
                 name="YOLOv3Loss", **kwargs):
        super(YOLOv3Loss, self).__init__(name=name, **kwargs)
        self.input_shape = input_shape
        self.anchors = np.array(anchors)
        self.anchors_mask = np.array(anchors_mask)
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.balance = balance
        self.box_ratio = box_ratio
        self.obj_ratio = obj_ratio
        self.cls_ratio = cls_ratio
        self.ciou_flag = ciou_flag
        self.print_loss = print_loss

    def __call__(self, y_true, y_pred):
        num_layers      = len(self.anchors_mask)

        input_shape = K.cast(self.input_shape[:-1], K.dtype(y_true[0]))

        grid_shapes = [K.cast(K.shape(y_pred[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]

        m = K.shape(y_pred[0])[0]

        loss    = 0

        for l in range(num_layers):

            object_mask         = y_true[l][..., 4:5]

            true_class_probs    = y_true[l][..., 5:]

            grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(y_pred[l], self.anchors[self.anchors_mask[l]], self.num_classes, input_shape, calc_loss=True)

            pred_box = K.concatenate([pred_xy, pred_wh])

            ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
            object_mask_bool = K.cast(object_mask, 'bool')
            ignore_temp = tf.constant([0.])

            def loop_body(b, ignore_mask):

                true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])

                iou = box_iou(pred_box[b], true_box)

                best_iou = K.max(iou, axis=-1)

                ignore_mask = ignore_mask.write(b, K.cast(best_iou < self.ignore_thresh, K.dtype(true_box)))
                return b+1, ignore_mask

            _, ignore_mask = tf.while_loop(lambda b, ignore_temp: b < m, loop_body, [0, ignore_mask])

            ignore_mask = ignore_mask.stack()

            ignore_mask = K.expand_dims(ignore_mask, -1)

            box_loss_scale  = 2 - y_true[l][...,2:3] * y_true[l][...,3:4]
            if self.ciou_flag:

                raw_true_box    = y_true[l][...,0:4]
                ciou            = box_ciou(pred_box, raw_true_box)
                ciou_loss       = object_mask * (1 - ciou)
                location_loss   = K.sum(ciou_loss)
            else:

                raw_true_xy     = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
                raw_true_wh     = K.log(y_true[l][..., 2:4] / self.anchors[self.anchors_mask[l]] * input_shape[::-1])

                raw_true_wh     = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))

                xy_loss         = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)

                wh_loss         = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[...,2:4])
                location_loss   = (K.sum(xy_loss) + K.sum(wh_loss)) * 0.1

            confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) + \
                        (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
            
            class_loss      = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

            num_pos         = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
            num_neg         = tf.maximum(K.sum(K.cast((1 - object_mask) * ignore_mask, tf.float32)), 1)

            location_loss   = location_loss * self.box_ratio / num_pos
            confidence_loss = K.sum(confidence_loss) * self.balance[l] * self.obj_ratio / (num_pos + num_neg)
            class_loss      = K.sum(class_loss) * self.cls_ratio / num_pos / self.num_classes

            loss            += location_loss + confidence_loss + class_loss
            if self.print_loss:
                loss = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss, tf.shape(ignore_mask)], summarize=100, message='loss: ')
        return loss