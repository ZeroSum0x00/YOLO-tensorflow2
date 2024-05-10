import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from utils.bboxes import yolo_correct_boxes, get_anchors_and_decode


class BaseDecoder:
    def __init__(self, 
                 anchors      = None,
                 anchor_masks = None,
                 input_size   = None,
                 num_classes  = 80,
                 max_boxes    = 100,
                 confidence   = 0.5,
                 nms_iou      = 0.5,
                 gray_padding = True):
        self.anchors      = anchors
        self.anchor_masks = anchor_masks
        self.input_size   = input_size
        self.num_classes  = num_classes
        self.max_boxes    = max_boxes
        self.confidence   = confidence
        self.nms_iou      = nms_iou
        self.gray_padding = gray_padding

    def __call__(self, inputs):
        image_shape = K.reshape(inputs[-1], [-1])
        box_xy = []
        box_wh = []
        box_confidence  = []
        box_class_probs = []
        for i in range(len(self.anchor_masks)):
            sub_box_xy, sub_box_wh, sub_box_confidence, sub_box_class_probs = get_anchors_and_decode(inputs[i], 
                                                                                                     self.anchors[self.anchor_masks[i]], 
                                                                                                     self.num_classes, 
                                                                                                     self.input_size[:-1])
            box_xy.append(K.reshape(sub_box_xy, [-1, 2]))
            box_wh.append(K.reshape(sub_box_wh, [-1, 2]))
            box_confidence.append(K.reshape(sub_box_confidence, [-1, 1]))
            box_class_probs.append(K.reshape(sub_box_class_probs, [-1, self.num_classes]))
        box_xy          = K.concatenate(box_xy, axis = 0)
        box_wh          = K.concatenate(box_wh, axis = 0)
        box_confidence  = K.concatenate(box_confidence, axis = 0)
        box_class_probs = K.concatenate(box_class_probs, axis = 0)

        boxes       = yolo_correct_boxes(box_xy, box_wh, self.input_size[:-1], image_shape, self.gray_padding)

        box_scores  = box_confidence * box_class_probs

        mask             = box_scores >= self.confidence
        max_boxes_tensor = K.constant(self.max_boxes, dtype='int32')
        boxes_out   = []
        scores_out  = []
        classes_out = []
        for c in range(self.num_classes):
            class_boxes      = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=self.nms_iou)

            class_boxes         = K.gather(class_boxes, nms_index)
            class_boxes_coord   = tf.concat([class_boxes[:, :2][..., ::-1], class_boxes[:, 2:][..., ::-1]], axis=-1)
            class_box_scores    = K.gather(class_box_scores, nms_index)
            classes             = K.ones_like(class_box_scores, 'int32') * c

            boxes_out.append(class_boxes_coord)
            scores_out.append(class_box_scores)
            classes_out.append(classes)
        boxes_out      = K.concatenate(boxes_out, axis=0)
        scores_out     = K.concatenate(scores_out, axis=0)
        classes_out    = K.concatenate(classes_out, axis=0)
        return boxes_out, scores_out, classes_out