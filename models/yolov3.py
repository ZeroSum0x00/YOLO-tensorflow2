import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

from models.layers.normalization import FrozenBatchNormalization
from models.layers.activations import Mish
from utils.bboxes import yolo_correct_boxes, get_anchors_and_decode
from configs import general_config as cfg


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 kernel_size = 3, 
                 downsample  = False, 
                 activation  = cfg.YOLO_ACTIVATION, 
                 norm_layer  = cfg.YOLO_NORMALIZATION, 
                 **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.downsample  = downsample
        self.activation  = activation
        self.norm_layer  = norm_layer

        if downsample:
            self.padding = 'valid'
            self.strides = 2
        else:
            self.padding = 'same'
            self.strides = 1

    def build(self, input_shape):
        self.padding_layer = ZeroPadding2D(((1, 0), (1, 0))) if self.downsample else None
        self.conv = Conv2D(filters=self.filters, 
                           kernel_size=self.kernel_size, 
                           strides=self.strides,
                           padding=self.padding, 
                           use_bias=False if self.norm_layer else True, 
                           kernel_regularizer=l2(0.0005),
                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                           bias_initializer=tf.constant_initializer(0.))
        self.norm_layer = self.__get_norm_from_name(self.norm_layer)
        self.activation = self.__get_activation_from_name(self.activation)

    def __get_norm_from_name(self, name):
        if name:
            if name.lower() == 'frozenbn':
                return FrozenBatchNormalization()
            elif name.lower() == 'bn' or name.lower() == 'batchnorm':
                return BatchNormalization()
            else:
                return None
        else:
            return None

    def __get_activation_from_name(self, name):
        if name:
            if name.lower() == 'relu':
                return Activation('relu')
            elif name.lower() == 'leaky' or name.lower() == 'leakyrelu':
                return LeakyReLU(alpha=0.1)
            elif name.lower() == 'mish':
                return Mish()
            else:
                return None
        else:
            return None

    def call(self, inputs, training=False):
        if self.downsample:
            inputs = self.padding_layer(inputs)
        x = self.conv(inputs, training=training)
        if self.norm_layer:
            x = self.norm_layer(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    
class FPNLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 activation      = cfg.YOLO_ACTIVATION, 
                 norm_layer      = cfg.YOLO_NORMALIZATION, 
                 name            = "FPNLayer", 
                 **kwargs):
        super(FPNLayer, self).__init__(name=name, **kwargs)
        self.activation     = activation
        self.norm_layer     = norm_layer

    def build(self, input_shape):
        self.P5_block = self._conv_block([512, 1024, 512, 1024, 512], self.activation, self.norm_layer)
        self.P5_up    = self._upsample_block(256, 2, self.activation, self.norm_layer)
        self.P4_block = self._conv_block([256, 512, 256, 512, 256], self.activation, self.norm_layer)
        self.P4_up    = self._upsample_block(128, 2, self.activation, self.norm_layer)
        self.P3_block = self._conv_block([128, 256, 128, 256, 128], self.activation, self.norm_layer)

    def _conv_block(self, num_filters, activation='leaky', norm_layer='batchnorm', name="conv_block"):
        return Sequential([
            ConvolutionBlock(filters, 1 if i % 2 == 0 else 3, False, activation, norm_layer) for i, filters in enumerate(num_filters)
        ], name=name)

    def _upsample_block(self, filters, upsample_size, activation='leaky', norm_layer='batchnorm', name='upsample_block'):
        return Sequential([
            ConvolutionBlock(filters, 1, False, activation, norm_layer),
            UpSampling2D(size=upsample_size,)
        ], name=name)

    def call(self, inputs, training=False):
        P3, P4, P5  = inputs
        P5          = self.P5_block(P5, training=training)
        P5_up       = self.P5_up(P5, training=training)
        P4          = tf.concat([P5_up, P4], axis=-1)
        P4          = self.P4_block(P4, training=training)
        P4_up       = self.P4_up(P4, training=training)
        P3          = tf.concat([P4_up, P3], axis=-1)
        P3          = self.P3_block(P3, training=training)
        return P3, P4, P5
    

class YOLOv3(tf.keras.Model):
    def __init__(self, 
                 backbone,
                 num_classes     = cfg.NUM_CLASSES,
                 strides         = cfg.YOLO_STRIDES,
                 anchors         = cfg.YOLO_ANCHORS,
                 anchor_mask     = cfg.YOLO_ANCHORS_MASK,
                 activation      = cfg.YOLO_ACTIVATION, 
                 norm_layer      = cfg.YOLO_NORMALIZATION,
                 max_boxes       = cfg.YOLO_MAX_BBOXES,
                 confidence      = 0.5,
                 nms_iou         = cfg.TEST_IOU_THRESHOLD,
                 input_size      = cfg.YOLO_TARGET_SIZE,
                 gray_padding    = True,
                 name            = "YOLOv3", 
                 **kwargs):
        super(YOLOv3, self).__init__(name=name, **kwargs)
        self.backbone       = backbone
        self.num_classes    = num_classes
        self.anchors        = np.array(anchors)
        self.num_anchors    = len(anchors) // len(strides)
        self.anchor_mask    = np.array(anchor_mask)
        self.activation     = activation
        self.norm_layer     = norm_layer
        self.max_boxes      = max_boxes
        self.confidence     = confidence
        self.nms_iou        = nms_iou
        self.input_size     = input_size
        self.gray_padding   = gray_padding

    def build(self, input_shape):
        self.neck       = FPNLayer(self.activation, self.norm_layer, name="FPNLayer")
        self.conv_lbbox = self._yolo_head(1024, self.activation, self.norm_layer, name='large_bbox_predictor')
        self.conv_mbbox = self._yolo_head(512, self.activation, self.norm_layer, name='medium_bbox_predictor')
        self.conv_sbbox = self._yolo_head(256, self.activation, self.norm_layer, name='small_bbox_predictor')

    def _yolo_head(self, filters, activation='leaky', norm_layer='batchnorm', name='upsample_block'):
        return Sequential([
            ConvolutionBlock(filters, 3, False, activation, norm_layer),
            ConvolutionBlock(self.num_anchors*(self.num_classes + 5), 1, False, None, None)
        ], name=name)
        
    def call(self, inputs, training=False):
        P3, P4, P5 = self.backbone(inputs, training=training)
        P3, P4, P5 = self.neck([P3, P4, P5], training=training)
        P5_out     = self.conv_lbbox(P5, training=training)
        P4_out     = self.conv_mbbox(P4, training=training)
        P3_out     = self.conv_sbbox(P3, training=training)
        return P5_out, P4_out, P3_out

    def decode(self, inputs):
        image_shape = K.reshape(inputs[-1],[-1])
        box_xy = []
        box_wh = []
        box_confidence  = []
        box_class_probs = []
        for i in range(len(self.anchor_mask)):
            sub_box_xy, sub_box_wh, sub_box_confidence, sub_box_class_probs = get_anchors_and_decode(inputs[i], self.anchors[self.anchor_mask[i]], self.num_classes, self.input_size[:-1])
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
