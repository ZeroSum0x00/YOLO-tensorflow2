import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, LeakyReLU, ZeroPadding2D, Input, UpSampling2D, Concatenate,Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from models.architectures.darknet53 import darknet53
from models.layers.normalization import FrozenBatchNormalization
from models.layers.activations import Mish
from utils.bboxes import yolo_correct_boxes, get_anchors_and_decode


class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 kernel_size=3, 
                 downsample=False, 
                 activation='leaky', 
                 norm_layer='batchnorm', 
                 **kwargs):
        super(ConvolutionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.activation = activation
        self.norm_layer = norm_layer

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


class YOLOv3Encoder(tf.keras.Model):
    def __init__(self, 
                 num_classes, 
                 num_anchor=3,
                 activation='leaky', 
                 norm_layer='bn', 
                 darknet_weight=None,
                 name="YOLOv3Encoder", **kwargs):
        super(YOLOv3Encoder, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.num_anchor = num_anchor
        self.activation = activation
        self.norm_layer = norm_layer
        self.darknet_weight = darknet_weight

    def build(self, input_shape):
        self.darknet53 = darknet53(input_shape=(416, 416, 3), activation=self.activation, norm_layer=self.norm_layer, model_weights=self.darknet_weight)
        self.block0 = self._conv_block([512, 1024, 512, 1024, 512], self.activation, self.norm_layer, name='convolution_extractor_0')
        self.conv_lobj = ConvolutionBlock(1024, 3, False, self.activation, self.norm_layer, name='large_object_predictor')
        self.conv_lbbox = ConvolutionBlock(self.num_anchor*(self.num_classes + 5), 1, False, None, None, name='large_bbox_predictor')
        self.upsample0 = self._upsample_block(256, 2, self.activation, self.norm_layer, name='upsample_block_0')
        self.block1 = self._conv_block([256, 512, 256, 512, 256], self.activation, self.norm_layer, name='convolution_extractor_1')
        self.conv_mobj = ConvolutionBlock(512, 3, False, self.activation, self.norm_layer, name='medium_object_predictor')
        self.conv_mbbox = ConvolutionBlock(self.num_anchor*(self.num_classes + 5), 1, False, None, None, name='medium_bbox_predictor')
        self.upsample1 = self._upsample_block(128, 2, self.activation, self.norm_layer, name='upsample_block_1')
        self.block2 = self._conv_block([128, 256, 128, 256, 128], self.activation, self.norm_layer, name='convolution_extractor_2')
        self.conv_sobj = ConvolutionBlock(256, 3, False, self.activation, self.norm_layer, name='small_object_predictor')
        self.conv_sbbox = ConvolutionBlock(self.num_anchor*(self.num_classes + 5), 1, False, None, None, name='small_bbox_predictor')

    @classmethod
    def _conv_block(cls, num_filters, activation='leaky', norm_layer='batchnorm', name="conv_block"):
        f0, f1, f2, f3, f4 = num_filters
        return Sequential([
            ConvolutionBlock(f0, 1, False, activation, norm_layer),
            ConvolutionBlock(f1, 3, False, activation, norm_layer),
            ConvolutionBlock(f2, 1, False, activation, norm_layer),
            ConvolutionBlock(f3, 3, False, activation, norm_layer),
            ConvolutionBlock(f4, 1, False, activation, norm_layer),
        ], name=name)

    @classmethod
    def _upsample_block(cls, filters, upsample_size, activation='leaky', norm_layer='batchnorm', name='upsample_block'):
        return Sequential([
            ConvolutionBlock(filters, 1, False, activation, norm_layer),
            UpSampling2D(size=upsample_size,)
        ], name=name)

    def call(self, inputs, training=False):
        x1, x2, x3 = self.darknet53(inputs, training=training)
        x3 = self.block0(x3, training=training)
        conv_lobj = self.conv_lobj(x3, training=training)
        layer82 = self.conv_lbbox(conv_lobj, training=training)

        x3 = self.upsample0(x3, training=training)
        x3 = tf.concat([x3, x2], axis=-1)
        x3 = self.block1(x3, training=training)
        conv_mobj = self.conv_mobj(x3, training=training)
        layer94 = self.conv_mbbox(conv_mobj, training=training)

        x3 = self.upsample1(x3, training=training)
        x3 = tf.concat([x3, x1], axis=-1)
        x3 = self.block2(x3, training=training)
        conv_sobj = self.conv_sobj(x3, training=training)
        layer106 = self.conv_sbbox(conv_sobj, training=training)
        return layer82, layer94, layer106


class YOLOv3Decoder:
    def __init__(self,
                 anchors,
                 num_classes,
                 input_size,
                 anchor_mask     = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 max_boxes       = 100,
                 confidence      = 0.5,
                 nms_iou         = 0.3,
                 letterbox_image = True):
        self.anchors = np.array(anchors)
        self.num_classes = num_classes
        self.input_size = input_size
        self.anchor_mask = np.array(anchor_mask)
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
    
    def decode_caculator(self, inputs):
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

        boxes       = yolo_correct_boxes(box_xy, box_wh, self.input_size[:-1], image_shape, self.letterbox_image)

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

    def __call__(self, inputs):
        return Lambda(self.decode_caculator)(inputs)
