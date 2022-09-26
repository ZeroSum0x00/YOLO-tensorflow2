import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import UpSampling2D
from models.yolov3 import ConvolutionBlock
from configs import base_config as cfg


class SPPLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 pool_sizes     = [13, 9, 5],
                 name            = "SPPLayer", 
                 **kwargs):
        super(SPPLayer, self).__init__(name=name, **kwargs)
        self.pool_sizes     = pool_sizes

    def build(self, input_shape):
        pool1, pool2, pool3 = self.pool_sizes
        self.small_feature  = MaxPool2D(pool_size=(pool1, pool1), padding='same', strides=1)
        self.medium_feature = MaxPool2D(pool_size=(pool2, pool2), padding='same', strides=1)
        self.large_feature  = MaxPool2D(pool_size=(pool3, pool3), padding='same', strides=1)
    
    def call(self, inputs, training=False):
        pooling_1 = self.small_feature(inputs, training=training)
        pooling_2 = self.medium_feature(inputs, training=training)
        pooling_3 = self.large_feature(inputs, training=training)
        x = concatenate([pooling_1, pooling_2, pooling_3, inputs], axis=-1)
        return x


class PANLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 activation      = 'leaky', 
                 norm_layer      = 'bn', 
                 name            = "PANLayer", 
                 **kwargs):
        super(PANLayer, self).__init__(name=name, **kwargs)
        self.activation     = activation
        self.norm_layer     = norm_layer

    def build(self, input_shape):
        self.P5_up     = self._upsample_block(256, 2, self.activation, self.norm_layer)
        self.P4_conv0  = ConvolutionBlock(256, 1, False, self.activation, self.norm_layer)
        self.P4_block0 = self._conv_block([256, 512, 256, 512, 256], self.activation, self.norm_layer)
        self.P4_up     = self._upsample_block(128, 2, self.activation, self.norm_layer)
        self.P3_conv0  = ConvolutionBlock(128, 1, False, self.activation, self.norm_layer)
        self.P3_block0 = self._conv_block([128, 256, 128, 256, 128], self.activation, self.norm_layer)
        self.P3_down   = ConvolutionBlock(256, 3, True, self.activation, self.norm_layer)
        self.P4_block1 = self._conv_block([256, 512, 256, 512, 256], self.activation, self.norm_layer)
        self.P4_down   = ConvolutionBlock(512, 3, True, self.activation, self.norm_layer)
        self.P5_block1 = self._conv_block([512, 1024, 512, 1024, 512], self.activation, self.norm_layer)

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
        P5_up   = self.P5_up(P5, training=training)
        P4      = self.P4_conv0(P4, training=training)
        P4      = concatenate([P4, P5_up], axis=-1)
        P4      = self.P4_block0(P4, training=training)
        P4_up   = self.P4_up(P4, training=training)
        P3      = self.P3_conv0(P3, training=training)
        P3      = concatenate([P3, P4_up], axis=-1)
        P3      = self.P3_block0(P3, training=training)
        P3_down = self.P3_down(P3, training=training)
        P4      = concatenate([P3_down, P4], axis=-1)
        P4      = self.P4_block1(P4, training=training)
        P4_down = self.P4_down(P4, training=training)
        P5      = concatenate([P4_down, P5], axis=-1)
        P5      = self.P5_block1(P5, training=training)
        return P3, P4, P5


class YOLOv4Encoder(tf.keras.Model):
    def __init__(self, 
                 backbone,
                 num_classes     = 80,
                 num_anchor      = 3,
                 activation      = 'leaky', 
                 norm_layer      = 'bn', 
                 name            = "YOLOv4Encoder", 
                 **kwargs):
        super(YOLOv4Encoder, self).__init__(name=name, **kwargs)
        self.backbone       = backbone
        self.num_classes    = num_classes
        self.num_anchor     = num_anchor
        self.activation     = activation
        self.norm_layer     = norm_layer

    def build(self, input_shape):
        self.block0     = self._conv_block([512, 1024, 512], self.activation, self.norm_layer, name='convolution_extractor_0')
        self.spp        = SPPLayer([13, 9, 5], name="SPPLayer")
        self.block1     = self._conv_block([512, 1024, 512], self.activation, self.norm_layer, name='convolution_extractor_1')
        self.pan        = PANLayer(self.activation, self.norm_layer, name="PANLayer")
        self.conv_lbbox = self._yolo_head(1024, self.activation, self.norm_layer, name='large_bbox_predictor')
        self.conv_mbbox = self._yolo_head(512, self.activation, self.norm_layer, name='medium_bbox_predictor')
        self.conv_sbbox = self._yolo_head(256, self.activation, self.norm_layer, name='small_bbox_predictor')

    def _conv_block(self, num_filters, activation='leaky', norm_layer='batchnorm', name="conv_block"):
        return Sequential([
            ConvolutionBlock(filters, 1 if i % 2 == 0 else 3, False, activation, norm_layer) for i, filters in enumerate(num_filters)
        ], name=name)

    def _yolo_head(self, filters, activation='leaky', norm_layer='batchnorm', name='yolo_head'):
        return Sequential([
            ConvolutionBlock(filters, 3, False, activation, norm_layer),
            ConvolutionBlock(self.num_anchor*(self.num_classes + 5), 1, False, None, None)
        ], name=name)

    def call(self, inputs, training=False):
        P3, P4, P5 = self.backbone(inputs, training=training)
        P5 = self.block0(P5, training=training)
        P5 = self.spp(P5, training=training)
        P5 = self.block1(P5, training=training)

        P3, P4, P5 = self.pan([P3, P4, P5], training=training)
        P3_out = self.conv_sbbox(P3, training=training)
        P4_out = self.conv_mbbox(P4, training=training)
        P5_out = self.conv_lbbox(P5, training=training)
        return P5_out, P4_out, P3_out


class YOLOv4Decoder:
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
