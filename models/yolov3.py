import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.utils import plot_model

from models.architectures.darknet53 import ConvolutionBlock
from utils.bboxes import yolo_correct_boxes, get_anchors_and_decode
from utils.train_processing import losses_prepare
from utils.logger import logger
from utils.constant import *


    
class FPNLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 activation = 'leaky-relu', 
                 normalizer = 'batch-norm', 
                 name       = "FPNLayer", 
                 **kwargs):
        super(FPNLayer, self).__init__(name=name, **kwargs)
        self.activation     = activation
        self.normalizer     = normalizer

    def build(self, input_shape):
        self.P5_block = self._conv_block([512, 1024, 512, 1024, 512])
        self.P5_up    = self._upsample_block(256, 2)
        self.P4_block = self._conv_block([256, 512, 256, 512, 256])
        self.P4_up    = self._upsample_block(128, 2)
        self.P3_block = self._conv_block([128, 256, 128, 256, 128])

    def _conv_block(self, num_filters):
        return Sequential([
            ConvolutionBlock(filters,
                             kernel_size   = 1 if i % 2 == 0 else 3,
                             downsample    = False,
                             dilation_rate = (1, 1),
                             groups        = 1,
                             activation    = self.activation,
                             normalizer    = self.normalizer) for i, filters in enumerate(num_filters)
        ])

    def _upsample_block(self, filters, upsample_size):
        return Sequential([
            ConvolutionBlock(filters,
                             kernel_size   = (1, 1),
                             downsample    = False,
                             dilation_rate = (1, 1),
                             groups        = 1,
                             activation    = self.activation, 
                             normalizer    = self.normalizer),
            UpSampling2D(size=upsample_size)
        ])

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
    
    def plot_model(self, input_shape, saved_path=""):
        large_shape = (input_shape[0]//8, input_shape[1]//8, 256)
        medium_shape = (input_shape[0]//16, input_shape[1]//16, 512)
        small_shape = (input_shape[0]//32, input_shape[1]//32, 1024)
        o3 = Input(shape=large_shape, name='small_object_scale')
        o4 = Input(shape=medium_shape, name='medium_object_scale')
        o5 = Input(shape=small_shape, name='large_object_scale')
        neck_model = Model(inputs=[o3, o4, o5], outputs=self.call([o3, o4, o5]))
        plot_model(neck_model, to_file=f'{saved_path}/{self.name}_architecture.png', show_shapes=True)
        del o3, o4, o5, neck_model
    

class YOLOv3(tf.keras.Model):
    def __init__(self, 
                 backbone,
                 head_dims    = [256, 512, 1024],
                 anchors      = yolo_anchors,
                 anchor_masks = yolo_anchor_masks,
                 num_classes  = 80,
                 activation   = 'leaky-relu', 
                 normalizer   = 'batch-norm', 
                 max_boxes    = 100,
                 confidence   = 0.5,
                 nms_iou      = 0.5,
                 input_size   = None,
                 gray_padding = True,
                 name         = "YOLOv3", 
                 **kwargs):
        super(YOLOv3, self).__init__(name=name, **kwargs)
        self.backbone             = backbone
        self.head_dims            = head_dims
        self.num_classes          = num_classes
        self.anchors              = np.array(anchors)
        self.num_anchor_per_scale = len(anchor_masks)
        self.anchor_masks         = np.array(anchor_masks)
        self.activation           = activation
        self.normalizer           = normalizer
        self.max_boxes            = max_boxes
        self.confidence           = confidence
        self.nms_iou              = nms_iou
        self.input_size           = input_size
        self.gray_padding         = gray_padding

    def build(self, input_shape):
        if isinstance(self.head_dims, (tuple, list)):
            if len(self.head_dims) != 3:
                raise ValueError("Length head_dims mutch equal 3")
        else:
            self.head_dims = [self.head_dims * 2**i for i in range(3)]

        h0, h1, h2 = self.head_dims
        self.neck       = FPNLayer(self.activation, self.normalizer, name="FPNLayer")
        self.conv_sbbox = self._yolo_head(h0, name='small_bbox_predictor')
        self.conv_mbbox = self._yolo_head(h1, name='medium_bbox_predictor')
        self.conv_lbbox = self._yolo_head(h2, name='large_bbox_predictor')

    def _yolo_head(self, filters, name='upsample_block'):
        return Sequential([
            ConvolutionBlock(filters,
                             kernel_size   = (3, 3),
                             downsample    = False,
                             dilation_rate = (1, 1),
                             groups        = 1,
                             activation    = self.activation, 
                             normalizer    = self.normalizer),
            ConvolutionBlock(self.num_anchor_per_scale*(self.num_classes + 5),
                             kernel_size   = (1, 1),
                             downsample    = False,
                             activation    = None,
                             normalizer    = None)
        ], name=name)

    def call(self, inputs, training=False):
        feature_maps = self.backbone(inputs, training=training)
        P3, P4, P5   = feature_maps[2:]
        P3, P4, P5   = self.neck([P3, P4, P5], training=training)
        P5_out       = self.conv_lbbox(P5, training=training)
        P4_out       = self.conv_mbbox(P4, training=training)
        P3_out       = self.conv_sbbox(P3, training=training)
        return P5_out, P4_out, P3_out

    @tf.function
    def predict(self, inputs):
        output_model = self(inputs, training=False)
        return output_model

    def calc_loss(self, y_true, y_pred, loss_object):
        loss = losses_prepare(loss_object)
        loss_value = 0
        if loss:
            loss_value += loss(y_true, y_pred)
        return loss_value
        
    def decode(self, inputs):
        image_shape = K.reshape(inputs[-1], [-1])
        box_xy = []
        box_wh = []
        box_confidence  = []
        box_class_probs = []
        for i in range(len(self.anchor_masks)):
            sub_box_xy, sub_box_wh, sub_box_confidence, sub_box_class_probs = get_anchors_and_decode(inputs[i], self.anchors[self.anchor_masks[i]], self.num_classes, self.input_size[:-1])
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

    def print_summary(self, input_shape):
        self.build(input_shape)
        o = Input(shape=input_shape, name='Input')
        yolo_model = Model(inputs=[o], outputs=self.call(o), name='YOLOv3').summary()
        del yolo_model
    
    def plot_model(self, input_shape, saved_path=""):
        self.build(input_shape)
        o = Input(shape=input_shape, name='Input')
        yolo_model = Model(inputs=[o], outputs=self.call(o))
        plot_model(yolo_model, to_file=f'{saved_path}/{self.name}_architecture.png', show_shapes=True)
        self.neck.plot_model(input_shape, saved_path)
        plot_model(self.backbone, to_file=f'{saved_path}/{self.backbone.name}_architecture.png', show_shapes=True)
        logger.info(f"Saved models graph in {saved_path}")
        del o, yolo_model
