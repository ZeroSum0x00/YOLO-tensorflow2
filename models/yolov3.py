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
        P3, P4, P5   = feature_maps[-3:]
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
