import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model

from .yolov3 import ConvolutionBlock
from utils.train_processing import losses_prepare
from utils.logger import logger
from utils.constant import *



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

    def plot_model(self, input_shape, saved_path=""):
        input_shape = (input_shape[0]//32, input_shape[1]//32, 512)
        o = Input(shape=input_shape, name='Input')
        model = Model(inputs=[o], outputs=self.call(o))
        plot_model(model, to_file=f'{saved_path}/{self.name}_architecture.png', show_shapes=True)
        del o, model
        

class PANLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 activation      = 'mish', 
                 normalizer      = 'batch-norm', 
                 name            = "PANLayer", 
                 **kwargs):
        super(PANLayer, self).__init__(name=name, **kwargs)
        self.activation     = activation
        self.normalizer     = normalizer

    def build(self, input_shape):
        self.P5_up     = self._upsample_block(256, 2)
        self.P4_conv0  = ConvolutionBlock(256, 1, False, 1, 1, self.activation, self.normalizer)
        self.P4_block0 = self._conv_block([256, 512, 256, 512, 256])
        self.P4_up     = self._upsample_block(128, 2)
        self.P3_conv0  = ConvolutionBlock(128, 1, False, 1, 1, self.activation, self.normalizer)
        self.P3_block0 = self._conv_block([128, 256, 128, 256, 128])
        self.P3_down   = ConvolutionBlock(256, 3, True, 1, 1, self.activation, self.normalizer)
        self.P4_block1 = self._conv_block([256, 512, 256, 512, 256])
        self.P4_down   = ConvolutionBlock(512, 3, True, 1, 1, self.activation, self.normalizer)
        self.P5_block1 = self._conv_block([512, 1024, 512, 1024, 512])

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
                             activation    = self.activation, 
                             normalizer    = self.normalizer),
            UpSampling2D(size=upsample_size,)
        ])

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

    def plot_model(self, input_shape, saved_path=""):
        large_shape = (input_shape[0]//8, input_shape[1]//8, 256)
        medium_shape = (input_shape[0]//16, input_shape[1]//16, 512)
        small_shape = (input_shape[0]//32, input_shape[1]//32, 512)
        o3 = Input(shape=large_shape, name='small_object_scale')
        o4 = Input(shape=medium_shape, name='medium_object_scale')
        o5 = Input(shape=small_shape, name='large_object_scale')
        model = Model(inputs=[o3, o4, o5], outputs=self.call([o3, o4, o5]))
        plot_model(model, to_file=f'{saved_path}/{self.name}_architecture.png', show_shapes=True)
        del o3, o4, o5, model
        
        
class YOLOv4(tf.keras.Model):
    def __init__(self, 
                 backbone,
                 head_dims    = [256, 512, 1024],
                 anchors      = yolo_anchors,
                 anchor_masks = yolo_anchor_masks,
                 num_classes  = 80,
                 activation   = 'mish', 
                 normalizer   = 'batch-norm', 
                 name         = "YOLOv4", 
                 **kwargs):
        super(YOLOv4, self).__init__(name=name, **kwargs)
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
        self.block0     = self._conv_block([512, 1024, 512], name='convolution_extractor_0')
        self.spp        = SPPLayer([13, 9, 5], name="SPPLayer")
        self.block1     = self._conv_block([512, 1024, 512], name='convolution_extractor_1')
        self.pan        = PANLayer(self.activation, self.normalizer, name="PANLayer")
        self.conv_sbbox = self._yolo_head(h0, name='small_bbox_predictor')
        self.conv_mbbox = self._yolo_head(h1, name='medium_bbox_predictor')
        self.conv_lbbox = self._yolo_head(h2, name='large_bbox_predictor')

    def _conv_block(self, num_filters, name="conv_block"):
        return Sequential([
            ConvolutionBlock(filters,
                             kernel_size   = 1 if i % 2 == 0 else 3,
                             downsample    = False,
                             dilation_rate = (1, 1),
                             groups        = 1,
                             activation    = self.activation, 
                             normalizer    = self.normalizer) for i, filters in enumerate(num_filters)
        ], name=name)

    def _yolo_head(self, filters, name='yolo_head'):
        return Sequential([
            ConvolutionBlock(filters,
                             kernel_size   = (3, 3),
                             downsample    = False,
                             dilation_rate = (1, 1),
                             groups        = 1,
                             activation    = self.activation, 
                             normalizer    = self.normalizer),
            ConvolutionBlock(self.num_anchor_per_scale * (self.num_classes + 5),
                             kernel_size   = (1, 1),
                             downsample    = False,
                             activation    = None,
                             normalizer    = None)
        ], name=name)

    def call(self, inputs, training=False):
        feature_maps = self.backbone(inputs, training=training)
        P3, P4, P5   = feature_maps[-3:]
        P5 = self.block0(P5, training=training)
        P5 = self.spp(P5, training=training)
        P5 = self.block1(P5, training=training)

        P3, P4, P5 = self.pan([P3, P4, P5], training=training)
        P3_out = self.conv_sbbox(P3, training=training)
        P4_out = self.conv_mbbox(P4, training=training)
        P5_out = self.conv_lbbox(P5, training=training)
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
        yolo_model = Model(inputs=[o], outputs=self.call(o), name=self.name).summary()
        del yolo_model

    def plot_model(self, input_shape, saved_path=""):
        self.build(input_shape)
        o = Input(shape=input_shape, name='Input')
        yolo_model = Model(inputs=[o], outputs=self.call(o), name=self.name)
        plot_model(yolo_model, to_file=f'{saved_path}/{self.name}_architecture.png', show_shapes=True)
        self.spp.plot_model(input_shape, saved_path)
        self.pan.plot_model(input_shape, saved_path)
        plot_model(self.backbone, to_file=f'{saved_path}/{self.backbone.name}_architecture.png', show_shapes=True)
        logger.info(f"Saved models graph in {saved_path}")
        del o, yolo_model