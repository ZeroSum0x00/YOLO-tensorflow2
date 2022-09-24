import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

from models.layers import get_activation_from_name, get_normalization_from_name
from configs import base_config as cfg


def convolutional_block(x, filters, kernel_size, downsample=False, activation='leaky', norm_layer='batchnorm'):
    if downsample:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    x = Conv2D(filters=filters, 
               kernel_size=kernel_size, 
               strides=strides,
               padding=padding, 
               use_bias=not norm_layer, 
               kernel_regularizer=l2(0.0005),
               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
               bias_initializer=tf.constant_initializer(0.))(x)
    if norm_layer:
        x = get_normalization_from_name(x, norm_layer)
    if activation:
        x = get_activation_from_name(x, activation)
    return x


def residual_block(x, num_filters, activation='leaky', norm_layer='batchnorm'):
    shortcut = x
    x = convolutional_block(x, filters=num_filters[0], kernel_size=1, activation=activation, norm_layer=norm_layer)
    x = convolutional_block(x, filters=num_filters[1], kernel_size=3, activation=activation, norm_layer=norm_layer)
    x = add([shortcut, x])
    return x


def DarkNet53(input_shape, activation='leaky', norm_layer='batchnorm', model_weights=None):
    input_data  = Input(input_shape)
    x = convolutional_block(input_data, 32, 3, activation=activation, norm_layer=norm_layer)
    x = convolutional_block(x, 64, 3, downsample=True, activation=activation, norm_layer=norm_layer)
    
    for i in range(1):
        x = residual_block(x,  [32, 64], activation=activation, norm_layer=norm_layer)

    x = convolutional_block(x, 128, 3, downsample=True, activation=activation, norm_layer=norm_layer)

    for i in range(2):
        x = residual_block(x, [64, 128], activation=activation, norm_layer=norm_layer)

    x = convolutional_block(x, 256, 3, downsample=True, activation=activation, norm_layer=norm_layer)

    for i in range(8):
        x = residual_block(x, [128, 256], activation=activation, norm_layer=norm_layer)

    route_1 = x
    x = convolutional_block(x, 512, 3, downsample=True, activation=activation, norm_layer=norm_layer)

    for i in range(8):
        x = residual_block(x, [256, 512], activation=activation, norm_layer=norm_layer)

    route_2 = x
    x = convolutional_block(x, 1024, 3, downsample=True, activation=activation, norm_layer=norm_layer)

    for i in range(4):
        x = residual_block(x, [512, 1024], activation=activation, norm_layer=norm_layer)
    
    model = Model(inputs=input_data, outputs=[route_1, route_2, x])
    if model_weights:
        load_yolo_weights(model, model_weights)
    return model


def CSPDarkNetBlock(x, num_filters, block_iter, activation, norm_layer):
    route = x
    route = convolutional_block(route, num_filters[1], 1, activation=activation, norm_layer=norm_layer)
    
    x = convolutional_block(x, num_filters[1], 1, activation=activation, norm_layer=norm_layer)

    for i in range(block_iter):
        x = residual_block(x,  [num_filters[0], num_filters[1]], activation=activation, norm_layer=norm_layer)

    x = convolutional_block(x, num_filters[1], 1, activation=activation, norm_layer=norm_layer)
    x = concatenate([x, route], axis=-1)
    x = convolutional_block(x, num_filters[0]*2, 1, activation=activation, norm_layer=norm_layer)
    return x


def CSPDarkNet53(input_shape, activation='mish', norm_layer='batchnorm', model_weights=None):
    input_data  = Input(input_shape)

    x = convolutional_block(input_data, 32, 3, activation=activation, norm_layer=norm_layer)

    # Downsample 1
    x = convolutional_block(x, 64, 3, downsample=True, activation=activation, norm_layer=norm_layer)
    
    # CSPResBlock 1
    x = CSPDarkNetBlock(x, [32, 64], 1, activation=activation, norm_layer=norm_layer)

    # Downsample 2
    x = convolutional_block(x, 128, 3, downsample=True, activation=activation, norm_layer=norm_layer)

    # CSPResBlock 2
    x = CSPDarkNetBlock(x, [64, 64], 2, activation=activation, norm_layer=norm_layer)

    # Downsample 3
    x = convolutional_block(x, 256, 3, downsample=True, activation=activation, norm_layer=norm_layer)

    # CSPResBlock 3
    x = CSPDarkNetBlock(x, [128, 128], 8, activation=activation, norm_layer=norm_layer)

    route_1 = x

    # Downsample 4
    x = convolutional_block(x, 512, 3, downsample=True, activation=activation, norm_layer=norm_layer)

    # CSPResBlock 4
    x = CSPDarkNetBlock(x, [256, 256], 8, activation=activation, norm_layer=norm_layer)

    route_2 = x

    # Downsample 5
    x = convolutional_block(x, 1024, 3, downsample=True, activation=activation, norm_layer=norm_layer)

    # CSPResBlock 5
    x = CSPDarkNetBlock(x, [512, 512], 4, activation=activation, norm_layer=norm_layer)

    model = Model(inputs=input_data, outputs=[x])
    return model


def load_yolo_weights(model, weights_file):
    #tf.keras.backend.clear_session() # used to reset layer names
    # load Darknet original weights to TensorFlow model
    range1 = 75
    range2 = [58, 66, 74]
    
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(range1):
            try:
                if i > 0:
                    conv_layer_name = 'conv2d_%d' %i
                else:
                    conv_layer_name = 'conv2d'
                    
                if j > 0:
                    bn_layer_name = 'batch_normalization_%d' %j
                else:
                    bn_layer_name = 'batch_normalization'
                
                conv_layer = model.get_layer(conv_layer_name)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

                if i not in range2:
                    # darknet weights: [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    # tf weights: [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = model.get_layer(bn_layer_name)
                    j += 1
                else:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if i not in range2:
                    conv_layer.set_weights([conv_weights])
                    bn_layer.set_weights(bn_weights)
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])
            except:
                print(f'load weight failt {i}')
        # assert len(wf.read()) == 0, 'failed to read all data'
