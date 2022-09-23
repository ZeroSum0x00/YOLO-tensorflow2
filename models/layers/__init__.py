from models.layers.activations import *
from models.layers.normalization import *
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization





def get_activation_from_name(x, name=None):
    if name:
        if name.lower() == 'relu':
            return Activation('relu')(x)
        elif name.lower() == 'leaky' or name.lower() == 'leakyrelu':
            return LeakyReLU(alpha=0.1)(x)
        elif name.lower() == 'mish':
            return Mish()(x)
        else:
            return x
    else:
        return x


def get_normalization_from_name(x, name=None, *args, **kwargs):
    if name:
        if name.lower() == 'frozenbn':
            return FrozenBatchNormalization()(x, *args, **kwargs)
        elif name.lower() == 'bn' or name.lower() == 'batchnorm':
            return BatchNormalization()(x, *args, **kwargs)
        else:
            return x
    else:
        return x
