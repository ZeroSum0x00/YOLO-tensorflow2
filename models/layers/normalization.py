import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization


class FrozenBatchNormalization(BatchNormalization):
    # "Frozen state" and "inference mode" are two separate concepts.
    # `layer.trainable = False` is to freeze the layer, so the layer will use
    # stored moving `var` and `mean` in the "inference mode", and both `gama`
    # and `beta` will not be updated !
    def __init__(self, **kwargs):
        super(FrozenBatchNormalization, self).__init__(**kwargs)

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)
