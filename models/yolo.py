import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model

from utils.logger import logger


class YOLO(tf.keras.Model):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 image_size=(416, 416, 3), 
                 is_training=False, 
                 **kwargs):
        super(YOLO, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.image_size = image_size
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    def compile(self, optimizer, loss, **kwargs):
        super(YOLO, self).compile(**kwargs)
        self.optimizer = optimizer
        self.yolo_loss = loss

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
        ]

    def train_step(self, data):
        images, targets = data
        targets = [targets[2], targets[1], targets[0]]
        with tf.GradientTape() as tape:
            y_pred      = self.encoder(images, training=True)
            loss_value  = self.yolo_loss(y_true=targets, y_pred=y_pred)
            loss_value  = tf.reduce_sum(self.encoder.losses) + loss_value
        grads = tape.gradient(loss_value, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))

        self.total_loss_tracker.update_state(loss_value)
        results = {m.name: m.result() for m in self.metrics}
        results['learning_rate'] = self.optimizer.lr
        return results

    def test_step(self, data):
        images, targets = data
        targets = [targets[2], targets[1], targets[0]]
        y_pred      = self.encoder(images, training=False)
        loss_value  = self.yolo_loss(y_true=targets, y_pred=y_pred)
        loss_value  = tf.reduce_sum(self.encoder.losses) + loss_value
        self.total_loss_tracker.update_state(loss_value)
        results = {m.name: m.result() for m in self.metrics}
        return results
    
    def call(self, inputs):
        try:
            out_boxes, out_scores, out_classes = self.predict(inputs, self.image_size)
            return out_boxes, out_scores, out_classes
        except:
            return inputs

    @tf.function
    def predict(self, inputs):
        images, origin_shape = inputs
        output_encoder = self.encoder(images, training=False)
        inputs  = [*output_encoder, origin_shape]
        out_boxes, out_scores, out_classes = self.decoder(inputs)
        return out_boxes, out_scores, out_classes

    def save_weights(self, weight_path, save_head=True, save_format='tf', **kwargs):
        if save_head:
            self.encoder.save_weights(weight_path, save_format=save_format, **kwargs)
        else:
            backup_model = copy.deepcopy(self.encoder)
            backup_model.get_layer("medium_bbox_predictor").pop()
            backup_model.get_layer("large_bbox_predictor").pop()
            backup_model.get_layer("small_bbox_predictor").pop()
            backup_model.save_weights(weight_path, save_format=save_format, **kwargs)

    def load_weights(self, weight_objects):
        for weight in weight_objects:
            weight_path = weight['path']
            custom_objects = weight['custom_objects']
            if weight_path:
                self.encoder.build(input_shape=self.image_size)
                self.encoder.built = True
                self.encoder.load_weights(weight_path)
                logger.info("Load yolo weights from {}".format(weight_path))

    def save_models(self, weight_path, save_format='tf'):
        self.encoder.save(weight_path, save_format=save_format)

    def load_models(self, weight_objects):
        for weight in weight_objects:
            weight_path = weight['path']
            custom_objects = weight['custom_objects']
            if weight_path:
                self.encoder = load_model(weight_path, custom_objects=custom_objects)
                logger.info("Load yolo model from {}".format(weight_path))

    def get_config(self):
        config = super().get_config()
        config.update({
                "encoder": self.encoder,
                "decoder": self.decoder,
                "total_loss_tracker": self.total_loss_tracker,
                "optimizer": self.optimizer
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
