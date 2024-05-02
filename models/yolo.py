import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from utils.post_processing import get_labels
from utils.logger import logger


class YOLO(tf.keras.Model):
    def __init__(self, 
                 architecture, 
                 image_size=(416, 416, 3),
                 classes=None,
                 **kwargs):
        super(YOLO, self).__init__(**kwargs)
        self.architecture            = architecture
        self.architecture.input_size = image_size
        self.image_size              = image_size
        if classes:
            if isinstance(classes, str):
                self.classes, _ = get_labels(classes)
            else:
                self.classes = classes
            self.architecture.num_classes = len(self.classes)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super(YOLO, self).compile(**kwargs)
        self.optimizer    = optimizer
        self.loss_object  = loss
        self.list_metrics = metrics
        
    @property
    def metrics(self):
        if self.list_metrics:
            return [
                self.total_loss_tracker,
                *self.list_metrics,
            ]
        else:
            return [self.total_loss_tracker]

    def train_step(self, data):
        loss_value = 0
        images, targets = data
        
        with tf.GradientTape() as tape:
            y_pred = self.architecture(images, training=True)
            
            for losses in self.loss_object:
                loss_value  += self.architecture.calc_loss(y_true=targets, 
                                                           y_pred=y_pred, 
                                                           loss_object=losses)
                
            loss_value  = tf.reduce_sum(self.architecture.losses) + loss_value
            
        grads = tape.gradient(loss_value, self.architecture.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.architecture.trainable_variables))
        self.total_loss_tracker.update_state(loss_value)
        
        if self.list_metrics:
            for metric in self.list_metrics:
                metric.update_state(targets, y_pred)

        results = {}
        for metric in self.metrics:
            if isinstance(metric.result(), dict):
                for k, v in metric.result().items():
                    results[k] = v
            else:
                results[metric.name] = metric.result()

        results['learning_rate'] = self.optimizer.lr
        return results

    def test_step(self, data):
        loss_value = 0
        images, targets = data
        y_pred  = self.architecture(images, training=False)
        
        for losses in self.loss_object:
            loss_value  += self.architecture.calc_loss(y_true=targets, 
                                                       y_pred=y_pred, 
                                                       loss_object=losses)
            
        loss_value  = tf.reduce_sum(self.architecture.losses) + loss_value
        self.total_loss_tracker.update_state(loss_value)

        if self.list_metrics:
            for metric in self.list_metrics:
                metric.update_state(targets, y_pred)

        results = {}
        for metric in self.metrics:
            if isinstance(metric.result(), dict):
                for k, v in metric.result().items():
                    results[k] = v
            else:
                results[metric.name] = metric.result()
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
        output_model = self.architecture(images, training=False)
        inputs  = [*output_model, origin_shape]
        out_boxes, out_scores, out_classes = self.architecture.decode(inputs)
        return out_boxes, out_scores, out_classes

    def save_weights(self, weight_path, save_head=True, save_format='tf', **kwargs):
        if save_head:
            self.architecture.save_weights(weight_path, save_format=save_format, **kwargs)
        else:
            backup_model = Model(inputs=self.architecture.input, outputs=self.architecture.output)
            backup_model.get_layer("medium_bbox_predictor").pop()
            backup_model.get_layer("large_bbox_predictor").pop()
            backup_model.get_layer("small_bbox_predictor").pop()
            backup_model.save_weights(weight_path, save_format=save_format, **kwargs)

    def load_weights(self, weights):
        self.architecture.load_weights(weights).expect_partial()
        logger.info("Load yolo weights from {}".format(weights))

    def save_models(self, weight_path, save_format='tf'):
        self.architecture.save(weight_path, save_format=save_format)

    def load_models(self, weight_objects):
        for weight in weight_objects:
            weight_path = weight['path']
            custom_objects = weight['custom_objects']
            if weight_path:
                self.architecture = load_model(weight_path, custom_objects=custom_objects)
                logger.info("Load yolo model from {}".format(weight_path))
                
    def summary(self):
        self.architecture.print_summary(self.image_size)
        
    def plot_model(self, saved_path=""):
        self.architecture.plot_model(self.image_size, saved_path)
        
    def get_config(self):
        config = super().get_config()
        config.update({
                "architecture": self.architecture,
                "total_loss_tracker": self.total_loss_tracker,
                "optimizer": self.optimizer
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
