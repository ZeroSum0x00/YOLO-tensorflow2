import os
import tensorflow as tf
from utils.logger import logger

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class MetricHistory(tf.keras.callbacks.Callback):
    def __init__(self,
                 result_path=None,
                 min_ratio=0.2,
                 save_best=False,
                 save_format='tf'):
        super(MetricHistory, self).__init__()
        self.result_path            = result_path
        self.min_ratio              = min_ratio
        self.save_best              = save_best
        self.save_format        = save_format

        self.metric_infomation = {}

    def on_epoch_end(self, epoch, logs={}):
        save_weight_path = os.path.join(self.result_path, 'weights')
        os.makedirs(save_weight_path, exist_ok=True)
        summary_path = os.path.join(self.result_path, 'summary')
        os.makedirs(summary_path, exist_ok=True)
        
        print('')
        for metric in self.model.list_metrics:
            metric_name = metric.name
            metric_type = metric.save_type.lower()
            if metric_name not in self.metric_infomation:
                self.metric_infomation[metric_name] = {}
                self.metric_infomation[metric_name]['train_object'] = []
                self.metric_infomation[metric_name]['valid_object'] = []
                if metric_type == "increase":
                    self.metric_infomation[metric_name]['train_value'] = 0.0
                    self.metric_infomation[metric_name]['valid_value'] = 0.0
                elif metric_type == "decrease":
                    self.metric_infomation[metric_name]['train_value'] = self.min_ratio
                    self.metric_infomation[metric_name]['valid_value'] = self.min_ratio
                    
            train_value = logs.get(metric_name)
            valid_value = logs.get('val_' + metric_name)
            self.metric_infomation[metric_name]['train_object'].append(train_value)
            self.metric_infomation[metric_name]['valid_object'].append(valid_value)

            with open(os.path.join(summary_path, f"train_{metric_name}.txt"), 'a') as f:
                f.write(f"Train {metric_name} in epoch {epoch + 1}: {str(train_value)}")
                f.write("\n")
            with open(os.path.join(summary_path, f"val_{metric_name}.txt"), 'a') as f:
                f.write(f"Valid {metric_name} in epoch {epoch + 1}: {str(valid_value)}")
                f.write("\n")
                
            iters = range(len(self.metric_infomation[metric_name]['train_object']))
    
            plt.figure()
            plt.plot(iters, self.metric_infomation[metric_name]['train_object'], 'red', linewidth=2, label=f'train {metric_name.replace("-", " ")}')
            plt.plot(iters, self.metric_infomation[metric_name]['valid_object'], 'coral', linewidth=2, label=f'valid {metric_name.replace("-", " ")}')
            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel(metric_name.replace("-", " ").title())
            plt.title(f'A {metric_name.replace("-", " ").title()} Curve')
            if metric_type == "increase":
                plt.legend(loc="lower right")
            else:
                plt.legend(loc="upper right")
            plt.savefig(os.path.join(summary_path, f"epoch_{metric_name}.png"))
            plt.cla()
            plt.close("all")

            if self.save_best:
                if metric_type == "increase":
                    if train_value > self.metric_infomation[metric_name]['train_value'] and train_value > self.min_ratio:
                        logger.info(f"Train {metric_name} score increase {self.metric_infomation[metric_name]['train_value']:.4f} to {train_value:.4f}")
                        logger.info(f'Save best train {metric_name} weights to {os.path.join(save_weight_path, f"best_train_{metric_name}")}')
                        self.model.save_weights(os.path.join(save_weight_path, f"best_train_{metric_name}"), save_format=self.save_format)
                        self.metric_infomation[metric_name]['train_value'] = train_value
                    if valid_value and valid_value > self.metric_infomation[metric_name]['valid_value'] and valid_value > self.min_ratio:
                        logger.info(f"Validation {metric_name} score increase {self.metric_infomation[metric_name]['valid_value']:.4f} to {valid_value:.4f}")
                        logger.info(f'Save best validation {metric_name} weights to {os.path.join(save_weight_path, f"best_valid_{metric_name}")}')
                        self.model.save_weights(os.path.join(save_weight_path, f"best_valid_{metric_name}"), save_format=self.save_format)
                        self.metric_infomation[metric_name]['valid_value'] = valid_value
                    if train_value == 1.0 and valid_value and valid_value > self.metric_infomation[metric_name]['valid_value']:
                        logger.info(f'Train {metric_name} is maximum, but validation {metric_name} increase to {valid_value:.4f}')
                        logger.info(f'Save best train {metric_name} weights to {os.path.join(save_weight_path, f"best_train_{metric_name}")}')
                        self.model.save_weights(os.path.join(save_weight_path, f"best_train_{metric_name}"), save_format=self.save_format)
                    if valid_value == 1.0 and valid_value and train_value > self.metric_infomation[metric_name]['train_value']:
                        logger.info(f'Validation {metric_name} is maximum, but train {metric_name} increase to {train_value:.4f}')
                        logger.info(f'Save best validation {metric_name} weights to {os.path.join(save_weight_path, f"best_valid_{metric_name}")}')
                        self.model.save_weights(os.path.join(save_weight_path, f"best_valid_{metric_name}"), save_format=self.save_format)
                elif metric_type == "decrease":
                    if train_value < self.metric_infomation[metric_name]['train_value'] and train_value < self.min_ratio:
                        if epoch == 0:
                            logger.info(f"First train {metric_name} score: {train_value:.4f}")
                        else:
                            logger.info(f"Train {metric_name} score decrease {self.metric_infomation[metric_name]['train_value']:.4f} to {train_value:.4f}")
                        logger.info(f'Save best train {metric_name} weights to {os.path.join(save_weight_path, f"best_train_{metric_name}")}')
                        self.model.save_weights(os.path.join(save_weight_path, f"best_train_{metric_name}"), save_format=self.save_format)
                        self.metric_infomation[metric_name]['train_value'] = train_value
                    if valid_value and valid_value < self.metric_infomation[metric_name]['valid_value'] and valid_value < self.min_ratio:
                        if epoch == 0:
                            logger.info(f"First validation {metric_name} score: {valid_value:.4f}")
                        else:
                            logger.info(f"Validation {metric_name} score decrease {self.metric_infomation[metric_name]['valid_value']:.4f} to {valid_value:.4f}")
                        logger.info(f'Save best validation {metric_name} weights to {os.path.join(save_weight_path, f"best_valid_{metric_name}")}')
                        self.model.save_weights(os.path.join(save_weight_path, f"best_valid_{metric_name}"))
                        self.metric_infomation[metric_name]['valid_value'] = valid_value
                    if train_value == 0.0 and valid_value and valid_value < self.metric_infomation[metric_name]['valid_value']:
                        logger.info(f'Train {metric_name} is minimum, but validation {metric_name} decrease to {valid_value:.4f}')
                        logger.info(f'Save best train {metric_name} weights to {os.path.join(save_weight_path, f"best_train_{metric_name}")}')
                        self.model.save_weights(os.path.join(save_weight_path, f"best_train_{metric_name}"))
                    if valid_value == 0.0 and valid_value and train_value < self.metric_infomation[metric_name]['train_value']:
                        logger.info(f'Validation {metric_name} is minimum, but train {metric_name} decrease to {train_value:.4f}')
                        logger.info(f'Save best validation {metric_name} weights to {os.path.join(save_weight_path, f"best_valid_{metric_name}")}')
                        self.model.save_weights(os.path.join(save_weight_path, f"best_valid_{metric_name}"))