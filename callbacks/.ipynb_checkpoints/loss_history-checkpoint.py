import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import scipy.signal
from utils.logger import logger

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, 
                 result_path=None, 
                 max_ratio=1.0,
                 save_best=False,
                 save_format='tf'):
        super(LossHistory, self).__init__()

        self.result_path        = result_path
        self.max_ratio          = max_ratio
        self.save_best          = save_best
        self.save_format        = save_format
                     
        self.train_loss_list    = []
        self.valid_loss_list    = []
        self.epoches            = [0]
        self.current_train_loss = 0.0
        self.current_valid_loss = 0.0
   
    def on_epoch_end(self, epoch, logs={}):
        save_weight_path = os.path.join(self.result_path, 'weights')
        os.makedirs(save_weight_path, exist_ok=True)
        summary_path = os.path.join(self.result_path, 'summary')
        os.makedirs(summary_path, exist_ok=True)
        
        train_loss = logs.get('loss')
        valid_loss = logs.get('val_loss')
        self.train_loss_list.append(train_loss)
        self.valid_loss_list.append(valid_loss)
            
        iters = range(len(self.train_loss_list))

        plt.figure()
        plt.plot(iters, self.train_loss_list, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.valid_loss_list, 'coral', linewidth = 2, label='valid loss')
        try:
            if len(self.train_loss_list) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.train_loss_list, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.valid_loss_list, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth valid loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(summary_path, "epoch_loss.png"))

        plt.cla()
        plt.close("all")
        
        if self.save_best:
            print('')
            if train_loss < self.current_train_loss and train_loss < self.max_ratio:
                logger.info(f'Train loss score increase {self.current_train_loss:.2f}% to {train_loss:.2f}%')
                logger.info(f'Save best train loss weights to {os.path.join(save_weight_path, f"best_train_loss")}')
                self.model.save_weights(os.path.join(save_weight_path, f'best_train_loss'), save_format=self.save_format)
                self.current_train_loss = train_loss
            if valid_loss < self.current_valid_loss and valid_loss < self.max_ratio:
                logger.info(f'Validation loss score increase {self.current_valid_loss:.2f}% to {valid_loss:.2f}%')
                logger.info(f'Save best validation loss weights to {os.path.join(save_weight_path, f"best_valid_loss")}')
                self.model.save_weights(os.path.join(save_weight_path, f'best_valid_loss'), save_format=self.save_format)
                self.current_valid_loss = valid_loss