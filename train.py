import os
import shutil
import argparse
import tensorflow as tf
from models import build_models
from losses import build_losses
from optimizers import build_optimizer
from callbacks import build_callbacks, mAPEvaluate
from data_utils.data_flow import get_train_test_data
from utils.train_processing import create_folder_weights, train_prepare
from utils.config_processing import load_config



def train(file_config):
    config = load_config(file_config)
    train_config = config['Train']
    data_config  = config['Dataset']
              
    if train_prepare(train_config['mode']):
        TRAINING_TIME_PATH = create_folder_weights(train_config['save_weight_path'])
        shutil.copy(file_config, os.path.join(TRAINING_TIME_PATH, os.path.basename(file_config)))

        model = build_models(config['Model'])
        
        train_generator, valid_generator, test_generator = get_train_test_data(data_dirs               = data_config['data_dir'], 
                                                                               annotation_dir          = data_config['annotation_dir'],
                                                                               classes                 = data_config['data_info']['classes'], 
                                                                               target_size             = config['Model']['input_shape'], 
                                                                               batch_size              = train_config['batch_size'], 
                                                                               yolo_strides            = config['Model']['strides'],
                                                                               yolo_anchors            = config['Model']['anchors'],
                                                                               yolo_anchors_mask       = config['Model']['anchor_masks'],
                                                                               max_bboxes              = data_config['data_info']['max_bboxes'],
                                                                               init_epoch              = train_config['epoch']['start'],
                                                                               end_epoch               = train_config['epoch']['end'],
                                                                               color_space             = data_config['data_info']['color_space'],
                                                                               augmentor               = data_config['data_augmentation'],
                                                                               endemic_augmentor       = data_config['data_endemic_augmention'],
                                                                               endemic_augmentor_proba = data_config['data_endemic_augmention_proba'],
                                                                               endemic_augmentor_ratio = data_config['data_endemic_augmention_ratio'],
                                                                               coordinate              = data_config['data_info']['coordinate'],
                                                                               normalizer              = data_config['data_normalizer']['norm_type'],
                                                                               mean_norm               = data_config['data_normalizer']['norm_mean'],
                                                                               std_norm                = data_config['data_normalizer']['norm_std'],
                                                                               data_type               = data_config['data_info']['data_type'],
                                                                               check_data              = data_config['data_info']['check_data'], 
                                                                               load_memory             = data_config['data_info']['load_memory'],
                                                                               exclude_difficult       = data_config['data_info']['exclude_difficult'],
                                                                               exclude_truncated       = data_config['data_info']['exclude_truncated'],
                                                                               dataloader_mode         = data_config['data_loader_mode'])

        losses    = build_losses(config['Losses'], model)

        callbacks = build_callbacks(config['Callbacks'], model, TRAINING_TIME_PATH)

        for callback in callbacks:
            if isinstance(callback, mAPEvaluate):
                callback.pass_data(valid_generator)

        optimizer = build_optimizer(config['Optimizer'])

        model.compile(optimizer=optimizer, loss=losses)

        if valid_generator is not None:
            model.fit(train_generator,
                      steps_per_epoch  = train_generator.N // train_config['batch_size'],
                      validation_data  = valid_generator,
                      validation_steps = valid_generator.N // train_config['batch_size'],
                      epochs           = train_config['epoch']['end'],
                      initial_epoch    = train_config['epoch']['start'],
                      callbacks        = callbacks)
        else:
            model.fit(train_generator,
                      steps_per_epoch     = train_generator.n // train_config['batch_size'],
                      epochs              = train_config['epoch']['end'],
                      initial_epoch       = train_config['epoch']['start'],
                      callbacks           = callbacks)
            
        if test_generator is not None:
            model.evaluate(test_generator)
            
        model.save_weights(TRAINING_TIME_PATH + 'weights/last_weights', save_format=train_config['save_weight_type'])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/yolov3.yaml", help="config file path")
    return parser.parse_args()

    
if __name__ == '__main__':
    cfg = parse_opt()
    file_config = cfg.config
    train(file_config)
