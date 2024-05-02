import os
import copy
import importlib
from callbacks.evaluate_map import mAPEvaluate
from callbacks.loss_history import LossHistory
from callbacks.train_logger import TrainLogger
from callbacks.train_summary import TrainSummary
from callbacks.warmup_lr import AdvanceWarmUpLearningRate
from tensorflow.keras.callbacks import *


def build_callbacks(config, model, result_path):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    callbacks = []
    if config:
        for cfg in config:
            save_path = result_path
            name = str(list(cfg.keys())[0])
            value = list(cfg.values())[0]
            extend_path = value.pop("extend_path", None)
            
            # if name == "mAPEvaluate":
            #     value.update({
            #         'input_shape': model.architecture.input_size,
            #         'classes': model.classes,
            #         'max_bboxes': model.architecture.max_boxes
            #     })
                
            if extend_path is not None:
                save_path = os.path.join(save_path, extend_path)
                
            if not value:
                value = {}

            arch = getattr(mod, name)(save_path, **value)
            callbacks.append(arch)
    return callbacks
