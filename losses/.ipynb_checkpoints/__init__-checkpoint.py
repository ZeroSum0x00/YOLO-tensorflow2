import copy
import importlib
import tensorflow as tf

from .yolo_loss import YOLOLoss



def build_losses(config, model):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    losses = []
    for cfg in config:
        name = str(list(cfg.keys())[0])
        value = list(cfg.values())[0]
        coeff = value.pop("coeff")
        add_value = {'input_shape': model.architecture.input_size,
                     'anchors': model.architecture.anchors,
                     'anchor_masks': model.architecture.anchor_masks,
                     'num_classes': model.architecture.num_classes}
        arch = getattr(mod, name)(**value, **add_value)
        losses.append({'loss': arch, 'coeff': coeff})
    return losses