import copy
import importlib
from models.yolo import YOLO
from models.yolov3 import YOLOv3
from models.yolov4 import YOLOv4
from models.architectures import (DarkNet53, DarkNet53_backbone,
                                  CSPDarkNet53, CSPDarkNet53_backbone)


def build_models(config, weights=None):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    input_shape = config.pop("input_shape")
    weight_path = config.pop("weight_path")
    load_weight_type = config.pop("load_weight_type")

    architecture_config = config['Architecture']
    architecture_name = architecture_config.pop("name")

    backbone_config = config['Backbone']
    backbone_config['input_shape'] = input_shape
    backbone_name = backbone_config.pop("name")
    backbone = getattr(mod, backbone_name)(**backbone_config)
    
    architecture_config['backbone'] = backbone
    architecture_config['anchors'] = config.pop("anchors")
    architecture_config['anchor_masks'] = config.pop("anchor_masks")
    architecture = getattr(mod, architecture_name)(**architecture_config)

    model = YOLO(architecture, image_size=input_shape, classes=config.pop('classes'))
    
    if weights:
        model.load_weights(weights)
    else:
        if load_weight_type and weight_path:
            if load_weight_type == "weights":
                model.load_weights(weight_path)
            elif load_weight_type == "models":
                model.load_models(weight_path)
    return model
