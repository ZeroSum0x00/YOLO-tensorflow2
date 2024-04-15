import copy
import importlib
from augmenter.geometric import *
from augmenter.photometric import *


def build_augmenter(config, additional_config=None):
    config = copy.deepcopy(config)
    mod = importlib.import_module(__name__)
    augmenter = dict()

    if config:
        for main_key, main_value in config.items():
            sub_augmenter = []
            if main_value:
                for cfg in main_value:
                    name = str(list(cfg.keys())[0])
                    value = list(cfg.values())[0]

                    if value and additional_config:
                        for add_name, add_value in additional_config.items():
                            if name == str(add_name) and eval(name):
                                value.update(add_value)
                    if not value:
                        value = {}

                    arch = getattr(mod, name)(**value)
                    sub_augmenter.append(arch)
            augmenter[main_key] = sub_augmenter
    return augmenter