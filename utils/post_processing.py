import cv2
import numpy as np
from configs import base_config as cfg
from utils.logger import logger


def get_label_name(classes, index):
    min_idx = sorted(classes.values())[0]
    for key, value in classes.items():
        if value - min_idx == index:
            return key
    print("key doesn't exist")


def resize_image(image, target_size, letterbox_image):
    h, w, _    = image.shape
    ih, iw, _  = target_size
    if letterbox_image:
        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_resized = cv2.resize(image, (nw, nh))
        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=image.dtype)
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        return image_paded
    else:
        image = cv2.resize(image, (iw, ih))
        return image


def preprocess_input(image):
    image /= 255.0
    return image
