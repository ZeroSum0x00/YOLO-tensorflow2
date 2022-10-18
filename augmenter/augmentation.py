import cv2
import numpy as np
from augmenter.augmentation import *
from configs import base_config as cfg


target_size=(416, 416, 3)
max_bboxes=100

basic_augmenter = {
    'train': {
        'main': [
            ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=True),
            RandomFlip(mode='horizontal'),
            LightIntensityChange(hue=.1, sat=0.7, val=0.4),
        ],
        'auxiliary': [ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=True)],
        'merge': [Mixup(target_size=target_size, max_bboxes=max_bboxes)]
    },
    'valid': {
        'main': [ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=False)],
        'auxiliary': [ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=False)],
        'merge': [Mixup(target_size=target_size, max_bboxes=max_bboxes)]
    },
    'test': {
        'main': [ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=False)],
        'auxiliary': [ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=False)],
        'merge': [Mixup(target_size=target_size, max_bboxes=max_bboxes)]
    }
}


endemic_augmenter = {
    'train': {
        'main': [
            Mosaic(target_size=target_size, max_bboxes=max_bboxes),
            RandomFlip(mode='horizontal'),
            LightIntensityChange(hue=.1, sat=0.7, val=0.4),
        ],
        'auxiliary': [ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=True)],
        'merge': [Mixup(target_size=target_size, max_bboxes=max_bboxes)]
    },
    'valid': {
        'main': [ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=False)],
        'auxiliary': [ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=False)],
        'merge': [Mixup(target_size=target_size, max_bboxes=max_bboxes)]
    },
    'test': {
        'main': [ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=False)],
        'auxiliary': [ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=False)],
        'merge': [Mixup(target_size=target_size, max_bboxes=max_bboxes)]
    }
}
