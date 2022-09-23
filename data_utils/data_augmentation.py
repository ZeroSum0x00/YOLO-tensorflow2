from augmenter.augmentation import *
from configs import base_config as cfg

class Augmentor():
    def __init__(self, target_size=(416, 416, 3), max_bboxes_per_scale=cfg.YOLO_MAX_BBOX_PER_SCALE, aug_mode='train'):
        if aug_mode == 'train':
            self.aug = Augment2(target_size=target_size, max_boxes=max_bboxes_per_scale)
        else:            
            self.aug = Augment1(target_size=target_size, max_boxes=max_bboxes_per_scale)

        self.sequence = [self.aug]

    def __call__(self, image, bboxes=None):
        for transform in self.sequence:
            image, bboxes = transform(image, bboxes)
        return image, bboxes
