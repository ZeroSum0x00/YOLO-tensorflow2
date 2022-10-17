from augmenter.augmentation import *
from configs import base_config as cfg

class Augmentor:
    def __init__(self, target_size=(416, 416, 3), max_bboxes=100, phase='train'):
        self.sequence_transform = []
        self.auxiliary_transform = []
        if phase == 'train':
            resize_padded = ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=True)
            light_changed = LightIntensityChange(hue=.1, sat=0.7, val=0.4)
            self.sequence_transform = [resize_padded, light_changed]
            self.auxiliary_transform = ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=True)
        else:
            resize_padded = ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=False)          
            self.sequence_transform = [resize_padded]
            self.auxiliary_transform = ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=False)

        mixup = Mixup(target_size=target_size, max_bboxes=max_bboxes)
        self.merger_transform   = [mixup]

    def __call__(self, images, bboxes=None, auxiliary_image=None, auxiliary_bboxes=None):
        if self.sequence_transform:
            for transform in self.sequence_transform:
                images, bboxes = transform(images, bboxes)
            if auxiliary_image is not None:
                if self.auxiliary_transform:
                    images2, bboxes2 = self.auxiliary_transform(auxiliary_image, auxiliary_bboxes)
                else:
                    images2, bboxes2 = auxiliary_image, auxiliary_bboxes
                for merger in self.merger_transform:
                    images, bboxes = merger([images, images2], [bboxes, bboxes2])
        return images, bboxes


class EndemicAugmentor:
    def __init__(self, target_size=(416, 416, 3), max_bboxes=100, phase='train'):
        self.sequence_transform  = []
        self.auxiliary_transform = []
        if phase == 'train':
            mosaic = Mosaic(target_size=target_size, max_bboxes=max_bboxes)
            self.sequence_transform = [mosaic]
            self.auxiliary_transform = ResizePadded(target_size=target_size, max_boxes=max_bboxes, jitter=.3, flexible=True)

        mixup = Mixup(target_size=target_size, max_bboxes=max_bboxes)
        self.merger_transform   = [mixup]

    def __call__(self, batch_image, batch_bboxes=None, auxiliary_image=None, auxiliary_bboxes=None):
        if self.sequence_transform:
            for transform in self.sequence_transform:
                images1, bboxes1 = transform(batch_image, batch_bboxes)
            if auxiliary_image is not None:
                if self.auxiliary_transform:
                    images2, bboxes2 = self.auxiliary_transform(auxiliary_image, auxiliary_bboxes)
                else:
                    images2, bboxes2 = auxiliary_image, auxiliary_bboxes
                for merger in self.merger_transform:
                    images, bboxes = merger([images1, images2], [bboxes1, bboxes2])
            else:
                images, bboxes = images1, bboxes1
            return images, bboxes
        else:
            return batch_image[0], batch_bboxes[0]
