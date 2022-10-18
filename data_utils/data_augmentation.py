from augmenter.augmentation import *
from configs import base_config as cfg


class Augmentor:
    def __init__(self, augment_objects, target_size=(416, 416, 3), max_bboxes=100, phase='train'):
        self.sequence_transform  = augment_objects[phase]['main']
        self.auxiliary_transform = augment_objects[phase]['auxiliary']
        self.merger_transform    = augment_objects[phase]['merge']

    def __call__(self, images, bboxes=None, auxiliary_image=None, auxiliary_bboxes=None):
        if self.sequence_transform:
            for transform in self.sequence_transform:
                images, bboxes = transform(images, bboxes)
            if auxiliary_image is not None:
                if self.auxiliary_transform:
                    for aux_transform in self.auxiliary_transform:
                        images2, bboxes2 = aux_transform(auxiliary_image, auxiliary_bboxes)
                else:
                    images2, bboxes2 = auxiliary_image, auxiliary_bboxes
                if self.merger_transform:
                    for merger in self.merger_transform:
                        images, bboxes = merger([images, images2], [bboxes, bboxes2])
        return images, bboxes


class EndemicAugmentor:
    def __init__(self, augment_objects, target_size=(416, 416, 3), max_bboxes=100, phase='train'):
        self.sequence_transform  = augment_objects[phase]['main']
        self.auxiliary_transform = augment_objects[phase]['auxiliary']
        self.merger_transform    = augment_objects[phase]['merge']

    def __call__(self, batch_image, batch_bboxes=None, auxiliary_image=None, auxiliary_bboxes=None):
        if self.sequence_transform:
            images1, bboxes1 = batch_image, batch_bboxes
            for transform in self.sequence_transform:
                images1, bboxes1 = transform(images1, bboxes1)
            if auxiliary_image is not None:
                if self.auxiliary_transform:
                    for aux_transform in self.auxiliary_transform:
                        images2, bboxes2 = aux_transform(auxiliary_image, auxiliary_bboxes)
                else:
                    images2, bboxes2 = auxiliary_image, auxiliary_bboxes
                if self.merger_transform:
                    for merger in self.merger_transform:
                        images, bboxes = merger([images1, images2], [bboxes1, bboxes2])
            else:
                images, bboxes = images1, bboxes1
            return images, bboxes
        else:
            return batch_image[0], batch_bboxes[0]
