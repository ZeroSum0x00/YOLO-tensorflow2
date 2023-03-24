import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image


class Hue:
    def __init__(self, delta):
        if not (-180 <= delta <= 180): raise ValueError("`delta` must be in the closed interval `[-180, 180]`.")
        self.delta = delta

    def __call__(self, image, bboxes):
        image[:, :, 0] = (image[:, :, 0] + self.delta) % 180.0
        return image, bboxes
    

class RandomHue:
    def __init__(self, max_delta=18, prob=0.5):
        if not (0 <= max_delta <= 180): raise ValueError("`max_delta` must be in the closed interval `[0, 180]`.")
        self.max_delta = max_delta
        self.prob = prob
        self.aug = Hue(delta=0)

    def __call__(self, image, bboxes):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.aug.delta = np.random.randint(-self.max_delta, self.max_delta)
            image, bboxes = self.aug(image, bboxes)
        return image, bboxes
