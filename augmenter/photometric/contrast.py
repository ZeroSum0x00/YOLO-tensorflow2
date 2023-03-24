import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image


class Contrast:
    def __init__(self, factor):
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, bboxes):
        image = np.clip(127.5 + self.factor * (image - 127.5), 0, 255)
        return image, bboxes

    
class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5, prob=0.5):
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.aug = Contrast(factor=1.0)
        
    def __call__(self, image, bboxes):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.aug.factor = np.random.uniform(self.lower, self.upper)
            image, labels = self.aug(image, bboxes)
        return image, bboxes
