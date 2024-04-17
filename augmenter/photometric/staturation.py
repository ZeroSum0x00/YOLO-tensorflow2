import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image


class Saturation:
    def __init__(self, factor):
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, bboxes):
        image[:,:,1] = np.clip(image[:,:,1] * self.factor, 0, 255)
        return image, bboxes

    
class RandomSaturation:
    def __init__(self, lower=0.3, upper=2.0, prob=0.5):
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob  = prob
        
    def __call__(self, image, bboxes):
        p = np.random.uniform(0,1)
        
        if p >= (1.0 - self.prob):
            factor = np.random.uniform(self.lower, self.upper)
            self.aug = Saturation(factor=factor)
            image, labels = self.aug(image, bboxes)
        return image, bboxes
