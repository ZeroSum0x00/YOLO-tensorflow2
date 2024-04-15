import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image



class Brightness:
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, image, bboxes):
        image = np.clip(image + self.delta, 0, 255)
        return image, bboxes
    

class RandomBrightness:
    def __init__(self, lower=0.3, upper=2.0, prob=0.5):
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = float(lower)
        self.upper = float(upper)
        self.prob = prob
        self.aug = Brightness(delta=0)

    def __call__(self, image, bboxes):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.aug.delta = np.random.uniform(self.lower, self.upper)
            image, bboxes = self.aug(image, bboxes)
        return image, bboxes
