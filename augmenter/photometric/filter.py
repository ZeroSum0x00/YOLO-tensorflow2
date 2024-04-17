import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range


class ErodeDilate:
    def __init__(self, kernel_size=(1, 1)):
        self.kernel = np.ones(kernel_size, np.uint8)
        
    def __call__(self, image, bboxes):
        img = image.copy()
        if np.random.rand() <= 0.5:
            img = cv2.erode(img, self.kernel, iterations=1)
        else:
            img = cv2.dilate(img, self.kernel, iterations=1)
        return img, bboxes


class RandomErodeDilate:
    def __init__(self, kernel_size=(1, 1), prob=0.5):
        self.kernel_size = kernel_size
        self.prob        = prob

    def __call__(self, image, bboxes):
        self.aug = ErodeDilate(self.kernel_size)
        
        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):
            image, bboxes = self.aug(image, bboxes)
        return image, bboxes