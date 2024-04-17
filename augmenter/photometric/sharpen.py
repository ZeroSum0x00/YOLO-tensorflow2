import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image, visual_image_with_bboxes


class Sharpen:
    def __init__(self, lightness=2, alpha=0.1, kernel=None, kernel_anchor=None):
        assert 0 <= alpha <= 1.0, "Alpha must be between 0.0 and 1.0"
        self.lightness        = lightness
        self.lightness_anchor = 8
        self.kernel           = np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1, -1]], dtype=np.float32) if kernel is None else kernel
        self.kernel_anchor    = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32) if kernel_anchor is None else kernel_anchor
        self.alpha            = alpha
        
    def __call__(self, image, bboxes):
        # alpha = np.random.uniform(*self.alpha)
        kernel = self.kernel_anchor  * (self.lightness_anchor + self.lightness) + self.kernel
        kernel -= self.kernel_anchor
        kernel = (1 - self.alpha) * self.kernel_anchor + self.alpha * self.kernel

        # Apply sharpening to each channel
        r, g, b = cv2.split(image.copy())
        r_sharp = cv2.filter2D(r, -1, kernel)
        g_sharp = cv2.filter2D(g, -1, kernel)
        b_sharp = cv2.filter2D(b, -1, kernel)

        # Merge the sharpened channels back into the original image
        img = cv2.merge([r_sharp, g_sharp, b_sharp])
        return img, bboxes


class RandomSharpen:
    def __init__(self, lightness_range=(0.75, 2.0), alpha_range=0.1, kernel=None, kernel_anchor=None, prob=0.5):
        self.lightness_range = lightness_range
        self.alpha_range = alpha_range
        self.kernel     = kernel
        self.kernel_anchor = kernel_anchor
        self.prob       = prob

    def __call__(self, image, bboxes):
        
        if isinstance(self.lightness_range, (list, tuple)):
            self.lightness = float(np.random.uniform(*self.lightness_range))
        else:
            self.lightness = float(np.random.uniform(0, self.lightness_range))
            
        if isinstance(self.alpha_range, (list, tuple)):
            self.alpha = float(np.random.uniform(*self.alpha_range))
        else:
            self.alpha = float(np.random.uniform(0, self.alpha_range))
            
        self.aug = Sharpen(self.lightness, self.alpha, self.kernel, self.kernel_anchor)
        
        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):
            image, bboxes = self.aug(image, bboxes)
        return image, bboxes