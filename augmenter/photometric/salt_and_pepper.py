import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range


class SaltAndPepper:
    def __init__(self, phi=0.1):
        assert 0 <= phi <= 1.0, "phi must be between 0.0 and 1.0"
        self.phi = phi
        
    def __call__(self, image, bboxes):
        try:
            img = image.copy()
            dtype = img.dtype
            intensity_levels = 2 ** (img[0, 0].nbytes * 8)
            min_intensity, max_intensity = 0, intensity_levels - 1
            random_image = np.random.choice([min_intensity, 1, np.nan], p=[self.phi / 2, 1 - self.phi, self.phi / 2], size=img.shape)
            img = img.astype(np.float32) * random_image
            img = np.nan_to_num(img, nan=max_intensity).astype(dtype)
            return img, bboxes
        except:
            return image, bboxes


class RandomSaltAndPepper:
    def __init__(self, phi_range=0.05, prob=0.5):
        self.phi_range   = phi_range
        self.prob        = prob

    def __call__(self, image):
        if isinstance(self.phi_range, (list, tuple)):
            phi = float(np.random.choice(self.phi_range))
        else:
            phi = float(np.random.uniform(0, self.phi_range))

        aug = SaltAndPepper(phi)
        
        p = np.random.uniform(0, 1)
        if p >= (1.0-self.prob):
            image, bboxes = aug(image, bboxes)
        return image, bboxes