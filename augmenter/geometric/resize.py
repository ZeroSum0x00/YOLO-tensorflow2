import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image, visual_image_with_bboxes


class Resize:
    def __init__(self, target_size=(416, 416, 3), max_bboxes=500, interpolation=None):
        self.target_size   = target_size[:-1] if len(target_size) == 3 else target_size
        self.max_bboxes    = max_bboxes
        self.interpolation = interpolation

    def __call__(self, image, bboxes):
        h, w, _    = image.shape
        ih, iw  = self.target_size
        scale = min(iw/w, ih/h)
        image = cv2.resize(image, dsize=self.target_size, interpolation=self.interpolation)
        flip = random_range() < .5
        if flip: 
            image = cv2.flip(image, 1)

        box_data = np.zeros((self.max_bboxes, 5))
        if len(bboxes) > 0:
            np.random.shuffle(bboxes)
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * iw / w
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * ih / h

            if flip: 
                bboxes[:, [0,2]] = iw - bboxes[:, [2,0]]

            bboxes[:, 0:2][bboxes[:, 0:2] < 0] = 0
            box_w = bboxes[:, 2] - bboxes[:, 0]
            box_h = bboxes[:, 3] - bboxes[:, 1]
            bboxes = bboxes[np.logical_and(box_w > 1, box_h > 1)]

            if len(bboxes) > self.max_bboxes: 
                bboxes = bboxes[:self.max_bboxes]

            box_data[:len(bboxes)] = bboxes
        return image, bboxes
      
      
image_path = "/content/sample_data/voc_tiny/train/000288.jpg"
image      = cv2.imread(image_path)
bboxes     = np.array([[443, 163, 479, 283, 14],
                        [419, 153, 439, 240, 14],
                        [440, 156, 450, 230, 14],
                        [412, 163, 430, 248, 14],
                        [407, 194, 422, 251, 14],
                        [373, 157, 401, 266, 14],
                        [330, 169, 380, 252, 14],
                        [1, 153, 20, 305, 14],
                        [1, 110, 33, 244, 14],
                        [30, 142, 77, 205, 6],
                        [67, 21, 382, 278, 5]])

if __name__ == "__main__":
    augment = Resize(target_size=(300, 300, 3))
    images, bboxes = augment(image, bboxes)
    visual_image_with_bboxes([np.array(images).astype(np.float32)/255.0], [bboxes], ['result'], size=(20, 20))
