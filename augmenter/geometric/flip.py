import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range, coordinates_converter
from visualizer.visual_image import visual_image, visual_image_with_bboxes


class Flip:
    def __init__(self, coords="corners", mode='horizontal', max_bboxes=100):
        self.coords     = coords
        self.mode       = mode
        self.max_bboxes = max_bboxes

    def __call__(self, image, bboxes):
        h, w, _ = image.shape
        horizontal_list = ['horizontal', 'h']
        vertical_list   = ['vertical', 'v']
        if self.mode.lower() in horizontal_list:
            image = cv2.flip(image, 1)
            if self.coords == "centroids":
                bboxes = coordinates_converter(bboxes, conversion="centroids2corners")
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]
            for bbox in bboxes:
                if bbox[0] > bbox[2]:
                    bbox[[0, 2]] = bbox[[2, 0]]
            if self.coords == "centroids":
                bboxes = coordinates_converter(bboxes, conversion="corners2centroids")
                
        elif self.mode.lower() in vertical_list:
            image = cv2.flip(image, 0)
            if self.coords == "centroids":
                bboxes = coordinates_converter(bboxes, conversion="centroids2corners")
            bboxes[:, [3,1]] = h - bboxes[:, [3,1]]
            for bbox in bboxes:
                if bbox[1] > bbox[3]:
                    bbox[[1, 3]] = bbox[[3, 1]]
            if self.coords == "centroids":
                bboxes = coordinates_converter(bboxes, conversion="corners2centroids")
                
        return image, bboxes


class RandomFlip:
    def __init__(self, coords="corners", mode='horizontal', max_bboxes=100, prob=0.5):
        self.coords     = coords
        self.mode       = mode
        self.max_bboxes = max_bboxes
        self.prob       = prob
        
    def __call__(self, image, bboxes):
        self.aug = Flip(coords=self.coords, mode=self.mode)
        p        = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            image, bboxes = self.aug(image, bboxes)
        return image, bboxes
      

if __name__ == "__main__":
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
    
    augment1 = Flip(mode='vertical')
    images1, bboxes1 = augment1(image, bboxes)
    visual_image_with_bboxes([np.array(images1).astype(np.float32)/255.0], [bboxes1], ['result'], size=(20, 20))

    augment2 = RandomFlip(prob=0.95, mode='vertical')
    images2, bboxes2 = augment2(image, bboxes)
    visual_image_with_bboxes([np.array(images2).astype(np.float32)/255.0], [bboxes2], ['result'], size=(20, 20))
