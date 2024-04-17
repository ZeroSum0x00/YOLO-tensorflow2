import cv2
import numpy as np

from utils.auxiliary_processing import coordinates_converter
from visualizer.visual_image import visual_image_with_bboxes


class Mixup:
    def __init__(self, target_size=(416, 416, 3), coords="corners", main_object_ratio=0.5, min_object_ratio=0.1, max_bboxes=500):
        assert coords in ('corners', 'centroids')
        self.target_size       = target_size
        self.coords            = coords
        self.main_object_ratio = main_object_ratio
        self.min_object_ratio  = min_object_ratio
        self.max_bboxes        = max_bboxes

    def __call__(self, images, bboxes):
        new_image   = 0.0
        n_sample    = len(images)
        bboxes_list = []
        remaining_object_ratio = (1. - self.main_object_ratio) / (n_sample - 1)

        for idx, (image, bbox) in enumerate(zip(images, bboxes)):
            h, w, _ = image.shape
            if h != self.target_size[0] or w != self.target_size[1]:
                image = cv2.resize(image, self.target_size[:-1])

            if idx == 0:
                current_ratio = self.main_object_ratio
                
            else:
                current_ratio = remaining_object_ratio

            new_image += np.array(image, np.float32) * current_ratio

            if current_ratio > self.min_object_ratio:
                if self.coords == "centroids":
                    bbox = coordinates_converter(bbox, conversion="centroids2corners")
                    
                if len(bbox) > 0:
                    for index, box in enumerate(bbox):
                        box[0] *= (self.target_size[1] / w)
                        box[1] *= (self.target_size[0] / h)
                        box[2] *= (self.target_size[1] / w)
                        box[3] *= (self.target_size[0] / h) 
                        bbox[index] = box
                        
                    if len(bbox) > self.max_bboxes: bbox = bbox[:self.max_bboxes]
                        
                if self.coords == "centroids":
                    bbox = coordinates_converter(bbox, conversion="corners2centroids")
                    box_wh    = bbox[:, 2:4]
                else:
                    box_wh    = bbox[:, 2:4] - bbox[:, 0:2]

                box_valid = box_wh[:, 0] > 0
                bboxes_list.append(bbox[box_valid, :])

        new_bboxes = np.concatenate(bboxes_list, axis=0)
        box_data = np.zeros((self.max_bboxes, 5))
        box_data[:, -1] = -1
        
        if len(new_bboxes) > 0:
            if len(new_bboxes) > self.max_bboxes: new_bboxes = new_bboxes[:self.max_bboxes]
            box_data[:len(new_bboxes)] = new_bboxes
        return new_image, box_data


if __name__ == "__main__":
    image_path1 = "/content/sample_data/voc_tiny/train/000555.jpg"
    image1      = cv2.imread(image_path1)
    bboxes1     = np.array([[2, 111, 262, 299, 17],
                            [266, 63, 332, 181, 14],
                            [160, 179, 422, 374, 14],
                            [1, 174, 186, 374, 14],
                            [61, 90, 213, 255, 14],
                            [157, 79, 261, 217, 14],
                            [270, 98, 339, 188, 8],
                            [422, 116, 500, 224, 8],
                            [380, 92, 499, 217, 14]])


    image_path2 = "/content/sample_data/voc_tiny/train/000699.jpg"
    image2      = cv2.imread(image_path2)
    bboxes2     = np.array([[29, 205, 337, 377, 11],
                            [102, 27, 366, 358, 11],
                            [309, 130, 472, 362, 11],
                            [1, 2, 500, 381, 8]])

    batch_img = [image1, image2]
    batch_box = [bboxes1, bboxes2]

    augment = Mixup()
    images, bboxes = augment(batch_img, batch_box)
    visual_image_with_bboxes([np.array(images).astype(np.float32)/255.0], [bboxes], ['result'], size=(20, 20))