import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range, coordinates_converter
from visualizer.visual_image import visual_image_with_bboxes


class Mosaic:
    def __init__(self, target_size=(416, 416, 3), coords="corners", max_bboxes=100, min_offset=[0.1, 0.9], jitter=0.3, padding_color=None):
        self.target_size   = target_size
        self.coords        = coords
        self.max_bboxes    = max_bboxes
        self.min_offset    = min_offset
        self.jitter        = jitter
        self.padding_color = padding_color
        
    @classmethod
    def merge_bboxes(cls, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def __call__(self, images, bboxes):
        ih, iw, _    = self.target_size
        min_offset_x = random_range(self.min_offset[0], self.min_offset[1])
        min_offset_y = random_range(self.min_offset[0], self.min_offset[1])

        image_datas = [] 
        box_datas   = []
        index       = 0
        fill_color  = self.padding_color if self.padding_color else [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        for image, box in zip(images, bboxes):
            if self.coords == "centroids":
                box = coordinates_converter(box, conversion="centroids2corners")

            h, w, _ = image.shape
            flip = random_range() < .5
            if flip and len(box) > 0: 
                image = cv2.flip(image, 1)
                box[:, [0,2]] = w - box[:, [2, 0]]        

            new_ar = w / h * random_range(1 - self.jitter, 1 + self.jitter) / random_range(1 - self.jitter, 1 + self.jitter)
            scale = random_range(.4, 1)

            if new_ar < 1:
                nh = int(scale * ih)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * iw)
                nh = int(nw / new_ar)

            if index == 0:
                dw = int(iw * min_offset_x) - nw
                dh = int(ih * min_offset_y) - nh
            elif index == 1:
                dw = int(iw * min_offset_x) - nw
                dh = int(ih * min_offset_y)
            elif index == 2:
                dw = int(iw * min_offset_x)
                dh = int(ih * min_offset_y)
            elif index == 3:
                dw = int(iw * min_offset_x)
                dh = int(ih * min_offset_y) - nh

            image_resized = cv2.resize(image, (nw, nh))
            height = max(ih, nh + abs(dh))
            width = max(iw, nw + abs(dw))
            image_paded = np.full(shape=[height, width, 3], fill_value=fill_color)

            if dw < 0 and dh >= 0:
                image_paded[dh:nh+dh, 0:nw, :] = image_resized
                if width == iw:
                    image_paded = image_paded[:ih, :iw]
                    box[:, [0, 2]] = box[:, [0, 2]] - (w/nw)*dw
                    box[:, [1, 3]] = box[:, [1, 3]]
                else:
                    image_paded = image_paded[:ih, abs(dw):abs(dw)+iw]
            elif dw >= 0 and dh < 0:
                image_paded[0:nh, dw:dw+nw, :] = image_resized
                if height == ih:
                    image_paded = image_paded[:ih, :iw]
                    box[:, [0, 2]] = box[:, [0, 2]]
                    box[:, [1, 3]] = box[:, [1, 3]] - (h/nh)*dh
                else:
                    image_paded = image_paded[abs(dh):abs(dh)+ih, :iw]
            elif dw < 0 and dh < 0:
                image_paded[0:nh, 0:nw, :] = image_resized
                if width == iw or height == ih:
                    box[:, [0, 2]] = box[:, [0, 2]] - (w/nw)*dw
                    box[:, [1, 3]] = box[:, [1, 3]] - (h/nh)*dh
                    image_paded = image_paded[:ih, :iw]
                else:
                    image_paded = image_paded[abs(dh):abs(dh)+ih, abs(dw):abs(dw)+iw]
            else:
                image_paded[dh:nh+dh, dw:dw+nw, :] = image_resized
                image_paded = image_paded[:ih, :iw]

            hpd, wpd, _ = image_paded.shape
            if hpd < ih or wpd <iw:
                image_temp = np.full(shape=[ih, iw, 3], fill_value=fill_color)
                image_temp[:hpd, :wpd] = image_paded
                image_paded = image_temp

            image = image_paded
            index = index + 1
            box_data = []

            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]]*nw/w + dw
                box[:, [1, 3]] = box[:, [1, 3]]*nh/h + dh
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > iw] = iw
                box[:, 3][box[:, 3] > ih] = ih            
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box

            image_datas.append(image)
            box_datas.append(box_data)

        cutx = int(iw * min_offset_x)
        cuty = int(ih * min_offset_y)

        new_image = np.zeros([ih, iw, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)
        if self.coords == "centroids":
            new_boxes = coordinates_converter(np.array(new_boxes), "corners2centroids")
            
        box_data = np.zeros((self.max_bboxes, 5))
        box_data[:, -1] = -1

        if len(new_boxes)>0:
            if len(new_boxes) > self.max_bboxes: 
                new_boxes = new_boxes[:self.max_bboxes]
            box_data[:len(new_boxes)] = new_boxes

        return new_image, box_data
    
    
if __name__ == "__main__":
    image_path1 = "/content/sample_data/voc_tiny/train/000017.jpg"
    image1      = cv2.imread(image_path1)
    bboxes1     = np.array([[185, 62, 279, 199, 14], [90, 78, 403, 336, 12]])


    image_path2 = "/content/sample_data/voc_tiny/train/000141.jpg"
    image2      = cv2.imread(image_path2)
    bboxes2     = np.array([[25, 53, 418, 399, 2]])


    image_path3 = "/content/sample_data/voc_tiny/train/000147.jpg"
    image3      = cv2.imread(image_path3)
    bboxes3     = np.array([[33, 105, 65, 146, 8],
                            [77, 114, 114, 149, 8],
                            [146, 125, 175, 150, 8],
                            [373, 121, 402, 153, 8],
                            [435, 112, 465, 149, 8]])


    image_path4 = "/content/sample_data/voc_tiny/train/000235.jpg"
    image4      = cv2.imread(image_path4)
    bboxes4     = np.array([[243, 255, 316, 375, 8],
                            [40, 266, 277, 375, 10],
                            [415, 192, 457, 254, 17]])


    batch_img = [image1, image2, image3, image4]
    batch_box = [bboxes1, bboxes2, bboxes3, bboxes4]


    augment = Mosaic()
    images, bboxes = augment(batch_img, batch_box)
    visual_image_with_bboxes([np.array(images).astype(np.float32)/255.0], [bboxes], ['result'], size=(20, 20))