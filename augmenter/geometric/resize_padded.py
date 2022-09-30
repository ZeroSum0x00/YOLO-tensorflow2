import cv2
import random
import numpy as np

from utils.auxiliary_processing import random_range
from visualizer.visual_image import visual_image, visual_image_with_bboxes


class ResizePadded:
    def __init__(self, target_size=(416, 416, 3), max_boxes=10, jitter=.3, flexible=False):
        self.target_size = target_size
        self.max_boxes = max_boxes
        self.jitter    = jitter
        self.flexible      = flexible

    def __call__(self, image, bboxes):
      h, w, _    = image.shape
      ih, iw, _  = self.target_size

      if not self.flexible:
          scale = min(iw/w, ih/h)
          nw, nh  = int(scale * w), int(scale * h)
          dw, dh = (iw - nw) // 2, (ih - nh) // 2
          image_resized = cv2.resize(image, (nw, nh))
          image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=image.dtype)
          image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized

          box_data = np.zeros((self.max_boxes, 5))
          if len(bboxes) > 0:
              np.random.shuffle(bboxes)
              bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + dw
              bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + dh
              bboxes[:, 0:2][bboxes[:, 0:2] < 0]   = 0
              bboxes[:, 2][bboxes[:, 2] > iw]      = iw
              bboxes[:, 3][bboxes[:, 3] > ih]      = ih
              box_w   = bboxes[:, 2] - bboxes[:, 0]
              box_h   = bboxes[:, 3] - bboxes[:, 1]
              bboxes  = bboxes[np.logical_and(box_w > 1, box_h > 1)]
              if len(bboxes) > self.max_boxes: bboxes = bboxes[:self.max_boxes]
              box_data[:len(bboxes)] = bboxes
          return image_paded, box_data

      new_ar = w / h * random_range(1 - self.jitter, 1 + self.jitter) / random_range(1 - self.jitter, 1 + self.jitter)
      scale = random_range(.25, 2)

      if new_ar < 1:
          nh = int(scale * ih)
          nw = int(nh * new_ar)
      else:
          nw = int(scale * iw)
          nh = int(nw / new_ar)
          
      dw = int(random_range(0, iw - nw))
      dh = int(random_range(0, ih - nh))

      image_resized = cv2.resize(image, (nw, nh))

      height = max(ih, nh + abs(dh))
      width = max(iw, nw + abs(dw))
      image_paded = np.full(shape=[height, width, 3], fill_value=128.0)
      if dw < 0 and dh >= 0:
          image_paded[dh:nh+dh, 0:nw, :] = image_resized
          if width == iw:
              image_paded = image_paded[:ih, :iw]
          else:
              image_paded = image_paded[:ih, abs(dw):abs(dw)+iw]
      elif dh < 0 and dw >= 0:
          image_paded[0:nh, dw:dw+nw, :] = image_resized
          if height == ih:
              image_paded = image_paded[:ih, :iw]
          else:
              image_paded = image_paded[abs(dh):abs(dh)+ih, :iw]
      elif dh < 0 and dw < 0:
          image_paded[0:nh, 0:nw, :] = image_resized
          if width == iw or height == ih:
              image_paded = image_paded[:ih, :iw]
          else:
              image_paded = image_paded[abs(dh):abs(dh)+ih, abs(dw):abs(dw)+iw]
      else:
          image_paded[dh:nh+dh, dw:dw+nw, :] = image_resized
          image_paded = image_paded[:ih, :iw]

      hpd, wpd, _ = image_paded.shape
      if hpd < ih or wpd <iw:
          image_temp = np.full(shape=[ih, iw, 3], fill_value=128.0)
          image_temp[:hpd, :wpd] = image_paded
          image_paded = image_temp


      image = image_paded

      flip = random_range() < .5
      if flip: 
          image = cv2.flip(image, 1)

      image_data      = np.array(image, np.uint8)

      box_data = np.zeros((self.max_boxes, 5))
      if len(bboxes) > 0:
          np.random.shuffle(bboxes)
          bboxes[:, [0,2]] = bboxes[:, [0, 2]]*nw/w + dw
          bboxes[:, [1,3]] = bboxes[:, [1, 3]]*nh/h + dh

          if flip: 
              bboxes[:, [0,2]] = iw - bboxes[:, [2,0]]

          bboxes[:, 0:2][bboxes[:, 0:2] < 0] = 0
          bboxes[:, 2][bboxes[:, 2] > iw] = iw
          bboxes[:, 3][bboxes[:, 3] > ih] = ih
          box_w = bboxes[:, 2] - bboxes[:, 0]
          box_h = bboxes[:, 3] - bboxes[:, 1]
          bboxes = bboxes[np.logical_and(box_w > 1, box_h > 1)]

          if len(bboxes) > self.max_boxes: 
              bboxes = bboxes[:self.max_boxes]

          box_data[:len(bboxes)] = bboxes

      return image_data, box_data
    
    
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

    augment = ResizePadded(flexible=True)
    images, bboxes = augment(image, bboxes)
    visual_image_with_bboxes([np.array(images).astype(np.float32)/255.0], [bboxes], ['result'], size=(20, 20))
