import cv2
import numpy as np


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

class Augment1:
    def __init__(self, target_size, max_boxes=10):
        self.max_boxes = max_boxes
        self.target_size = target_size

    def __call__(self, image, bboxes):
      h, w, _    = image.shape
      ih, iw, _  = self.target_size

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


class Augment2:
    def __init__(self, target_size, max_boxes=10, jitter=.3, hue=.1, sat=0.7, val=0.4):
        self.target_size = target_size
        self.max_boxes = max_boxes
        self.jitter    = jitter
        self.hue       = hue
        self.sat       = sat
        self.val       = val
    
    def __call__(self, image, bboxes):
      h, w, _    = image.shape
      ih, iw, _  = self.target_size

      new_ar = w / h * rand(1 - self.jitter, 1 + self.jitter) / rand(1 - self.jitter, 1 + self.jitter)
      scale = rand(.25, 2)

      if new_ar < 1:
          nh = int(scale * ih)
          nw = int(nh * new_ar)
      else:
          nw = int(scale * iw)
          nh = int(nw / new_ar)
          
      dw = int(rand(0, iw - nw))
      dh = int(rand(0, ih - nh))

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

      flip = rand() < .5
      if flip: 
          image = cv2.flip(image, 1)

      image_data      = np.array(image, np.uint8)

      r               = np.random.uniform(-1, 1, 3) * [self.hue, self.sat, self.val] + 1

      hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV))
      dtype           = image_data.dtype

      x       = np.arange(0, 256, dtype=r.dtype)
      lut_hue = ((x * r[0]) % 180).astype(dtype)
      lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
      lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

      image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
      image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2BGR)

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