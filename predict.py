import os
import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf

from models.yolo import YOLO
from models.yolov3 import YOLOv3Encoder, YOLOv3Decoder
from models.architectures.darknet53 import DarkNet53
from utils.post_processing import get_label_name
from visualizer.visual_image import visual_image, visual_image_with_bboxes
from configs import base_config as cfg


def resize_image(image, target_size, letterbox_image):
    h, w, _    = image.shape
    ih, iw, _  = target_size
    if letterbox_image:
        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_resized = cv2.resize(image, (nw, nh))
        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=image.dtype)
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        return image_paded
    else:
        image = cv2.resize(image, (iw, ih))
        return image


def preprocess_input(image):
    image /= 255.0
    return image


def detect_image(img_name, model, target_shape, class_names, crop=False, count=False, letterbox_image=False):
    num_classes = len(class_names)
    hsv_tuples  = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    image = cv2.imread(img_name)
    original_shape = image.shape

    image_data  = resize_image(image, target_shape, letterbox_image=True)
    image_data  = preprocess_input(image_data.astype(np.float32))
    image_data  = np.expand_dims(image_data, axis=0)
    
    input_image_shape = tf.expand_dims(tf.constant([original_shape[0], original_shape[1]], dtype=tf.float32), axis=0)
    
    out_boxes, out_scores, out_classes = model.predict([image_data, input_image_shape])
    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

    bbox_thick = int(0.6 * (original_shape[0] + original_shape[1]) / 1000)
    if bbox_thick < 1: bbox_thick = 1
    fontScale = 0.75 * bbox_thick

    if crop:
        dir_save_path = "./saved_weights/"
        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        for i, c in list(enumerate(out_boxes)):
            x_min, y_min, x_max, y_max = out_boxes[i]
            x_min = max(0, np.floor(x_min).astype('int32'))
            y_min = max(0, np.floor(y_min).astype('int32'))
            x_max = min(original_shape[1], np.floor(x_max).astype('int32'))
            y_max = min(original_shape[0], np.floor(y_max).astype('int32'))
            crop_image = image[y_min:y_max, x_min:x_max]
            cv2.imwrite(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), crop_image)
            print("save crop_" + str(i) + ".png to " + dir_save_path)
          
    if count:
        print("top_label:", out_classes)
        classes_nums    = np.zeros([num_classes])
        for i in range(num_classes):
            num = np.sum(out_classes == i)
            if num > 0:
                print(class_names[i], " : ", num)
            classes_nums[i] = num
        print("classes_nums:", classes_nums)


    for i, c in list(enumerate(out_classes)):
        predicted_class = get_label_name(class_names, int(c))
        box             = out_boxes[i]
        score           = out_scores[i]

        x_min, y_min, x_max, y_max = box
        x_min = max(0, np.floor(x_min).astype('int32'))
        y_min = max(0, np.floor(y_min).astype('int32'))
        x_max = min(original_shape[1], np.floor(x_max).astype('int32'))
        y_max = min(original_shape[0], np.floor(y_max).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        print(label, x_min, y_min, x_max, y_max)

        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[c], bbox_thick*2)

        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness=bbox_thick)
        # put filled text rectangle
        image = cv2.rectangle(image, (x_min, y_min), (x_min + text_width, y_min - text_height - baseline), colors[c], thickness=cv2.FILLED)

        # put text above rectangle
        image = cv2.putText(image, label, (x_min, y_min - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0,0,0), bbox_thick, lineType=cv2.LINE_AA)
    
    cv2.imwrite('./saved_weights/result.jpg', image)
    return image


classes = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8,
                      'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 12, 'parking meter': 13, 'bench': 14, 'bird': 15, 'cat': 16,
                      'dog': 17, 'horse': 18, 'sheep': 19, 'cow': 20, 'elephant': 21, 'bear': 22, 'zebra': 23, 'giraffe': 24,
                      'backpack': 25, 'umbrella': 26, 'handbag': 27, 'tie': 28, 'suitcase': 29, 'frisbee': 30, 'skis': 31, 'snowboard': 32,
                      'sports ball': 33, 'kite': 34, 'baseball bat': 35, 'baseball glove': 36, 'skateboard': 37, 'surfboard': 38, 'tennis racket': 39, 'bottle': 40,
                      'wine glass': 41, 'cup': 42, 'fork': 43, 'knife': 44, 'spoon': 45, 'bowl': 46, 'banana': 47, 'apple': 48,
                      'sandwich': 49, 'orange': 50, 'broccoli': 51, 'carrot': 52, 'hot dog': 53, 'pizza': 54, 'donut': 55, 'cake': 56,
                      'chair': 57, 'couch': 58, 'potted plant': 59, 'bed': 60, 'dining table': 61, 'toilet': 62, 'tv': 63, 'laptop': 64,
                      'mouse': 65, 'remote': 66, 'keyboard': 67, 'cell phone': 68, 'microwave': 69, 'oven': 70, 'toaster': 71, 'sink': 72,
                      'refrigerator': 73, 'book': 74, 'clock': 75, 'vase': 76, 'scissors': 77,  'teddy bear': 78, 'hair drier': 79, 'toothbrush': 80}
num_classes = len(classes)

backbone = DarkNet53(input_shape   = cfg.YOLO_TARGET_SIZE, 
                     activation    = cfg.YOLO_BACKBONE_ACTIVATION, 
                     norm_layer    = cfg.YOLO_BACKBONE_NORMALIZATION)

encoder = YOLOv3Encoder(backbone    = backbone,
                        num_classes = num_classes, 
                        num_anchor  = 3,
                        activation  = cfg.YOLO_ACTIVATION,
                        norm_layer  = cfg.YOLO_NORMALIZATION)

decoder = YOLOv3Decoder(anchors     = cfg.YOLO_ANCHORS,
                        num_classes = num_classes,
                        input_size  = cfg.YOLO_TARGET_SIZE,
                        anchor_mask = cfg.YOLO_ANCHORS_MASK,
                        max_boxes   = cfg.YOLO_MAX_BBOXES,
                        confidence  = cfg.TEST_CONFIDENCE_THRESHOLD,
                        nms_iou     = cfg.TEST_IOU_THRESHOLD,
                        letterbox_image=True)

model = YOLO(encoder, decoder)

load_type                          = "weights"

weight_objects                    = [        
                                    {
                                        'path': './saved_weights/20221024-235617/best_weights_mAP',
                                        'stage': 'full',
                                        'custom_objects': None
                                    }
                                ]

if load_type and weight_objects:
    if load_type == "weights":
        model.load_weights(weight_objects)
    elif load_type == "models":
        model.load_models(weight_objects)
        
image = "/home/vbpo/Desktop/TuNIT/working/Yolo/Yolo - pythonlessons/IMAGES/city.jpg"
img = detect_image(image, 
                   model, 
                   cfg.YOLO_TARGET_SIZE, 
                   classes, 
                   crop=True, 
                   count=False, 
                   letterbox_image=True)
