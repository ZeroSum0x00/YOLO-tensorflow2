import os
import cv2
import colorsys
import numpy as np

from models.yolo import YOLO
from models.yolov3 import YOLOv3Encoder, YOLOv3Decoder

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
    original_shape = [image.shape[1], image.shape[0]]

    image_data  = resize_image(image, (target_shape[0], target_shape[1], 3), letterbox_image)
    image_data  = preprocess_input(image_data.astype(np.float32))
    image_data  = np.expand_dims(image_data, axis=0)
    
    out_boxes, out_scores, out_classes = model.predict(image_data, original_shape) 

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
            x_max = min(original_shape[0], np.floor(x_max).astype('int32'))
            y_max = min(original_shape[1], np.floor(y_max).astype('int32'))
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
        predicted_class = class_names[int(c)]
        box             = out_boxes[i]
        score           = out_scores[i]

        x_min, y_min, x_max, y_max = box
        x_min = max(0, np.floor(x_min).astype('int32'))
        y_min = max(0, np.floor(y_min).astype('int32'))
        x_max = min(original_shape[0], np.floor(x_max).astype('int32'))
        y_max = min(original_shape[1], np.floor(y_max).astype('int32'))

        label = '{} {:.2f}'.format(predicted_class, score)
        print(label, x_min, y_min, x_max, y_max)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colors[c], bbox_thick*2)

        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness=bbox_thick)
        # put filled text rectangle
        cv2.rectangle(image, (x_min, y_min), (x_min + text_width, y_min - text_height - baseline), colors[c], thickness=cv2.FILLED)

        # put text above rectangle
        cv2.putText(image, label, (x_min, y_min - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0,0,0), bbox_thick, lineType=cv2.LINE_AA)
    cv2.imwrite('./saved_weights/result.jpg', image)
    return image

classes_path                        = './saved_weights/voc_classes.txt'

train_annotation_path               = '/content/sample_data/OD_xml_tiny/train.txt'
val_annotation_path                 = '/content/sample_data/OD_xml_tiny/validation.txt'
anchors_path                        = './configs/yolo_anchors.txt'
anchors_mask                        = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

input_shape                         = [416, 416]

load_type                          = "weights"

weight_objects                    = [        
                                    {
                                        'path': './saved_weights/checkpoints/last_epoch_weights',
                                        'stage': 'full',
                                        'custom_objects': None
                                    }
                                ]

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

class_names, num_classes            = get_classes(classes_path)
anchors, num_anchors                = get_anchors(anchors_path)

encoder = YOLOv3Encoder(num_classes, num_anchor=3, darknet_weight=None)
decoder = YOLOv3Decoder(anchors,
              num_classes,
              input_shape,
              anchors_mask,
              max_boxes = 100,
              confidence = 0.3,
              nms_iou = 0.3,
              letterbox_image=True)

model = YOLO(encoder, decoder)

if load_type and weight_objects:
    if load_type == "weights":
        model.load_weights(weight_objects)
    elif load_type == "models":
        model.load_models(weight_objects)


image = "/home/vbpo/Desktop/TuNIT/working/Yolo/Yolo - pythonlessons/IMAGES/city.jpg"

img = detect_image(image, model, input_shape, class_names, crop=False, count=False, letterbox_image=True)
