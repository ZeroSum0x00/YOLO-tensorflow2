import cv2
import colorsys
import numpy as np
from matplotlib import pyplot as plt


def generate_colors(classes):
    if classes:
        num_classes = len(classes)
        hsv_tuples  = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    else:
        colors = [(255, 0, 0)]
    return colors


def visual_image(imgs, titles=None, rows=1, columns=2, size=(10, 10), mode=None):
    figure = plt.figure(figsize=size)
    show_imgs = [imgs] if not isinstance(imgs, list) else imgs
    if titles is not None:
        show_titles = [titles] if not isinstance(titles, list) else titles
    else:
        if not isinstance(imgs, list):
            show_titles = ["show screen"]
        else:
            show_titles = [idx for idx in range(len(imgs))]

    for index, (img, title) in enumerate(zip(show_imgs, show_titles)):
        plt.subplot(rows, columns, index + 1)

        if not (np.min(img) > -1 and np.max(img) < 1):
            if np.any((img < 0)):
                img = np.clip(img, 0, 1)

        plt.title(title)
        if mode and mode.lower() == 'bgr2rgb':
            try:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except:
                plt.imshow(img)
        else:
            plt.imshow(img)


def draw_boxes_on_image(image, boxes, classes, colors):
    image_shape = image.shape
    bbox_thick = int(0.6 * (image_shape[0] + image_shape[1]) / 1000)
    if bbox_thick < 1: bbox_thick = 1
    fontScale = 0.75 * bbox_thick

    for idx, box in enumerate(boxes):
        x_min, y_min, x_max, y_max, cls = box
        if cls != -1.:
            x_min = max(0, np.floor(x_min).astype('int32'))
            y_min = max(0, np.floor(y_min).astype('int32'))
            x_max = min(image_shape[1], np.floor(x_max).astype('int32'))
            y_max = min(image_shape[0], np.floor(y_max).astype('int32'))
            cls   = int(cls)
            label = classes[cls] if classes else None
            color = colors[0] if len(colors) == 1 else colors[cls]
            cv2.rectangle(img=image, 
                          pt1=(x_min, y_min), 
                          pt2=(x_max, y_max), 
                          color=color,
                          thickness=2)
            if label:
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness=bbox_thick)
                image = cv2.rectangle(image, (x_min, y_min), (x_min + text_width, y_min - text_height - baseline), color, thickness=cv2.FILLED)
                image = cv2.putText(image, label, (x_min, y_min - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0,0,0), bbox_thick, lineType=cv2.LINE_AA)

    image = np.clip(image, 0, 1)
    return image


def visual_image_with_bboxes(images, bboxes=None, classes=None, titles=None, rows=1, columns=2, size=(10, 10), mode=None):
    colors = generate_colors(classes)
    images_copied = np.copy(images)
    bboxes_copied = np.copy(bboxes)
    figure = plt.figure(figsize=size)
    for index, (img, bbox, title) in enumerate(zip(images_copied, bboxes_copied, titles)):
        plt.subplot(rows, columns, index + 1)

        if not (np.min(img) > -1 and np.max(img) < 1):
            if np.any((img < 0)):
                img = np.clip(img, 0, 1)

        plt.title(title)
        if bbox is not None:
            img = draw_boxes_on_image(img, bbox, classes, colors)
        if mode and mode.lower() == 'bgr2rgb':
            try:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            except:
                plt.imshow(img)
        else:
            plt.imshow(img)

    plt.show()