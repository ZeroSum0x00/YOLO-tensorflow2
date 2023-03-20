import os
import cv2
import time
import random
import colorsys
import numpy as np
import tensorflow as tf

from models.yolo import YOLO
from models.architectures.darknet53 import DarkNet53, CSPDarkNet53
from models.yolov3 import YOLOv3
from utils.post_processing import get_labels
from visualizer.visual_image import visual_image, visual_image_with_bboxes
from configs import general_config as cfg
from predict import detect_image


video_path      = 0
video_save_path = "./b.mp4"
video_fps       = 25.0
video_path = "./a.mp4"

if __name__ == "__main__":
    classes, num_classes = cfg.YOLO_CLASSES, cfg.NUM_CLASSES

    backbone = DarkNet53(input_shape   = cfg.YOLO_TARGET_SIZE, 
                         activation    = cfg.YOLO_BACKBONE_ACTIVATION, 
                         norm_layer    = cfg.YOLO_BACKBONE_NORMALIZATION)

    architecture = YOLOv3(backbone     = backbone,
                          num_classes  = num_classes,
                          anchors      = cfg.YOLO_ANCHORS,
                          anchor_mask  = cfg.YOLO_ANCHORS_MASK,
                          activation   = cfg.YOLO_BACKBONE_ACTIVATION, 
                          norm_layer   = cfg.YOLO_BACKBONE_NORMALIZATION,
                          max_boxes    = cfg.YOLO_MAX_BBOXES,
                          confidence  = 0.5,
                          nms_iou      = cfg.TEST_IOU_THRESHOLD,
                          gray_padding = True)
        
    model = YOLO(architecture, image_size=cfg.YOLO_TARGET_SIZE)

    load_type                          = "weights"

    weight_objects                    = [        
                                        {
                                            'path': './saved_weights/best_weights/best_weights_mAP',
                                            'stage': 'full',
                                            'custom_objects': None
                                        }
                                    ]

    if load_type and weight_objects:
        if load_type == "weights":
            model.load_weights(weight_objects)
        elif load_type == "models":
            model.load_models(weight_objects)
    
    capture = cv2.VideoCapture(video_path)
    if video_save_path!="":
        fourcc  = cv2.VideoWriter_fourcc(*'XVID')
        size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    ref, frame = capture.read()
    if not ref:
        raise ValueError("Failed to read the camera (video) correctly, please pay attention to whether the camera is installed correctly (whether the video path is filled in correctly).")

    fps = 0.0
    while(True):
        t1 = time.time()
        ref, frame = capture.read()
        if not ref:
            break
        h, w, _ = frame.shape
        frame = detect_image(frame, 
                       model, 
                       cfg.YOLO_TARGET_SIZE, 
                       classes, 
                       crop=False, 
                       count=False, 
                       letterbox_image=True)

        fps  = (fps + (1./(time.time() - t1))) / 2
        print("fps= %.2f"%(fps))
        frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video",frame)
        c= cv2.waitKey(1) & 0xff 
        if video_save_path!="":
            out.write(frame)

        if c==27:
            capture.release()
            break

    print("Video Detection Done!")
    capture.release()
    if video_save_path!="":
        print("Save processed video to the path :" + video_save_path)
        out.release()
    cv2.destroyAllWindows()
