import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils.calc_map import get_coco_map, get_map
from utils.post_processing import resize_image, preprocess_input
from utils.auxiliary_processing import change_color_space
from utils.logger import logger

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class mAPEvaluate(tf.keras.callbacks.Callback):
    def __init__(self, 
                 result_path    = None, 
                 input_shape    = None, 
                 classes        = None, 
                 max_bboxes     = 100, 
                 minoverlap     = 0.5,
                 mode           = 'voc',
                 color_space    = 'BGR',
                 min_ratio      = 0.2,
                 saved_best_map = True,
                 show_frequency = 100):
        super(mAPEvaluate, self).__init__()
        self.input_shape          = input_shape
        self.classes              = classes
        self.result_path          = result_path
        self.max_bboxes           = max_bboxes
        self.minoverlap           = minoverlap
        self.mode                 = mode
        self.color_space          = color_space
        self.min_ratio            = min_ratio
        self.saved_best_map       = saved_best_map
        self.show_frequency       = show_frequency
        self.map_out_path         = result_path + ".temp_map_out"
        self.maps                 = [0]
        self.epoches              = [0]
        self.current_map          = 0.0
        self.val_dataset          = None
        self.data_path            = None

    def pass_data(self, data):
        self.val_dataset = data
        self.data_path   = self.val_dataset.data_path

    def get_map_txt(self, image, original_image_shape, image_name, classes, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_name + ".txt"),"w") 
        input_image_shape = tf.expand_dims(tf.constant([original_image_shape[0], original_image_shape[1]], dtype=tf.float32), axis=0)

        image_data  = np.expand_dims(image, axis=0)

        out_boxes, out_scores, out_classes = self.model.predict([image_data, input_image_shape])
        out_boxes = out_boxes.numpy()
        out_scores = out_scores.numpy()
        out_classes = out_classes.numpy()
        
        top_100     = np.argsort(out_scores)[..., ::-1][:self.max_bboxes]
        out_boxes   = out_boxes[top_100]
        out_scores  = out_scores[top_100]
        out_classes = out_classes[top_100]
        
        for i, c in enumerate(out_classes):
            predicted_class             = self.classes[int(c)]
            try:
                score                   = str(out_scores[i].numpy())
            except:
                score                   = str(out_scores[i])
            x_min, y_min, x_max, y_max  = out_boxes[i]
            if predicted_class not in classes:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(x_min)), str(int(y_min)), str(int(x_max)), str(int(y_max))))

        f.close()
        return 
    
    def on_epoch_end(self, epoch, logs=None):
        temp_epoch = epoch + 1
        if temp_epoch % self.show_frequency == 0:
            if self.val_dataset is not None:
                if not os.path.exists(self.map_out_path):
                    os.makedirs(self.map_out_path)
                if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                    os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
                if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                    os.makedirs(os.path.join(self.map_out_path, "detection-results"))
                print("\nGet map.")
                
                for ann_dataset in tqdm(self.val_dataset.dataset):
                    img_name = ann_dataset['filename'].split('.')[0]
                    img_path = self.data_path + ann_dataset['filename']
                    
                    image = cv2.imread(img_path)
                    image = change_color_space(image, 'bgr', self.color_space)
                    original_image_shape = image.shape
                    
                    gt_boxes = ann_dataset['bboxes']
    
                    image  = resize_image(image, self.input_shape, letterbox_image=True)
                    image  = preprocess_input(image.astype(np.float32))
    
                    self.get_map_txt(image, original_image_shape, img_name, self.classes, self.map_out_path)
                    
                    with open(os.path.join(self.map_out_path, "ground-truth/" + img_name + ".txt"), "w") as new_f:
                        for box in gt_boxes:
                            x_min, y_min, x_max, y_max, obj = box
                            obj_name =  self.classes[int(obj)]
                            new_f.write("%s %s %s %s %s\n" % (obj_name, x_min, y_min, x_max, y_max))
                            
                print("Calculate Map.")
                
                if self.mode.lower() == 'coco':
                    map_result = get_coco_map(class_names=self.classes, path=self.map_out_path)[1]
                else:
                    map_result = get_map(self.minoverlap, False, path=self.map_out_path)
    
                if self.saved_best_map:
                    if map_result > self.current_map and map_result > self.min_ratio:
                        logger.info(f'mAP score increase {self.current_map*100:.2f}% to {map_result*100:.2f}%')
                        logger.info(f'Save best mAP weights to {os.path.join(self.result_path, "weights")}best_weights_mAP')                    
                        self.model.save_weights(os.path.join(self.result_path, "weights", 'best_weights_mAP'))
                        self.current_map = map_result
    
                self.maps.append(map_result)
                self.epoches.append(temp_epoch)
    
                with open(os.path.join(self.result_path, 'summary', "epoch_map.txt"), 'a') as f:
                    if epoch == 0:
                        f.write(f"mAP score in epoch 0: 0.0")
                    f.write(f"mAP score in epoch {epoch + 1}: {str(map_result*100)}")
                    f.write("\n")
                
                plt.figure()
                plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='mAP map')
    
                plt.grid(True)
                plt.xlabel('Epoch')
                plt.ylabel('Map %s'%str(self.minoverlap))
                plt.title('A Map Curve')
                plt.legend(loc="lower right")
    
                plt.savefig(os.path.join(self.result_path, 'summary', "epoch_map.png"))
                plt.cla()
                plt.close("all")
                shutil.rmtree(self.map_out_path)
            else:
                print('\nYou need to pass data in using the pass_data function.')
