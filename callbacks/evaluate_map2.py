import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils.calc_map import get_coco_map, get_map
from utils.post_processing import get_label_name, resize_image, preprocess_input
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class mAPEvaluate(tf.keras.callbacks.Callback):
    def __init__(self, input_shape, classes, val_lines, log_dir, map_out_path=".temp_map_out", max_boxes=100, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(mAPEvaluate, self).__init__()
        self.input_shape        = input_shape
        self.classes        = classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_map_txt(self, image_id, image, original_image_shape, classes, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")

        input_image_shape = tf.expand_dims(tf.constant([original_image_shape[0], original_image_shape[1]], dtype=tf.float32), axis=0)


        image_data  = np.expand_dims(image, axis=0)

        out_boxes, out_scores, out_classes = self.model.predict([image_data, input_image_shape])
        out_boxes = out_boxes.numpy()
        out_scores = out_scores.numpy()
        out_classes = out_classes.numpy()
        
        top_100     = np.argsort(out_scores)[::-1][:self.max_boxes]
        out_boxes   = out_boxes[top_100]
        out_scores  = out_scores[top_100]
        out_classes = out_classes[top_100]
        
        for i, c in enumerate(out_classes):
            predicted_class             = get_label_name(self.classes, int(c))
            try:
                score                   = str(out_scores[i].numpy())
            except:
                score                   = str(out_scores[i])
            x_min, y_min, x_max, y_max  = out_boxes[i]
            if predicted_class not in classes:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(x_min)), str(int(y_min)), str(int(x_max)),str(int(y_max))))

        f.close()
        return 
    
    def on_epoch_end(self, epoch, logs=None):
        temp_epoch = epoch + 1
        if temp_epoch % self.period == 0 and self.eval_flag:
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("\nGet map.")
            for annotation_line in tqdm(self.val_lines):
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]

                image       = cv2.imread(line[0])
                original_image_shape = image.shape
                gt_boxes    = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

                image  = resize_image(image, self.input_shape, self.letterbox_image)
                image  = preprocess_input(image.astype(np.float32))

                self.get_map_txt(image_id, original_image_shape, image, self.classes, self.map_out_path)
                
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = get_label_name(self.classes, obj)
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
                        
            print("Calculate Map.")
            temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)
            # try:
            #     temp_map = get_coco_map(class_names=self.class_names, path=self.map_out_path)[1]
            # except:
            #     temp_map = get_map(self.MINOVERLAP, False, path=self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(temp_epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)
