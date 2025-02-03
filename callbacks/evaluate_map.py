import os
import cv2
import shutil
import colorsys
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils.coco_map_calculator import get_coco_map
from utils.voc_map_calculator import get_voc_map
#from utils.aio_map_calculator import get_aio_map
from utils.post_processing import resize_image, preprocess_input
from utils.auxiliary_processing import change_color_space
from utils.logger import logger

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



class mAPEvaluate(tf.keras.callbacks.Callback):
    def __init__(self, 
                 result_path    = None, 
                 classes        = None, 
                 minoverlap     = 0.5,
                 eval_type      = 'VOC',
                 color_space    = 'BGR',
                 min_ratio      = 0.2,
                 saved_best_map = True,
                 show_top_care  = -1,
                 show_frequency = 100):
        super(mAPEvaluate, self).__init__()
        self.result_path          = result_path
        self.classes              = classes
        self.minoverlap           = minoverlap
        self.eval_type            = eval_type
        self.color_space          = color_space
        self.min_ratio            = min_ratio
        self.saved_best_map       = saved_best_map
        self.show_top_care        = show_top_care
        self.show_frequency       = show_frequency
        self.map_out_path         = result_path + ".temp_map_out"
        num_maps                  = 12 if eval_type.lower() == "coco" else 1
        self.maps                 = [[0.0] for i in range(num_maps)]
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
        
        top_100     = np.argsort(out_scores)[..., ::-1][:self.model.decoder.max_boxes]
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
        if self.classes is None:
            self.classes = self.model.classes

        temp_epoch = epoch + 1
        if temp_epoch % self.show_frequency == 0:
            if self.val_dataset is not None and self.classes:
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"), exist_ok=True)
                os.makedirs(os.path.join(self.map_out_path, "detection-results"), exist_ok=True)

                print("\nGet map.")
                
                for ann_dataset in tqdm(self.val_dataset.dataset):
                    img_name = ann_dataset['filename'].split('.')[0]
                    img_path = self.data_path + ann_dataset['filename']
                    
                    image = cv2.imread(img_path)
                    image = change_color_space(image, 'bgr', self.color_space)
                    original_image_shape = image.shape
                    
                    gt_boxes = ann_dataset['bboxes']
    
                    image  = resize_image(image, self.model.image_size, letterbox_image=True)
                    image  = preprocess_input(image.astype(np.float32))
    
                    self.get_map_txt(image, original_image_shape, img_name, self.classes, self.map_out_path)
                    
                    with open(os.path.join(self.map_out_path, "ground-truth/" + img_name + ".txt"), "w") as new_f:
                        for box in gt_boxes:
                            x_min, y_min, x_max, y_max, obj = box
                            obj_name =  self.classes[int(obj)]
                            new_f.write("%s %s %s %s %s\n" % (obj_name, x_min, y_min, x_max, y_max))
                            
                print("Calculate Map.")
                
                if self.eval_type.lower() == 'coco':
                    map_titles  = ['AP@0.50:0.95', 'AP@0.50', 'AP@0.75', 'AP@0.50:0.95[S]', 'AP@0.50:0.95[M]', 'AP@0.50:0.95[L]',
                                   'AR@0.50:0.95[d1]', 'AR@0.50:0.95[d10]', 'AR@0.50:0.95[d100]', 'AR@0.50:0.95[S]', 'AR@0.50:0.95[M]', 'AR@0.50:0.95[L]']
                    map_colors  = ['red', 'orange', 'gold', 'chartreuse', 'green', 'cyan', 'deepskyblue', 'blue', 'indigo', 'darkviolet', 'darkmagenta', 'deeppink']
                    map_results = get_coco_map(class_names=self.classes, path=self.map_out_path)
                else:
                    map_titles  = ['mAP@0.5']
                    map_colors  = ['red']
                    map_results = get_voc_map(self.minoverlap, False, path=self.map_out_path)

                for i in range(len(map_results)):
                    self.maps[i].append(map_results[i])
                        
                if self.saved_best_map:
                    if map_results[0] > self.current_map and map_results[0] > self.min_ratio:
                        logger.info(f'mAP score increase {self.current_map*100:.2f}% to {map_results[0]*100:.2f}%')
                        logger.info(f'Save best mAP weights to {os.path.join(self.result_path, "weights", "best_weights_mAP")}')                    
                        self.model.save_weights(os.path.join(self.result_path, "weights", "best_weights_mAP"))
                        self.current_map = map_results[0]
                    
                self.epoches.append(temp_epoch)
                with open(os.path.join(self.result_path, 'summary', "epoch_map.txt"), 'a') as f:
                    if len(map_results) == 1:
                        f.write(f"{map_titles[0]} score in epoch {epoch + 1}: {map_results[0]*100}\n")
                    else:
                        f.write(f"mAP score in epoch {epoch + 1}:\n")
                        for title, mAP in zip(map_titles, map_results):
                            f.write(f"\t{title}: {mAP * 100:.3f}\n")
                            
                f = plt.figure()
                max_height = np.max(self.maps)
                max_width  = np.max(self.epoches)
                for i in range(len(self.maps)):
                    max_index = np.argmax(self.maps[i])
                    if (self.show_top_care != -1) and (i not in self.show_top_care):
                        continue
                        
                    linewidth = 4 if i == 0 else 2
                    plt.plot(self.epoches, self.maps[i], linewidth=linewidth, color=map_colors[i], label=map_titles[i])
                        
                    if round(np.max(self.maps[i]), 3) <= 0.:
                        continue

                    temp_text = plt.text(0, 0, 
                                         f'{self.maps[i][max_index]:0.3f}', 
                                         alpha=0,
                                         fontsize=8, 
                                         fontweight=600,
                                         color='white')
                    r = f.canvas.get_renderer()
                    bb = temp_text.get_window_extent(renderer=r)
                    width = bb.width
                    height = bb.height
                    text = plt.text(self.epoches[max_index] + (width * 0.00027 + 0.01) * max_width, 
                                    self.maps[i][max_index] + (height * 0.0017 + 0.012) * max_height, 
                                    f'{self.maps[i][max_index]:0.3f}', 
                                    fontsize=8, 
                                    fontweight=600,
                                    color='white')
                    plt.gca().add_patch(
                        plt.Rectangle(
                            (self.epoches[max_index] + width * 0.00027 * max_width, self.maps[i][max_index] + height * 0.0017 * max_height),
                            width * 0.003 * max_width,
                            height * 0.005 * max_height,
                            # alpha=0.85,
                            facecolor='hotpink'
                    ))
                    plt.scatter(self.epoches[max_index], self.maps[i][max_index], s=80, facecolor='red')

                plt.grid(True)
                plt.xlabel('Epoch')
                plt.ylabel('mAP')
                plt.title('A mAP graph')
                plt.legend(fontsize=7, loc="upper left")
    
                plt.savefig(os.path.join(self.result_path, 'summary', "epoch_maps.png"))
                plt.cla()
                plt.close("all")
                shutil.rmtree(self.map_out_path)
            else:
                print('\nYou need to pass data in using the pass_data function.')
