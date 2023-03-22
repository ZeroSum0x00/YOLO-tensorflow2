import os
import cv2
import imagesize
from tqdm import tqdm
from configs import general_config as cfg


class ParseYOLO:
    def __init__(self, 
                 data_dir,
                 annotation_dir,
                 labels,
                 load_memory, 
                 check_data):
        self.data_dir          = data_dir
        self.annotation_dir    = annotation_dir if annotation_dir else data_dir
        self.labels            = labels
        self.load_memory       = load_memory
        self.check_data        = check_data
        
    def __call__(self, txt_files):
        data_extraction = []
        for txt_file in tqdm(txt_files, desc="Load dataset"):
            txt_path = os.path.join(self.annotation_dir, txt_file)

            if self.check_data:
                image_file = txt_file.replace('txt', 'jpg')
                image_path = os.path.join(self.data_dir, image_file)
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if len(img.shape) != 3:
                        print(f"Error: Image file {image_file} must be 3 channel in shape")
                        continue
                except Exception as e:
                    print(f"Error: File {image_file} is can't loaded: {e}")
                    continue
            
            open_file = open(txt_path, "r")
            raw_data = open_file.readlines()
            open_file.close()

            # Initialise the info dict
            info_dict = {}
            info_dict['bboxes'] = []
            info_dict['image'] = []
            
            image_name = txt_file.replace('txt', 'jpg')
            info_dict['filename'] = image_name
            width, height = imagesize.get(os.path.join(self.data_dir, image_name))
            info_dict['image_size'] = (height, width)
            
            if self.load_memory:
                img = cv2.imread(os.path.join(self.data_dir, image_name))
                info_dict['image'].append(img)

            if raw_data:
                for data in raw_data:
                    bbox = [0, 0, 0, 0, 0]
                    label, x_center, y_center, box_width, box_height = data.strip().split(' ')
                    bbox[0]    = float(x_center) * width
                    bbox[1]    = float(y_center) * height
                    bbox[2]    = float(box_width) * width
                    bbox[3]    = float(box_height) * height
                    bbox[4]    = int(label)

                    if bbox != [0, 0, 0, 0, 0]:
                        info_dict['bboxes'].append(bbox)
            if len(info_dict['bboxes']) > 0:
                data_extraction.append(info_dict)
        return data_extraction
