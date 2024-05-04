import os
import cv2
from tqdm import tqdm


class ParseTXT:
    def __init__(self, 
                 data_dir,
                 annotation_file,
                 load_memory,
                 check_data,
                 exclude_difficult = False, 
                 exclude_truncated = False,
                ):
        self.data_dir          = data_dir
        txt_file = open(annotation_file, "r")
        self.annotation_data   = txt_file.readlines()
        txt_file.close()
        self.load_memory       = load_memory
        self.check_data        = check_data

    def __call__(self):
        data_extraction = []
        for data in tqdm(self.annotation_data, desc="Load dataset"):
            info_dict = {}
            info_dict['bboxes'] = []
            info_dict['image'] = []
            data = data.strip().split(' ')
            image_file = data.pop(0)
            info_dict['filename'] = image_file
            
            if self.check_data:
                image_path = os.path.join(self.data_dir, image_file)
                try:
                    # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    img = cv2.imread(image_path, 1)
                    if len(img.shape) != 3:
                        print(f"Error: Image file {image_file} must be 3 channel in shape")
                        continue
                except Exception as e:
                    print(f"Error: File {image_file} is can't loaded: {e}")
                    continue
                    
            if self.load_memory:
                img = cv2.imread(os.path.join(self.data_dir, image_file))
                info_dict['image'].append(img)

            for data_object in data:
                data_object = data_object.split(',')
                bbox  = [0, 0, 0, 0, 0]
                bbox[0]  = int(data_object[0])
                bbox[1]  = int(data_object[1])
                bbox[2]  = int(data_object[2])
                bbox[3]  = int(data_object[3])
                bbox[4] = int(data_object[4])
                
                if bbox != [0, 0, 0, 0, 0]:
                    info_dict['bboxes'].append(bbox)

            if len(info_dict['bboxes']) > 0:
                data_extraction.append(info_dict)
        return data_extraction
