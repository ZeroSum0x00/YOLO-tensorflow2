import os
import cv2
import json
from tqdm import tqdm


class ParseCOCO:
    def __init__(self, 
                 data_dir, 
                 annotation_file   = None,
                 load_memory       = False, 
                 exclude_crowd     = True, 
                 exclude_difficult = True, 
                 exclude_truncated = False,
                 check_data        = False):
        self.data_dir          = data_dir
        self.annotation_path   = annotation_file
        json_file = open(annotation_file)
        self.annotation_data = json.load(json_file)
        json_file.close()
        self.COCO_images, self.COCO_segments, self.COCO_categories  = self._get_COCO_data()

        self.load_memory       = load_memory
        self.exclude_crowd = exclude_crowd
        self.check_data        = check_data

    def _get_COCO_data(self):
        COCO_images     = {}
        COCO_segments   = {}
        COCO_categories = {}
        for image in self.annotation_data['images']:
            image_id = image['id']
            if image_id in COCO_images:
                print("ERROR: Skipping duplicate image id: {}".format(image))
            else:
                COCO_images[image_id] = image
                
        for segmentation in self.annotation_data['annotations']:
            image_id = segmentation['image_id']
            if image_id not in COCO_segments:
                COCO_segments[image_id] = []
            COCO_segments[image_id].append(segmentation)
            
        for index, cat in enumerate(self.annotation_data['categories']):
            if cat['name'] == '_background_':
                continue
            COCO_categories[cat['id']] = index + 1

        COCO_image_keys = COCO_images.keys()
        COCO_segment_keys = COCO_segments.keys()
        for img_key in list(COCO_image_keys):
            if img_key not in COCO_segment_keys:
                del COCO_images[img_key]
        return COCO_images, COCO_segments, COCO_categories

    def __call__(self):
        data_extraction = []
        for image_id in tqdm(self.COCO_images.keys(), desc="Load dataset"):
            info_dict = {}
            info_dict['bboxes'] = []
            info_dict['image'] = []

            process_data = self.COCO_images[image_id]
            image_name  = process_data['file_name']
            info_dict['filename'] = image_name
            info_dict['image_size'] = tuple([process_data['height'], process_data['width']])

            if self.check_data:
                image_path  = os.path.join(self.data_dir, image_name)
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if len(img.shape) != 3:
                        print(f"Error: Image file {image_name} must be 3 channel in shape")
                        continue
                except Exception as e:
                    print(f"Error: File {image_name} is can't loaded: {e}")
                    continue

            if self.load_memory:
                image_path  = os.path.join(self.data_dir, image_name)
                img = cv2.imread(image_path, 1)
                info_dict['image'].append(img)

            target      = self.COCO_segments[image_id]
            crowd       = [x for x in target if ('iscrowd' in x and x['iscrowd'] and not self.exclude_crowd)]
            target      = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
            num_crowds  = len(crowd)
            target     += crowd

            if len(target) > 0:
                boxes_classes = []
                bbox = [0, 0, 0, 0, 0]
                for obj in target:
                    if obj['bbox']:
                        bbox        = obj['bbox']
                        final_box   = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]), self.COCO_categories[obj['category_id']] - 1]
                        info_dict['bboxes'].append(final_box)
                        
            if len(info_dict['bboxes']) > 0:
                data_extraction.append(info_dict)
        return data_extraction
