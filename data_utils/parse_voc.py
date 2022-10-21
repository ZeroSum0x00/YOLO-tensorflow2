import os
import cv2
import xml.etree.ElementTree as ET
from configs import base_config as cfg


class ParseVOC:
    def __init__(self, 
                 data_dir          = cfg.DATA_PATH, 
                 labels            = cfg.OBJECT_CLASSES,
                 load_memory       = cfg.DATA_LOAD_MEMORY, 
                 exclude_difficult = cfg.DATA_EXCLUDE_DIFFICULT, 
                 exclude_truncated = cfg.DATA_EXCLUDE_TRUNCATED,
                 check_data        = cfg.CHECK_DATA):
        self.data_dir          = data_dir
        self.labels            = labels
        self.load_memory       = load_memory
        self.exclude_difficult = exclude_difficult
        self.exclude_truncated = exclude_truncated
        self.check_data        = check_data
        
    def __call__(self, xml_files):
        data_extraction = []
        for xml_file in xml_files:
            xml_path = os.path.join(self.data_dir, xml_file)
            
            if self.check_data:
                image_file = xml_file.replace('xml', 'jpg')
                image_path = os.path.join(self.data_dir, image_file)
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if len(img.shape) != 3:
                        print(f"Error: Image file {image_file} must be 3 channel in shape")
                        continue
                    try:
                        with open(xml_path, 'rb') as f:
                            xml.sax.parse(f, xml.sax.ContentHandler())
                    except:
                        print(f"Error: XML file {xml_file} is missing or not in correct format")
                        continue
                except Exception as e:
                    print(f"Error: File {image_file} is can't loaded: {e}")
                    continue
                    
            xml_root = ET.parse(xml_path).getroot()
            # Initialise the info dict
            info_dict = {}
            info_dict['bboxes'] = []
            info_dict['image'] = []
            if self.load_memory:
                img = cv2.imread(os.path.join(self.data_dir, xml_file.replace('xml', 'jpg')))
                info_dict['image'].append(img)

            # Parse the XML Tree
            for elem in xml_root:
                if elem.tag == "filename":
                    info_dict['filename'] = elem.text
                elif elem.tag == "size":
                    image_size = []
                    for subelem in elem:
                        image_size.append(int(subelem.text))

                    info_dict['image_size'] = tuple(image_size)
                elif elem.tag == "object":
                    bbox = [0, 0, 0, 0, 0]
                    for subelem in elem:
                        if subelem.tag == "name":
                            bbox[4] = self.labels[subelem.text] if sorted(self.labels.values())[0] == 0 else self.labels[subelem.text] - sorted(self.labels.values())[0]
                        elif subelem.tag == "truncated" and self.exclude_truncated:
                            if int(subelem.text) == 1:
                                bbox = [0, 0, 0, 0, 0]
                                break
                        elif subelem.tag == "difficult" and self.exclude_difficult:
                            if int(subelem.text) == 1:
                                bbox = [0, 0, 0, 0, 0]
                                break
                        elif subelem.tag == "bndbox":
                            for subsubelem in subelem:
                                if 'xmin' in subsubelem.tag:
                                    bbox[0] = int(round(float(subsubelem.text)))
                                if 'ymin' in subsubelem.tag:
                                    bbox[1] = int(round(float(subsubelem.text)))
                                if 'xmax' in subsubelem.tag:
                                    bbox[2] = int(round(float(subsubelem.text)))
                                if 'ymax' in subsubelem.tag:
                                    bbox[3] = int(round(float(subsubelem.text)))
                    if bbox != [0, 0, 0, 0, 0]:
                        info_dict['bboxes'].append(bbox)
            if len(info_dict['bboxes']) > 0:
                data_extraction.append(info_dict)
        return data_extraction
