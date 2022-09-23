OBJECT_CLASSES = {"aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3,
                  "bottle": 4, "bus": 5, "car": 6, "cat": 7, "chair": 8,
                  "cow": 9, "diningtable": 10, "dog": 11, "horse": 12,
                  "motorbike": 13, "person": 14, "pottedplant": 15,
                  "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19}

YOLO_STRIDES                = [8, 16, 32]

YOLO_MAX_BBOX_PER_SCALE     = 100

YOLO_ANCHORS = [[ 10,  13],
                [ 16,  30],
                [ 33,  23],
                [ 30,  61],
                [ 62,  45],
                [ 59, 119],
                [116,  90],
                [156, 198],
                [373, 326]]

YOLO_ANCHORS_MASK                        = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

YOLO_ACTIVATION = 'leaky'
YOLO_BACTNORM = 'batchnorm'

# Training
DATA_PATH = "/home/vbpo/Desktop/TuNIT/working/Yolo/yolo-project/datasets/VOC2017"
#DATA_PATH = "/home/vbpo/Desktop/TuNIT/working/Yolo/yolo-project2/datasets/VOCTiny/voc_tiny"
DESTINATION_PATH = None
TRAIN_TARGET_SIZE = (416, 416, 3)
BATCH_SIZE = 8
AUGMENTATION = {"train": "train", "validation": "validation", "test": "test"}
NORMALIZER = 'divide'
DATA_TYPE = 'voc'
CHECK_DATA = False


TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45
TEST_MAP_PATH               = "./saved_weights/"



TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 2
TRAIN_EPOCHS                = 1000


LOAD_MEMORY = False
DATA_EXCLUDE_DIFFICULT = True
DATA_EXCLUDE_TRUNCATED = False
