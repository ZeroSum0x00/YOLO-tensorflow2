# YOLO hyper-parameters
OBJECT_CLASSES              = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                               "bottle": 5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                               "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                               "motorbike": 14, "person": 15, "pottedplant": 16,
                               "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}

YOLO_ACTIVATION             = 'leaky'

YOLO_NORMALIZATION          = 'batchnorm'

YOLO_BACKBONE_WEIGHTS       = "/home/vbpo/Desktop/TuNIT/working/Yolo/yolo-project/saved_weights/yolov3.weights"

YOLO_TARGET_SIZE            = [416, 416, 3]

YOLO_ANCHORS                = [[ 10,  13],
                               [ 16,  30],
                               [ 33,  23],
                               [ 30,  61],
                               [ 62,  45],
                               [ 59, 119],
                               [116,  90],
                               [156, 198],
                               [373, 326]]

YOLO_ANCHORS_MASK           = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

YOLO_STRIDES                = [8, 16, 32]

YOLO_MAX_BBOXES             = 100

YOLO_BALANCE_LOSS           = [0.4, 1.0, 4]

YOLO_BOX_RATIO_LOSS         = 0.05

YOLO_OBJ_RATIO_LOSS         = 5 * (YOLO_TARGET_SIZE[0] * YOLO_TARGET_SIZE[1]) / (416 ** 2)

YOLO_CLS_RATIO_LOSS         = 1 * (len(OBJECT_CLASSES) / 80)


# Training hyper-parameters
DATA_PATH                   = "/home/vbpo/Desktop/TuNIT/working/Yolo/yolo-project/datasets/VOC2017"

DATA_ANNOTATION_PATH        = None

DATA_DESTINATION_PATH       = None

DATA_AUGMENTATION           = {"train": "train", "validation": "validation", "test": "test"}

DATA_NORMALIZER             = 'divide'

DATA_TYPE                   = 'voc'

CHECK_DATA                  = False

DATA_LOAD_MEMORY            = False

DATA_EXCLUDE_DIFFICULT      = True

DATA_EXCLUDE_TRUNCATED      = False

TRAIN_BATCH_SIZE            = 8

TRAIN_EPOCHS                = 300

TRAIN_OPTIMIZER             = 'sgd'

TRAIN_MOMENTUM              = 0.937

TRAIN_NESTEROV              = True

TRAIN_LR_INIT               = 1e-2

TRAIN_LR_END                = 1e-4

TRAIN_WARMUP_EPOCH_RATIO    = 0.05

TRAIN_WARMUP_LR_RATIO       = 0.1

WITHOUT_AUG_EPOCH_RATIO     = 0.05

TRAIN_WEIGHT_TYPE           = None

TRAIN_WEIGHT_OBJECTS        = [        
                                {
                                  'path': './saved_weights/20220912-222333/best_weights',
                                  'stage': 'full',
                                  'custom_objects': None
                                }
                              ]

TRAIN_SHOW_FREQUENCY        = 10

TRAIN_SAVE_WEIGHT_FREQUENCY = 10

TRAIN_SAVED_PATH            = './saved_weights/'


# Inference (validation-testing-predict) hyper-parameters
TEST_CONFIDENCE_THRESHOLD   = 0.05

TEST_IOU_THRESHOLD          = 0.5

TEST_MIN_OVERLAP            = 0.5
