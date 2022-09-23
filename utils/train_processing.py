import os
import datetime
from utils.logger import logger
from utils.files import verify_folder


def create_folder_weights(saved_dir):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    TRAINING_TIME_PATH = saved_dir + current_time
    access_rights = 0o755
    try:
        os.makedirs(TRAINING_TIME_PATH, access_rights)
        logger.info("Successfully created the directory %s" % TRAINING_TIME_PATH)
        return verify_folder(TRAINING_TIME_PATH)
    except: 
        logger.error("Creation of the directory %s failed" % TRAINING_TIME_PATH)
