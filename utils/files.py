import os
from utils.logger import logger


def extract_zip(archive, destination):
    import patoolib
    try:
        if not os.path.exists(destination):
            os.mkdir(destination)
        logger.info("Extract files to {}".format(destination))
        patoolib.extract_archive(archive, outdir=destination, verbosity=-1)
        logger.info("Extract Done!")
    except BaseException as e:
        logger.error("Error: ", e)


def verify_folder(folder):
    if folder[-1] != '/':
        folder += '/'
    return folder