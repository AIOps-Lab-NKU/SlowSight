import logging
import sys


def get_logger(name):
    is_debug = True if sys.gettrace() else False
    logger = logging.getLogger(name)

    if is_debug:
        log_level = logging.DEBUG
        # logging.basicConfig(level=logging.DEBUG)
    else:
        log_level = logging.INFO
        # logging.basicConfig(level=logging.INFO)

    logger.propagate = False

    if not logger.hasHandlers():
        console = logging.StreamHandler()
        # formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")
        console.setFormatter(formatter)
        console.setLevel(log_level)
        logger.addHandler(console)
        logger.setLevel(log_level)
    return logger