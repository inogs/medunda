import logging
import time


def configure_logger(logger: logging.Logger):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Ensure that this formatter uses local time and not UTC
    formatter.converter = time.localtime

    logger.setLevel(logging.DEBUG)

    # Remove all handlers associated with the copernicusmarine logger
    copernicusmarine_logger = logging.getLogger("copernicusmarine")
    while copernicusmarine_logger.hasHandlers():
        copernicusmarine_logger.handlers.pop()

    logging.getLogger("botocore").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("h5py").setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
