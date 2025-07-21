import logging


def configure_logger(logger):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

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
