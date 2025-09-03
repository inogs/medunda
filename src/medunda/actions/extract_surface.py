import logging

import xarray as xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "extract_surface"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="Extract the values of the cells on the surface"
    )


def extract_surface (data: xr.Dataset) -> xr.Dataset:
    LOGGER.info(f"reading the file: {data}")
    
    surface_layer = data.isel(depth=0)
    
    LOGGER.info(f"Surface layer extraction completed.")

    return surface_layer