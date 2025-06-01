import logging

import xarray as xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "extract_surface"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="Extract the values of the cells on the surface"
    )


def extract_surface (input_file, output_file):
    LOGGER.info(f"reading the file: {input_file}")
    with xr.open_dataset(input_file) as ds:
        surface_layer = ds.isel(depth=0)
        LOGGER.info(f"writing the file: {output_file}")
        surface_layer.to_netcdf(output_file)
        print(surface_layer)
