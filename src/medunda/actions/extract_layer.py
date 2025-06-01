import logging

import xarray as xr


LOGGER = logging.getLogger(__name__)
ACTION_NAME = "extract_layer"


def configure_parser(subparsers):
    extract_layer_parser = subparsers.add_parser(
        ACTION_NAME,
        help="Extract the values of a specific depth (in metres)"
    )
    extract_layer_parser.add_argument(
        "--depth",
        type=float,
        required=True,
        help="Depth of the layer that must be extracted"
    )


def extract_layer (input_file, output_file, depth):
    LOGGER.info(f"reading the file: {input_file}")
    with xr.open_dataset(input_file) as ds :
        #print(ds['depth'].values)
        bottom_layer = ds.sel(depth=depth, method="nearest")
        LOGGER.info(f"writing the file: {output_file}")
        bottom_layer.to_netcdf(output_file)
    LOGGER.info("done")
    print(bottom_layer.coords)
