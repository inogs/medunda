import logging

import xarray as xr

from medunda.actions.averaging_between_layers import averaging_between_layers


LOGGER = logging.getLogger(__name__)
ACTION_NAME = "compute_depth_average"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="Compute the average on all the vertical levels"
    )


def compute_depth_average(data: xr.Dataset, output_file):
    """    Compute the average on all the vertical levels of the dataset.

    Args:
        data (xr.Dataset): Input dataset with depth as one of the dimensions.
        output_file (str): Path to save the output dataset.
    """
    LOGGER.info("Computing depth average")
    depth_min = data.depth.min().item()
    depth_max = data.depth.max().item()
    
    averaging_between_layers(data, output_file, depth_min - 1, depth_max + 1)
    LOGGER.info("Depth average computed and saved to %s", output_file)
