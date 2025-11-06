import logging

import xarray as xr

from medunda.actions.average_between_layers import average_between_layers

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "compute_depth_average"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="Compute the average on all the vertical levels"
    )


def compute_depth_average(data: xr.Dataset) -> xr.Dataset :
    """    Compute the average on all the vertical levels of the dataset.

    Args:
        data (xr.Dataset): Input dataset with depth as one of the dimensions.
    """
    LOGGER.info("Computing depth average")
    depth_min = data.depth.min().item()
    depth_max = data.depth.max().item()
    
    depth_average = average_between_layers(data, depth_min - 1, depth_max + 1)
    LOGGER.info("Depth average computed successfully")

    return depth_average
