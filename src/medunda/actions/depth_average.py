import logging

import numpy as np
import xarray as xr

from medunda.actions.average_between_layers import average_between_layers

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "compute_average"

AXIS = {
    "depth": ["depth"],
    "space": ["latitude", "longitude"],
    "time": ["time"],
}


def configure_parser(subparsers):
    average_parser = subparsers.add_parser(
        ACTION_NAME, help="Compute the average"
    )
    average_parser.add_argument(
        "--axis",
        type=str,
        choices=["depth", "space", "time"],
        nargs="+",
        required=True,
        help="Choose the axis on which the average will be computed.",
    )


def get_volume():
    pass


def compute_depth_average(data: xr.Dataset, axis) -> xr.Dataset:
    """Compute the average on a given axis.

    Args:
        data (xr.Dataset): Input dataset with depth as one of the dimensions.
    """
    LOGGER.info(f"Computing average over axis '{axis}'")

    if axis == "depth":
        depth_min = data.depth.min().item()
        depth_max = data.depth.max().item()
        averaged_dataset = average_between_layers(
            data, depth_min - 1, depth_max + 1
        )
        LOGGER.info("Depth average computed successfully")

    elif axis == "space":
        mask = np.ma.getmaskarray(data.to_masked_array(copy=False))[0, :, :, :]

        weights = get_volume(data)
        weights = weights * (~mask)

        averaged_dataset = xr.Dataset()
        for var in data.data_vars:
            da = data[var]
            weighted_sum = (da * weights).sum(dim=["latitude", "longitude"])
            total_weights = weights.sum(dim=["latitude", "longitude"])
            averaged_dataset[var] = weighted_sum / total_weights

        LOGGER.info("Space average computed successfully")

    elif axis == "time":
        averaged_dataset = data.mean(dim="time")

        LOGGER.info("Time average computed successfully")

    else:
        raise ValueError(
            f"Axis '{axis}' is not valid. Choose from {list(AXIS.keys())}"
        )

    return averaged_dataset
