import logging

import numpy as np
import xarray as xr

from medunda.tools.layers import compute_layer_height


LOGGER = logging.getLogger(__name__)
ACTION_NAME = "averaging_between_layers"


def configure_parser(subparsers):
    averaging_between_layers_parser = subparsers.add_parser(
        ACTION_NAME,
        help="Compute the vertical average between two specific depths"
    )
    averaging_between_layers_parser.add_argument(
        "--depth-min",
        type=float,
        required=True,
        help="minimum limit of the layer"
    )
    averaging_between_layers_parser.add_argument(
        "--depth-max",
        type=float,
        required=True,
        help="maximum limit of the layer"
    )


def averaging_between_layers (data: xr.Dataset, depth_min, depth_max) -> xr.Dataset:
    """ Computes the vertical average of variables between two specified depths.
        Returns a dataset containing the weighted average of this strata.
    """
    averaged_variables = {}
    for variable in data.data_vars:
        if variable in ["depth", "latitude", "longitude", "time"]:
            continue

        selected_layer = data[variable].sel(depth=slice(depth_min, depth_max))
        selected_depth = selected_layer.depth.values

        layer_height = compute_layer_height (selected_depth)
        layer_height_extended = xr.DataArray (layer_height, dims=["depth"])

        mask = np.ma.getmaskarray(selected_layer.to_masked_array(copy=False))[0,:,:,:]
        mask_extended = xr.DataArray(
            mask,
            dims=("depth", "latitude", "longitude")
        )

        total_height = (layer_height_extended * ~mask_extended).sum(dim="depth")

        weighted_average = (selected_layer*layer_height_extended).sum(dim="depth", skipna=True) / total_height
        averaged_variables[variable] = weighted_average


    final_dataset = xr.Dataset(averaged_variables)
    
    return final_dataset
