import logging

import xarray as xr

from medunda.tools.layers import compute_layer_height

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "integrate_between_layers"


def configure_parser(subparsers):
    integrate_between_layers_parser = subparsers.add_parser(
        ACTION_NAME,
        help="Compute the vertical integral between two specific depths",
    )
    integrate_between_layers_parser.add_argument(
        "--depth-min",
        type=float,
        required=True,
        help="minimum limit of the layer",
    )
    integrate_between_layers_parser.add_argument(
        "--depth-max",
        type=float,
        required=True,
        help="maximum limit of the layer",
    )


def integrate_between_layers(
    data: xr.Dataset, depth_min, depth_max
) -> xr.Dataset:
    """Computes the vertical integral of variables between two specified depths.
    Returns a dataset containing the weighted average of this strata.
    """
    integrated_variables = {}
    for variable in data.data_vars:
        if variable in ["depth", "latitude", "longitude", "time"]:
            continue

        selected_layer = data[variable].sel(depth=slice(depth_min, depth_max))
        selected_depth = selected_layer.depth.values

        layer_height = compute_layer_height(selected_depth)
        layer_height_extended = xr.DataArray(layer_height, dims=["depth"])

        weighted_average = (selected_layer * layer_height_extended).sum(
            dim="depth", skipna=True
        )
        integrated_variables[variable] = weighted_average

    final_dataset = xr.Dataset(integrated_variables)

    return final_dataset
