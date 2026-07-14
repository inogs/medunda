import logging

from medunda.tools.layers import compute_layer_height
from medunda.tools.lazy_imports import np
from medunda.tools.lazy_imports import xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "average_between_layers"


def configure_parser(subparsers):
    average_between_layers_parser = subparsers.add_parser(
        ACTION_NAME,
        help="Compute the vertical average between two specific depths",
    )
    average_between_layers_parser.add_argument(
        "--depth-min",
        type=float,
        required=False,
        help="minimum limit of the layer",
    )
    average_between_layers_parser.add_argument(
        "--depth-max",
        type=float,
        required=False,
        help="maximum limit of the layer",
    )


def average_between_layers(
    data: "xr.Dataset", depth_min, depth_max
) -> "xr.Dataset":
    """Compute the depth-weighted vertical average between two specified depths.

    For each variable in the input dataset that has a ``depth`` dimension, the
    function selects the depth levels within ``[depth_min, depth_max]`` and
    computes a weighted average, where the weight of each depth cell is its
    layer height.  Masked (NaN) cells are excluded from both the weighted sum
    and the normalisation, so the result is always a proper average of the
    valid cells.  Variables that do not have a ``depth`` dimension are passed
    through unchanged.

    Args:
        data (xr.Dataset): Input dataset containing the variables to average.
            Must include a ``depth`` coordinate.
        depth_min (float): Upper bound of the depth range (shallowest depth).
        depth_max (float): Lower bound of the depth range (deepest depth).

    Returns:
        xr.Dataset: Dataset with the same variables as the input, but with the
        ``depth`` dimension collapsed.  Each variable is replaced by its
        depth-weighted average over the selected depth range.
    """
    averaged_variables = {}
    for variable in data.data_vars:
        if variable in ["depth", "latitude", "longitude", "time"]:
            continue

        if "depth" not in data.data_vars[variable].dims:
            averaged_variables[variable] = data.data_vars[variable]
            continue

        if depth_min is None and depth_max is None:
            depth_min = data.depth.min().values
            depth_max = data.depth.max().values
        else:
            depth_min = depth_min
            depth_max = depth_max

        selected_layer = data[variable].sel(depth=slice(depth_min, depth_max))
        selected_depth = selected_layer.depth.values

        layer_height = compute_layer_height(selected_depth)
        layer_height_extended = xr.DataArray(layer_height, dims=["depth"])

        mask = np.ma.getmaskarray(selected_layer.to_masked_array(copy=False))[
            0, :, :, :
        ]
        mask_extended = xr.DataArray(
            mask, dims=("depth", "latitude", "longitude")
        )

        total_height = (layer_height_extended * ~mask_extended).sum(
            dim="depth"
        )

        weighted_average = (selected_layer * layer_height_extended).sum(
            dim="depth", skipna=True
        ) / total_height
        averaged_variables[variable] = weighted_average

    final_dataset = xr.Dataset(averaged_variables)

    return final_dataset
