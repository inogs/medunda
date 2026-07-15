import logging

from medunda.tools.layers import compute_layer_height
from medunda.tools.lazy_imports import np
from medunda.tools.lazy_imports import xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "compute_depth_integral"


def configure_parser(subparsers):
    compute_depth_integral_parser = subparsers.add_parser(
        ACTION_NAME,
        help="Compute the vertical integral between two specific depths",
    )
    compute_depth_integral_parser.add_argument(
        "--depth-min",
        type=float,
        required=False,
        help="minimum limit of the layer",
    )
    compute_depth_integral_parser.add_argument(
        "--depth-max",
        type=float,
        required=False,
        help="maximum limit of the layer",
    )


def compute_depth_integral(
    data: "xr.Dataset", depth_min: float, depth_max: float
) -> "xr.Dataset":
    """Compute the vertical integral of variables across the water column
    or between two specified depths.

    For each variable in the input dataset that has a ``depth`` dimension, the
    function selects the depth levels within ``[depth_min, depth_max]`` and
    integrates over those levels by weighting each cell by its layer height.
    If ``depth_min`` and ``depth_max`` are not specified, the function integrates
    the variable aver the full depth column.
    Grid points that are masked (NaN) at the shallowest selected level are set
    to NaN in the output, preserving the land-sea mask.  Variables that do not
    have a ``depth`` dimension are omitted from the output.

    Args:
        data (xr.Dataset): Input dataset containing the variables to integrate.
            Must include a ``depth`` coordinate.
        depth_min (float): Upper bound of the depth range (shallowest depth).
        depth_max (float): Lower bound of the depth range (deepest depth).

    Returns:
        xr.Dataset: Dataset containing the vertically integrated values for all
        depth-dependent variables over the selected depth range.  The ``depth``
        dimension is collapsed in the output.
    """
    integrated_variables = {}
    for variable in data.data_vars:
        if variable in ["depth", "latitude", "longitude", "time"]:
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

        weighted_average = (selected_layer * layer_height_extended).sum(
            dim="depth", skipna=True
        )

        selection = {"depth": 0}
        if "time" in data.dims:
            selection["time"] = 0

        dims = tuple(d for d in weighted_average.dims)

        weighted_average = xr.where(
            np.isnan(data[variable].isel(**selection)),
            np.nan,
            weighted_average,
        ).transpose(*dims)

        integrated_variables[variable] = weighted_average

    final_dataset = xr.Dataset(integrated_variables)

    return final_dataset
