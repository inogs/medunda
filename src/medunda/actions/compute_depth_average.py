import logging

from medunda.actions.reduce_axes import reduce_axes
from medunda.tools.lazy_imports import xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "compute_depth_average"


def configure_parser(subparsers):
    compute_depth_average_parser = subparsers.add_parser(
        ACTION_NAME,
        help="Compute the vertical average between two specific depths",
    )
    compute_depth_average_parser.add_argument(
        "--depth-min",
        type=float,
        required=False,
        help="minimum limit of the layer (if not specified, the "
        "shallowest depth in the dataset is used)",
    )
    compute_depth_average_parser.add_argument(
        "--depth-max",
        type=float,
        required=False,
        help="maximum limit of the layer (if not specified, the "
        "deepest depth in the dataset is used)",
    )


def compute_depth_average(
    data: "xr.Dataset", depth_min, depth_max
) -> "xr.Dataset":
    """Compute the depth-weighted vertical average across the water column
    or between two specified depths.

    If ``depth_min`` and ``depth_max`` are not specified, the average is computed
    across the entire water column. In this case, the minimum and maximum depths
    are automatically determined from the dataset.

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
    return reduce_axes(
        data=data,
        axes=["depth"],
        depth_min=depth_min,
        depth_max=depth_max,
        operator="mean",
    )
