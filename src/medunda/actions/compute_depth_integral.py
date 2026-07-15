import logging

from medunda.actions.reduce_axes import reduce_axes
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
        help="minimum limit of the layer (if not specified, the "
        "shallowest depth in the dataset is used)",
    )
    compute_depth_integral_parser.add_argument(
        "--depth-max",
        type=float,
        required=False,
        help="maximum limit of the layer (if not specified, the "
        "deepest depth in the dataset is used)",
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
    return reduce_axes(
        data=data,
        axes=["depth"],
        depth_min=depth_min,
        depth_max=depth_max,
        operator="integral",
    )
