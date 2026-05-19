import logging

from medunda.tools.layers import compute_layer_height
from medunda.tools.lazy_imports import xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "compute_integral"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="Compute the integrated depth value of all variables with a 'depth' dimension",
    )


def compute_integral(data: "xr.Dataset") -> "xr.Dataset":
    """Compute the vertical integral of all depth-dependent variables.

    For each variable in the input dataset that includes a ``depth``
    dimension, the function integrates the variable over the full depth column
    by weighting each depth cell by its layer height (in metres).  Variables
    without a ``depth`` dimension are omitted from the output.

    Args:
        data (xr.Dataset): Input dataset.  Must include a ``depth``
            coordinate from which layer heights can be derived.

    Returns:
        xr.Dataset: Dataset containing the vertically integrated values for
        all depth-dependent variables.  The ``depth`` dimension is collapsed
        in the output.
    """
    layer_height = compute_layer_height(data.depth.values)
    lh = xr.DataArray(layer_height, dims=["depth"])

    ds_integrated = xr.Dataset()

    for var_name in data.data_vars:
        var = data[var_name]

        if "depth" in var.dims:
            weighted_avg = lh * var
            ds_integrated[var_name] = weighted_avg.sum(dim="depth")

    return ds_integrated
