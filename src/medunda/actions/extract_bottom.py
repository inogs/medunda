import logging

from medunda.tools.lazy_imports import np
from medunda.tools.lazy_imports import xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "extract_bottom"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME, help="Extract the values of the cells on the bottom"
    )


def extract_bottom(data: "xr.Dataset") -> "xr.Dataset":
    """Extract the bottom-most valid grid cell for each spatial location.

    For each variable with a ``depth`` dimension, the function uses the
    land-sea mask (derived from the first time step) to identify the deepest
    unmasked (valid) depth level at every (latitude, longitude) grid point and
    returns the corresponding values.  Variables without a ``depth`` dimension
    are passed through unchanged.

    Args:
        data (xr.Dataset): Input dataset.  Must include a ``depth`` coordinate
            and at least one time step for each variable with a depth
            dimension.

    Returns:
        xr.Dataset: Dataset with the same variables as the input but with the
        ``depth`` dimension removed.  Each value corresponds to the deepest
        valid cell at the corresponding spatial location.
    """
    LOGGER.info(f"reading the file: {data}")

    variables = {}
    for var_name in data.data_vars:
        if var_name in ["depth", "latitude", "longitude", "time"]:
            continue
        LOGGER.debug("Computing variable %s", var_name)

        if "depth" not in data[var_name].dims:
            variables[var_name] = data[var_name]
            continue

        fixed_time_mask = np.ma.getmaskarray(
            data[var_name].isel(time=0).to_masked_array()
        )
        index_map = np.count_nonzero(~fixed_time_mask, axis=0)
        index_map_labeled = xr.DataArray(
            dims=["latitude", "longitude"], data=index_map
        )

        new_data = data[var_name][:, index_map_labeled - 1]

        variables[var_name] = new_data

    final_dataset = xr.Dataset(variables)

    return final_dataset
