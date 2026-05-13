import logging

import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "extract_annual_extremes"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="extract the minimum and maximum value of a variable for each year",
    )


def _extract_annual_extremes(
    data: xr.Dataset, include_depth: bool
) -> xr.Dataset:
    """Extracts the annual extremes (maximum and minimum) from the dataset.

    This is an internal function that performs the actual extraction of annual extremes.
    The `include_depth` parameter determines whether the depth dimension should be included
    in the reduction process. If `include_depth` is True, the minimum and maximum values
    are computed across all time and spatial coordinates, including depth. If False, the
    depth dimension is not reduced, and the minimum and maximum values are computed for each
    depth layer separately.

    In this way, we have a single implementation of the extraction logic, and we can easily
    create different public functions.
    """
    if "time" not in data.coords:
        raise ValueError('The dataset must contain a "time" coordinate.')

    reduce_dims = ["time"]
    for dim in ["latitude", "longitude", "depth"]:
        if dim == "depth" and not include_depth:
            continue
        if dim in data.coords:
            reduce_dims.append(dim)

    grouped = data.groupby("time.year")
    ds_min = grouped.min(dim=reduce_dims).rename(
        {name: f"{name}_min" for name in data.data_vars}
    )
    ds_max = grouped.max(dim=reduce_dims).rename(
        {name: f"{name}_max" for name in data.data_vars}
    )

    years = ds_min["year"].to_numpy()
    time_values = pd.to_datetime(years.astype(str), format="%Y")
    ds_min = ds_min.rename(year="time").assign_coords(time=time_values)
    ds_max = ds_max.rename(year="time").assign_coords(time=time_values)

    output = xr.merge([ds_min, ds_max])
    return output


def extract_annual_extremes(data: xr.Dataset) -> xr.Dataset:
    """Extracts the annual extremes of a variable (maximum and minimum) from the dataset.

    This action groups the data by year and computes the minimum and maximum values for
    each variable across all time and spatial coordinates. The resulting dataset contains
    the annual minimum and maximum values for each variable, indexed by the starting date
    of each year.

    All the spatial coordinates (latitude, longitude, depth) are reduced
    in the process, so the output dataset only contains the "time" dimension, which
    corresponds to the years.

    For each variable in the input dataset, the output dataset
    will contain two variables: one for the minimum values (named "{variable}_min") and
    one for the maximum values (named "{variable}_max").
    """
    return _extract_annual_extremes(data, include_depth=True)
