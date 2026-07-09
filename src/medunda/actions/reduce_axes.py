import logging
from collections.abc import Sequence
from typing import Literal
from warnings import warn

import numpy as np

import medunda.tools.lazy_imports.bitsea.geodistances as bitsea_geodistances
import medunda.tools.lazy_imports.bitsea.grid as bitsea_grid
from medunda.tools.lazy_imports import xarray as xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "reduce_axes"


def configure_parser(subparsers):
    average_parser = subparsers.add_parser(
        ACTION_NAME,
        help="Reduce one or more axes of a dataset, by "
        "applying an operator on those axes",
    )
    average_parser.add_argument(
        "--axes",
        type=str,
        nargs="+",
        choices=["depth", "latitude", "longitude", "time"],
        required=True,
        help="Axes on which the operation will be computed.",
    )

    average_parser.add_argument(
        "--operation",
        type=str,
        choices=["mean", "integral", "max", "min"],
        required=False,
        default="mean",
        help="Operation to apply on the axes.",
    )

    average_parser.add_argument(
        "--depth-min",
        type=float,
        required=False,
        default=None,
        help="Ignore all the values of the dataset that are above this depth "
        "(in metres).",
    )
    average_parser.add_argument(
        "--depth-max",
        type=float,
        required=False,
        default=None,
        help="Ignore all the values of the dataset that are below this depth "
        "(in metres). ",
    )


def _drop_unused(ds: "xr.Dataset") -> "xr.Dataset":
    """Remove unused coordinates from a Dataset.

    A coordinate is considered unused if none of its dimensions are referenced
    by any data variable in the dataset. Removing such coordinates also causes
    any dimensions that are no longer referenced by any variable or coordinate
    to disappear automatically from the resulting dataset.

    Args:
        ds: The input ``xarray.Dataset``.

    Returns:
        A new ``xarray.Dataset`` with all unused coordinates removed.

    Notes:
        This function only removes coordinates. Dimensions are not explicitly
        dropped because, in xarray, they are defined by variables and
        coordinates. Once all references to a dimension are removed, the
        dimension disappears automatically.
    """
    # Dimensions used by data variables
    used_dims = set()
    for da in ds.data_vars.values():
        used_dims.update(da.dims)

    # Coordinates that should be kept
    used_coords = set(used_dims)

    for name, coord in ds.coords.items():
        if any(dim in used_dims for dim in coord.dims):
            used_coords.add(name)

    # Drop unused coordinates
    unused_coords = set(ds.coords) - used_coords
    return ds.drop_vars(unused_coords)


def compute_average(
    data: "xr.Dataset", axes: Sequence[str], compute_integral: bool = False
):
    """
    Computes the weighted average of data along specified axes, taking into
    account grid cell area and depth.

    This function calculates weights for averaging based on the grid
    configuration and aggregation needs.
    The function ensures the appropriate handling of cell areas and volumes,
    allowing precise computation of the weighted average for multidimensional
    data arrays.

    Args:
    data: The input dataset represented as an ``xarray.Dataset`` containing
        data variables and coordinates.
    axes: A list of string axis names along which the averaging should be
        performed (e.g., "latitude", "longitude", "depth").
    :param compute_integral: An optional flag to indicate whether an integral
        calculation is required instead of an average. If this is ``True``,
        we sum the values (along the required axes taking into account the
        relative weights). Defaults to ``False``.

    Returns:
         An ``xarray.Dataset`` object containing the aggregated data with
         axes reduced as defined by the input.
    """
    # We check if we have the information about the latitude and the longitude
    # and if we really need it. If this is not the case, we set the cell area
    # to 1.
    need_lat = "latitude" in data.coords and "latitude" in axes
    need_lon = "longitude" in data.coords and "longitude" in axes

    if need_lat and need_lon:
        LOGGER.debug(
            "Computing cell area because we must have both latitude"
            "and longitude and we must aggregate on both axes."
        )
        latitudes = data.latitude.values
        longitudes = data.longitude.values
        grid = bitsea_grid.RegularGrid(lat=latitudes, lon=longitudes)
        cell_area = grid.area
    elif need_lat:
        LOGGER.debug(
            'Using "latitude" as proxy for the cell area since we do not '
            "need to aggregate on the longitude axis."
        )
        latitudes = data.latitude.values
        cell_area = latitudes[:, None]
    elif need_lon:
        LOGGER.debug(
            'Using "longitude" as proxy for the cell area since we do not '
            "need to aggregate on the latitude axis."
        )
        longitudes = data.longitude.values
        cell_area = longitudes[None, :]
    else:
        LOGGER.debug(
            "Using 1 as proxy for the cell area since we do not need to "
            "aggregate on latitude nor on longitude."
        )
        cell_area = np.array([[1.0]])

    # Now we check if we also need the depth information
    if "depth" in data.coords and "depth" in axes:
        LOGGER.debug("Computing volumes because we must aggregate on depth.")
        depths = data.depth.values
        level_boundaries = bitsea_geodistances.extend_from_average(
            depths, 0, 0.0
        )
        e3t = level_boundaries[1:] - level_boundaries[:-1]
        volumes = e3t[:, None, None] * cell_area
    else:
        LOGGER.debug(
            "Using areas as volumes since we do not need to aggregate on "
            "depth."
        )
        volumes = cell_area[None, :, :]

    # Here we have the weights that we have to use when we must compute the
    # average
    weights = xr.DataArray(
        data=volumes,
        dims=["depth", "latitude", "longitude"],
    )
    new_data = data.copy()

    # For each variable, we need to understand along
    for var_name in data.data_vars:
        # Ignore variables that are coordinates
        if var_name in ("depth", "latitude", "longitude", "time"):
            continue
        LOGGER.debug("Aggregating variable %s", var_name)

        aggregated_axis = []
        for axis in data[var_name].dims:
            if axis in axes:
                aggregated_axis.append(axis)
        if len(aggregated_axis) == 0:
            LOGGER.debug(
                "Variable %s has axes %s and does not need to be aggregated",
                var_name,
                data[var_name].dims,
            )
            continue

        LOGGER.debug(
            "Aggregating variable %s along axes %s", var_name, aggregated_axis
        )

        # We remove from the weights the axes that are not needed for the
        # current variable
        not_needed_for_weights = []
        for axis in ("latitude", "longitude", "depth"):
            if axis not in aggregated_axis:
                not_needed_for_weights.append(axis)
        var_weights = weights.isel(
            **{dim: 0 for dim in not_needed_for_weights}
        )

        # Finally, we can perform the weighted average
        if compute_integral:
            operator = "sum"
        else:
            operator = "mean"

        new_data[var_name] = getattr(
            data[var_name].weighted(var_weights), operator
        )(dim=aggregated_axis)

    return _drop_unused(new_data)


def reduce_axes(
    data: "xr.Dataset",
    axes: Sequence[str],
    depth_min: float | None,
    depth_max: float | None,
    operator: Literal["mean", "integral", "max", "min"],
):
    valid_axis_names = ("depth", "latitude", "longitude", "time")
    for axis_name in axes:
        if axis_name not in valid_axis_names:
            raise ValueError(
                f"Invalid axis name: {axis_name}. "
                f"Valid axis names are: 'depth', 'latitude', 'longitude',"
                f"'time'."
            )

    # Cut the dataset to the depth limits if specified
    if depth_min is not None or depth_max is not None:
        LOGGER.debug(f"Depth limits specified: {depth_min} to {depth_max}")
        if "depth" not in data.dims:
            raise ValueError("Depth dimension not found in the dataset.")

        data_selected = data.sel(depth=slice(depth_min, depth_max))
    else:
        LOGGER.debug("No depth limits specified. Using the entire dataset.")
        data_selected = data

    # Keep only the axes that are in the dataset; raise a warning if there is
    # an axis that is not in the dataset.
    axes_raw = axes
    axes = []
    for axis_name in axes_raw:
        if axis_name not in data.coords:
            warn(
                f"An aggregation on axis {axis_name} has been requested "
                "but the dataset does not contain this axis.",
                UserWarning,
            )
        elif data.coords[axis_name].size == 1:
            warn(
                f"An aggregation on axis {axis_name} has been requested, "
                f"but the dataset has only one value for this axis.",
                UserWarning,
            )
        else:
            axes.append(axis_name)
    LOGGER.debug("Aggregating on axes: %s", axes)

    if operator == "mean" or operator == "integral":
        return compute_average(
            data_selected, axes, compute_integral=operator == "integral"
        )
    elif operator == "max":
        return data_selected.max(dim=axes)
    elif operator == "min":
        return data_selected.min(dim=axes)
    else:
        raise ValueError(f"Invalid operator: {operator}")
