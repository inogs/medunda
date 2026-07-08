import logging
from warnings import warn

import numpy as np

import medunda.tools.lazy_imports.bitsea.geodistances as bitsea_geodistances
import medunda.tools.lazy_imports.bitsea.grid as bitsea_grid
from medunda.tools.lazy_imports import xarray as xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "compute_average"


def configure_parser(subparsers):
    average_parser = subparsers.add_parser(
        ACTION_NAME, help="Compute the average"
    )
    average_parser.add_argument(
        "--axes",
        type=str,
        nargs="+",
        choices=["depth", "latitude", "longitude", "time"],
        required=True,
        help="Axes on which the average will be computed.",
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


def compute_average(data, axes, depth_min, depth_max):
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
        if axis_name not in data_selected.coords:
            warn(
                f"An aggregation on axis {axis_name} has been requested "
                "but the dataset does not contain this axis.",
                UserWarning,
            )
        elif data_selected.coords[axis_name].size == 1:
            warn(
                f"An aggregation on axis {axis_name} has been requested, "
                f"but the dataset has only one value for this axis.",
                UserWarning,
            )
        else:
            axes.append(axis_name)
    LOGGER.debug("Aggregating on axes: %s", axes)

    # We check if we have the information about the latitude and the longitude
    # and if we really need it. If this is not the case, we set the cell area
    # to 1.
    need_lat = "latitude" in data_selected.coords and "latitude" in axes
    need_lon = "longitude" in data_selected.coords and "longitude" in axes

    if need_lat and need_lon:
        LOGGER.debug(
            "Computing cell area because we must have both latitude"
            "and longitude and we must aggregate on both axes."
        )
        latitudes = data_selected.latitude.values
        longitudes = data_selected.longitude.values
        grid = bitsea_grid.RegularGrid(lat=latitudes, lon=longitudes)
        cell_area = grid.area
    elif need_lat:
        LOGGER.debug(
            'Using "latitude" as proxy for the cell area since we do not '
            "need to aggregate on the longitude axis."
        )
        latitudes = data_selected.latitude.values
        cell_area = latitudes[:, None]
    elif need_lon:
        LOGGER.debug(
            'Using "longitude" as proxy for the cell area since we do not '
            "need to aggregate on the latitude axis."
        )
        longitudes = data_selected.longitude.values
        cell_area = longitudes[None, :]
    else:
        LOGGER.debug(
            "Using 1 as proxy for the cell area since we do not need to "
            "aggregate on latitude nor on longitude."
        )
        cell_area = np.array([[1.0]])

    # Now we check if we also need the depth information
    if "depth" in data_selected.coords and "depth" in axes:
        LOGGER.debug("Computing volumes because we must aggregate on depth.")
        depths = data_selected.depth.values
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
    new_data = data_selected.copy()

    # For each variable, we need to understand along
    for var_name in data_selected.data_vars:
        # Ignore variables that are coordinates
        if var_name in valid_axis_names:
            continue
        LOGGER.debug("Aggregating variable %s", var_name)

        aggregated_axis = []
        for axis in data_selected[var_name].dims:
            if axis in axes:
                aggregated_axis.append(axis)
        if len(aggregated_axis) == 0:
            LOGGER.debug(
                "Variable %s has axes %s and does not need to be aggregated",
                var_name,
                data_selected[var_name].dims,
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
        new_data[var_name] = (
            data_selected[var_name]
            .weighted(var_weights)
            .mean(dim=aggregated_axis)
        )

    return new_data
