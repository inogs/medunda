import logging

import medunda.tools.lazy_imports.bitsea.mask as bitsea
from medunda.actions.average_between_layers import average_between_layers
from medunda.tools.lazy_imports import np
from medunda.tools.lazy_imports import xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "compute_average"

VALID_AXIS = {
    "depth": ["depth"],
    "space": ["latitude", "longitude"],
    "time": ["time"],
}


def configure_parser(subparsers):
    average_parser = subparsers.add_parser(
        ACTION_NAME, help="Compute the average"
    )
    average_parser.add_argument(
        "--axis",
        type=str,
        choices=sorted(VALID_AXIS),
        required=True,
        help="Choose the axis on which the average will be computed.",
    )


def get_volume(data: "xr.Dataset") -> "xr.DataArray":
    """Compute the cell volume"""
    data_var = list(data.data_vars)[0]
    if "time" in data[data_var].dims:
        reference = data[data_var].isel(time=0)
    else:
        reference = data[data_var]
    tmask = np.logical_not(np.isnan(reference))
    mask = bitsea.Mask.from_xarray(dataset=xr.Dataset({"tmask": tmask}))
    area = xr.DataArray(mask.area, dims=("latitude", "longitude"))
    e3t = xr.DataArray(mask.e3t, dims=("depth", "latitude", "longitude"))
    vol_cell = area * e3t
    return xr.DataArray(
        vol_cell.transpose("depth", "latitude", "longitude"),
        dims=("depth", "latitude", "longitude"),
        coords={
            "depth": data.depth,
            "latitude": data.latitude,
            "longitude": data.longitude,
        },
    )


def compute_average(data: "xr.Dataset", axis) -> "xr.Dataset":
    """Compute the average of all variables along a specified axis.

    Three axes are supported:

    * ``"depth"``: Computes the depth-weighted vertical average over the full
      depth column using :func:`~medunda.actions.average_between_layers.average_between_layers`.
    * ``"space"``: Computes a volume-weighted spatial average over all
      (latitude, longitude) grid points using the cell volumes derived from
      the grid mask.
    * ``"time"``: Computes a simple arithmetic mean over the time dimension.

    Args:
        data (xr.Dataset): Input dataset.  Must include ``depth``,
            ``latitude``, ``longitude``, and ``time`` coordinates as required
            by the chosen axis.
        axis (str): Axis along which to compute the average.  One of
            ``"depth"``, ``"space"``, or ``"time"``.

    Returns:
        xr.Dataset: Dataset with the chosen dimension collapsed, containing
        the averaged values for each variable.

    Raises:
        ValueError: If *axis* is not one of the valid choices.
    """
    if axis not in VALID_AXIS.keys():
        raise ValueError(
            f"Axis '{axis}' is not valid. Choose from {list(VALID_AXIS.keys())}"
        )

    LOGGER.info(f"Computing average over axis '{axis}'")

    if axis == "depth":
        depth_min = float(data.depth.min())
        depth_max = float(data.depth.max())
        averaged_dataset = average_between_layers(data, depth_min, depth_max)
        LOGGER.info("Depth average computed successfully")

    elif axis == "space":
        weights = get_volume(data)
        weights = weights.expand_dims({"time": data.time})

        averaged_dataset = xr.Dataset()

        for var in data.data_vars:
            da = data[var]

            weighted_sum = (da * weights).sum(dim=("latitude", "longitude"))
            total_weights = weights.sum(dim=("latitude", "longitude"))
            averaged_dataset[var] = weighted_sum / total_weights

        LOGGER.info("Space average computed successfully")

    elif axis == "time":
        averaged_dataset = data.mean(dim="time")

        LOGGER.info("Time average computed successfully")

    return averaged_dataset
