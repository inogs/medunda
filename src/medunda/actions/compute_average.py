import logging
from collections.abc import Sequence

import medunda.tools.lazy_imports.bitsea.mask as bitsea
from medunda.actions.average_between_layers import average_between_layers
from medunda.tools.lazy_imports import np
from medunda.tools.lazy_imports import xr

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
        help="Minimum depth for the vertical average.",
    )
    average_parser.add_argument(
        "--depth-max",
        type=float,
        required=False,
        help="Maximum depth for the vertical average.",
    )


def get_area(data: "xr.Dataset") -> "xr.DataArray":
    """Compute the cell area"""

    data_var = list(data.data_vars)[0]

    if "time" in data[data_var].dims:
        reference = data[data_var].isel(time=0)
    else:
        reference = data[data_var]

    tmask = np.logical_not(np.isnan(reference))

    mask = bitsea.Mask.from_xarray(dataset=xr.Dataset({"tmask": tmask}))

    area = xr.DataArray(mask.area, dims=("latitude", "longitude"))

    return xr.DataArray(
        area,
        dims=("latitude", "longitude"),
        coords={
            "latitude": data.latitude,
            "longitude": data.longitude,
        },
    )


class Aggregation:
    def __init__(self, data: xr.Dataset):
        self.ds = data
        self.weights = get_area(data)

    def reduce_depth(self, data: xr.Dataset, depth_min=None, depth_max=None):
        if "depth" not in data.dims:
            LOGGER.warning("Depth dimensions not found, skipping depth")

        depth_min = float(data.depth.min()) if depth_min is None else depth_min
        depth_max = float(data.depth.max()) if depth_max is None else depth_max

        result = average_between_layers(data, depth_min, depth_max)

        return result

    def reduce_lat_lon(self, data: xr.Dataset):
        weights = self.weights.broadcast_like(data)

        averaged_dataset = xr.Dataset()

        for var in data.data_vars:
            da = data[var]

            mask = da.notnull()
            w = weights * mask

            weighted_sum = (da.fillna(0) * w).sum(
                dim=("latitude", "longitude")
            )

            total_weights = w.sum(dim=("latitude", "longitude"))

            averaged_dataset[var] = weighted_sum / total_weights

        result = averaged_dataset
        return result

    def reduce_lat(self, data: xr.Dataset):
        weights = self.weights

        if "time" in data.dims:
            weights = weights.expand_dims({"time": data.time})

        averaged_dataset = xr.Dataset()

        for var in data.data_vars:
            da = data[var]

            weighted_sum = (da * weights).sum(dim="latitude")

            total_weights = weights.sum(dim="latitude")

            averaged_dataset[var] = weighted_sum / total_weights

        result = averaged_dataset
        return result

    def reduce_lon(self, data: xr.Dataset):
        weights = self.weights

        if "time" in data.dims:
            weights = weights.expand_dims({"time": data.time})

        averaged_dataset = xr.Dataset()

        for var in data.data_vars:
            da = data[var]

            weighted_sum = (da * weights).sum(dim="longitude")

            total_weights = weights.sum(dim="longitude")

            averaged_dataset[var] = weighted_sum / total_weights

        result = averaged_dataset
        return result

    def reduce_time(self, data: xr.Dataset):
        result = data.mean(dim="time")
        return result

    def averaging(
        self, axes: Sequence[str], depth_min=None, depth_max=None
    ) -> xr.Dataset:
        result = self.ds

        if "depth" in axes:
            result = self.reduce_depth(result, depth_min, depth_max)

        if "latitude" in axes and "longitude" in axes:
            result = self.reduce_lat_lon(result)

        elif "latitude" in axes:
            result = self.reduce_lat(result)

        elif "longitude" in axes:
            result = self.reduce_lon(result)

        if "time" in axes:
            result = self.reduce_time(result)

        return result


def compute_average(data, axes, depth_min, depth_max):
    aggregator = Aggregation(data)

    return aggregator.averaging(
        axes=axes,
        depth_min=depth_min,
        depth_max=depth_max,
    )
