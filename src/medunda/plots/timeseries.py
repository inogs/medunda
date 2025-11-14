import logging

import matplotlib.pyplot as plt
import xarray as xr

from medunda.tools.argparse_utils import date_from_str

LOGGER = logging.getLogger(__name__)
PLOT_NAME = "plotting_timeseries"


def configure_parser(subparsers):
    plotting_timeseries = subparsers.add_parser(
        PLOT_NAME,
        help="Plots timeseries for a specific variable over a defined period of time",
    )
    plotting_timeseries.add_argument(
        "--start-time",
        type=date_from_str,
        required=True,
        help="Start date of the period to plot (format: YYYY-MM-DD)",
    )
    plotting_timeseries.add_argument(
        "--end-time",
        type=date_from_str,
        required=True,
        help="End date of the period to plot (format: YYYY-MM-DD)",
    )


def plotting_timeseries(
    data: xr.DataArray, metadata: dict, start_time, end_time
):
    start = start_time
    end = end_time

    if "time" not in data.dims:
        raise ValueError(
            "This dataset has no 'time' dimension therefore cannot plot time series."
        )

    # Aggregate spatial dims by mean over lat and lon if they exist
    spatial_dims = [
        dim
        for dim in ["lat", "latitude", "lon", "longitude"]
        if dim in data.dims
    ]
    data_mean = data.mean(dim=spatial_dims)

    ts = data_mean.sel(time=slice(start, end))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts.time, ts)

    ylabel = f"{metadata['label']}"
    if len(metadata["unit"]) > 0:
        ylabel += "[" + metadata["unit"] + "]"

    ax.set_title(f"Time Series of {metadata['label']}")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)

    plt.show()
