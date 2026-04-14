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
        "--start-date",
        type=date_from_str,
        required=False,
        help="Start date of the period to plot (format: YYYY-MM-DD)",
    )
    plotting_timeseries.add_argument(
        "--end-date",
        type=date_from_str,
        required=False,
        help="End date of the period to plot (format: YYYY-MM-DD)",
    )


def plotting_timeseries(
    data: xr.DataArray, metadata: dict, start_date, end_date
):
    if "time" not in data.dims:
        raise ValueError(
            "This dataset has no 'time' dimension therefore cannot plot time series."
        )

    if start_date is None and end_date is None:
        start_date = data["time"].values[0]
        end_date = data["time"].values[-1]
    else:
        if start_date is not None:
            start_date = start_date
        if end_date is not None:
            end_date = end_date

    spatial_dims = [
        dim
        for dim in ["lat", "latitude", "lon", "longitude"]
        if dim in data.dims
    ]

    data_mean = data.mean(dim=spatial_dims)

    ts = data_mean.sel(time=slice(start_date, end_date))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts.time, ts)

    ylabel = f"{metadata['label']}"
    if len(metadata["unit"]) > 0:
        ylabel += "[" + metadata["unit"] + "]"

    ax.set_title(f"Time Series of {metadata['label']}")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)

    plt.show()
