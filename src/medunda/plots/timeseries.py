import logging

from medunda.tools.argparse_utils import date_from_str
from medunda.tools.lazy_imports import pyplot as plt
from medunda.tools.lazy_imports import xr

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
    data: "xr.DataArray",
    metadata: dict,
    start_date,
    end_date,
    output_dir,
    show_plot,
):
    if "time" not in data.dims:
        raise ValueError(
            "This dataset has no 'time' dimension therefore cannot plot time series."
        )

    spatial_dims = [
        dim
        for dim in ["lat", "latitude", "lon", "longitude"]
        if dim in data.dims
    ]

    data_mean = data.mean(dim=spatial_dims)

    ts = data_mean.sel(time=slice(start_date, end_date))

    _, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts.time, ts)

    ylabel = f"{metadata['label']}"
    if len(metadata["unit"]) > 0:
        ylabel += "[" + metadata["unit"] + "]"

    ax.set_title(f"Time Series of {metadata['label']}")
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{metadata['label']}_timeseries.png")
    if show_plot:
        plt.show()
