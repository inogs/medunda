import logging

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from medunda.tools.argparse_utils import date_from_str

LOGGER = logging.getLogger(__name__)
PLOT_NAME = "plotting_maps"


def configure_parser(subparsers):
    plotting_maps = subparsers.add_parser(
        PLOT_NAME,
        help="Plots 2D maps for a specific variable at a chosen date",
    )
    plotting_maps.add_argument(
        "--time",
        type=date_from_str,
        required=True,
        help="Date to plot the map for (format: YYYY-MM-DD)",
    )

    plotting_maps.add_argument(
        "--aggregation-dimension",
        type=str,
        help="Dimension along which to aggregate the data"
        "By default, no aggregation is performed and the data is plotted as is."
        "If the data has more than 2 dimensions, an aggregation method must be specified using '--aggregation-method'",
    )
    plotting_maps.add_argument(
        "--aggregation-method",
        type=str,
        choices=["mean", "max", "min"],
        default="mean",
        help="Method to use for aggregating data along the specified dimension",
    )


def plotting_maps(
    data: xr.DataArray,
    metadata: dict,
    time,
    aggregation_dimension,
    aggregation_method,
):
    selected_time = pd.to_datetime(time)

    data["time"] = pd.to_datetime(data["time"].values)

    try:
        data_slice = data.sel(time=selected_time)
    except Exception:
        try:
            data_slice = data.sel(time=selected_time, method="nearest")
        except Exception as e:
            raise ValueError(f"Could not select time {selected_time}: {e}")

    n_dims = len(data_slice.dims)

    if n_dims == 2:
        if (
            "latitude" not in data_slice.dims
            or "longitude" not in data_slice.dims
        ):
            raise ValueError(
                "Data must have 'latitude' and 'longitude' dimensions for 2D plotting."
            )
        else:
            LOGGER.debug("Data has 2 dimensions, proceeding with plotting.")

        if aggregation_dimension:
            if aggregation_dimension in data_slice.dims:
                raise ValueError(
                    f"Aggregation dimension '{aggregation_dimension}' cannot be used for 2D data with dimensions {data_slice.dims}"
                )
            LOGGER.debug(
                f"Applying {aggregation_method} aggregation along dimension '{aggregation_dimension}'"
            )

            data_slice = getattr(data_slice, aggregation_method)(
                dim=aggregation_dimension
            )

    elif n_dims > 2:
        if not aggregation_method:
            raise ValueError(
                "Data has more than 2 dimensions, an aggregation method must be specified."
                " Please provide an aggregation method using the '--aggregation-method'"
            )

        if aggregation_dimension not in data_slice.dims:
            raise ValueError(
                f"Aggregation dimension '{aggregation_dimension}' not found in data dimensions: {data_slice.dims}"
            )

        LOGGER.info(
            f"Applying {aggregation_method} aggregation along dimension '{aggregation_dimension}'"
        )

        data_slice = getattr(data_slice, aggregation_method)(
            dim=aggregation_dimension
        )

    else:
        raise ValueError(
            f"Data has {n_dims} dimensions, expected at least 2 dimensions for plotting."
        )

    actual_time = pd.to_datetime(data_slice["time"].values)

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = metadata.get("cmap", "viridis")

    im = ax.pcolormesh(data_slice, cmap=cmap)

    title_str = f"{metadata['label']} at {actual_time.strftime('%Y-%m-%d')}"
    ax.set_title(title_str)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label(f"{metadata['label']} ({metadata.get('unit', '')})")

    plt.show()
