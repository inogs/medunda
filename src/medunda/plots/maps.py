import logging

import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr 

from medunda.tools.argparse_utils import date_from_str

LOGGER = logging.getLogger(__name__)
PLOT_NAME = "plotting_maps"

def configure_parser(subparsers):
    plotting_maps = subparsers.add_parser(
        PLOT_NAME,
        help="Plots 2D maps for a specific variable at a chosen date"
    )
    plotting_maps.add_argument(
        "--time",
        type=date_from_str,
        required=True,
        help="Date to plot the map for (format: YYYY-MM-DD)"
    )
    plotting_maps.add_argument(
        "--latitude",
        type=float,
        required=False,
        help="Latitude of the area to plot"
    )
    plotting_maps.add_argument(
        "--longitude",
        type=float,
        required=False,
        help="Longitude of the area to plot"
    )

def plotting_maps (data: xr.DataArray, metadata: dict, time):
    
    selected_time = pd.to_datetime(time)
    
    data["time"] = pd.to_datetime(data["time"].values)

    try:
        data_slice = data.sel(time=selected_time)
    except:
        try:
            data_slice = data.sel(time=selected_time, method="nearest")
        except Exception as e:
            raise ValueError(f"Could not select time {selected_time}: {e}")

    non_spatial_dims = [dim for dim in data_slice.dims if dim not in ['lat', 'latitude', 'lon', 'longitude']]
    if non_spatial_dims:
        data_slice = data_slice.mean(dim=non_spatial_dims)

    fig, ax = plt.subplots(figsize=(10,5))

    cmap = metadata.get('cmap', 'viridis')

    im = ax.pcolormesh(data_slice, cmap=cmap)

    title_str = f"{metadata['label']} at {selected_time.strftime('%Y-%m-%d')}"
    ax.set_title(title_str)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label(metadata.get('unit', ''))

    plt.show()