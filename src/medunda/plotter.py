from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import xarray as xr

from sources.cmems import VARIABLES


def parse_args ():
    """
    parse command line arguments: 
    --variable: the variable to download:
    --frequency: choose the frequency of the download: monthly or annualy:
    --output-dir: directory to save the download file
    """
    parser = argparse.ArgumentParser(
        description="Read the data downloaded by the downloader.py script and generate a 2D plot")

    parser.add_argument(   
        "--variable",  
        type=str,
        choices=VARIABLES,
        required=True,
        help="Name of the variable to download"
    )

    parser.add_argument( 
        "--frequency",
        type=str,
        choices=["monthly", "daily"],
        required=False,
        default="monthly",
        help="frequency of the downloaded data"
    )

    parser.add_argument(      #input the directory to save the file
        "--output-dir",
        type=Path,
        default=Path("."),
        help="directory where the downloaded file will be saved",
    )
    return parser.parse_args()


def extract_and_plot_layers(filepath: Path, variable: str):
    """Extracts and plots surface, bottom, and average layers of the given variable."""
    
    ds = xr.open_dataset(filepath)
    
    surface = ds[variable].isel(depth=0, time=0)      # Surface layer (depth=0)
    bottom = ds[variable].isel(depth=-1, time=0)       # Bottom layer (last depth index)
    mean_layer = ds[variable].isel(time=0).mean(dim="depth", skipna=True)      # Mean over the water column

    if not {'depth', 'latitude', 'longitude'}.issubset(ds[variable].dims) and not {'depth', 'lat', 'lon'}.issubset(data.dims):
        raise ValueError("The variable does not have the expected spatial dimensions (depth, lat/lon).")

    data=ds[variable]
    data=data.rename({'lat': 'latitude', 'lon':'longitude'}) if 'lat' in data else data
   
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    surface.plot(ax=axes[0], cmap="coolwarm")
    axes[0].set_title("Surface Layer")

    bottom.plot(ax=axes[1], cmap="coolwarm")
    axes[1].set_title("Bottom Layer")

    mean_layer.plot(ax=axes[2], cmap="coolwarm")
    axes[2].set_title("Mean over Water Column")

    plt.tight_layout()
    plt.show()

    ds.close()


def main ():
    args=parse_args()

    output_dir = args.output_dir
    variable = args.variable
    frequency = args.frequency

    data_dir = output_dir / variable / frequency

    if not data_dir.is_dir():
        raise ValueError(f"Unable to find the path {data_dir}")

    # Generate a list of all the files that are inside the data_dir
    data_file_list = list(data_dir.iterdir())

    # The file that we want is the only file inside the directory
    # TODO: change the logic so that the code is more robust and looks exactly
    # for the file that it needs
    data_file = data_file_list[0]

    extract_and_plot_layers(filepath=data_file, variable=variable)


if __name__ == '__main__':
    main()
