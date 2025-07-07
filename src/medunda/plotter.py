import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from medunda.sources.cmems import VARIABLES
from medunda.tools.layers import compute_layer_height 

def parse_args ():
    """
    parse command line arguments: 
    --input-file: path of the input file
    --output-dir: directory to save the download file
    """
    parser = argparse.ArgumentParser(
        description="Read the data downloaded by the downloader.py script and generate a 2D plot")

    parser.add_argument(   
        "--input-file",  
        type=Path,
        required=True,
        help="Path of the input file"
    )
    parser.add_argument(     
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where the downloaded file will be saved",
    )
    return parser.parse_args()

def compute_ch_integral (ds_chl):

    ds = xr.load_dataset(ds_chl)

    layer_height = compute_layer_height(ds.depth.values) 

    chl = ds['chl'] 
    chl_integrated = (chl * layer_height[:, np.newaxis, np.newaxis]).sum(dim='depth')

    return chl_integrated


def plot_timeseries (input_file): 
    
    pass


def plot_maps (input_file, var):

    pass

def extract_and_plot_layers(filepath: Path, variable: str):
    """Extracts and plots surface, bottom, and average layers of the given variable."""
    
    ds = xr.open_dataset(filepath)
    
    if not {'depth', 'latitude', 'longitude'}.issubset(ds[variable].dims) and not {
        'depth', 'lat', 'lon'}.issubset(data.dims):
        raise ValueError("The variable does not have the expected spatial dimensions (depth, lat/lon).")

    data=ds[variable]
    data=data.rename({'lat': 'latitude', 'lon':'longitude'}) if 'lat' in data else data
   
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plt.tight_layout()
    plt.show()

    ds.close()


def main ():
    args=parse_args()

    output_dir = args.output_dir

    frequency = args.frequency

    data_dir = output_dir / frequency

    if not data_dir.is_dir():
        raise ValueError(f"Unable to find the path {data_dir}")

    # Generate a list of all the files that are inside the data_dir
    data_file_list = list(data_dir.iterdir())

    # The file that we want is the only file inside the directory
    # TODO: change the logic so that the code is more robust and looks exactly
    # for the file that it needs
    data_file = data_file_list[0]

    extract_and_plot_layers(filepath=data_file,)


if __name__ == '__main__':
    main()
