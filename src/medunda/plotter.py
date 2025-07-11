import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import cmocean
import numpy as np
import xarray as xr


from medunda.sources.cmems import VARIABLES
from medunda.tools.argparse_utils import date_from_str
from medunda.tools.layers import compute_layer_height
from medunda.tools.logging_utils import configure_logger

LOGGER = logging.getLogger(__name__)

VAR_METADATA = {
    "o2": {'label': 'Oxygen',                 
            'unit':'µmol/m³',
            'cmap': cmocean.cm.deep},
    "chl": {'label': 'Chlorophyll-a',       
            'unit':'mg/m³',
            'cmap':cmocean.cm.algae},
    "nppv": {'label': 'Net Primary Production',
            'unit':'mg C/m²/day',   
            'cmap':cmocean.cm.matter},
    "thetao": {'label': 'Temperature', 
               'unit':'°C', 
               'cmap':'coolwarm'},
    "so": {'label': 'Salinity', 
           'unit': 'PSU',
           'cmap':'viridis'},
}


def parse_args ():
    """
    parse command line arguments: 
    --input-file: path of the input file
    --variable: name of the variable to plot
    --mode: type of the plot that can be either 'time series' or 'maps'
    --output-dir: directory to save the download file
    """
    parser = argparse.ArgumentParser(
        description="plots timeseries and maps")    ######

    parser.add_argument(   
        "--input-file",  
        type=Path,
        required=True,
        help="Path of the input file"
    )
    parser.add_argument(   
        "--variable",  
        type=str,
        choices=VARIABLES,
        required=True,
        help="Name of the variable to plot"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["timeseries", "maps"],
        required=True,
        help="Type of the plot that can be either 'time series' or 'maps' "
    )
    parser.add_argument(   
        "--time",
        type=date_from_str,
        required=True,
        help="The time of the plot"
    )
    parser.add_argument(     
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where the downloaded files are saved",
    )
    return parser.parse_args()


def check_variable (ds, var):

    if var not in ds:
        raise ValueError(f"Variable '{var}' not found in dataset")
    if var not in VAR_METADATA:
        raise ValueError(f"Metadata for variable '{var}' is missing")
    return ds[var], VAR_METADATA[var]


def compute_ch_integral (ds):

    ds = xr.load_dataset(ds)

    layer_height = compute_layer_height(ds.depth.values) 

    chl = ds['chl'] 
    chl_integrated = (chl * layer_height[:, np.newaxis, np.newaxis]).sum(dim='depth')

    return chl_integrated


def plot_timeseries (data:xr.DataArray, metadata: dict):

    # Aggregate spatial dims by mean over lat and lon if they exist
    spatial_dims = [dim for dim in ['lat', 'latitude', 'lon', 'longitude'] if dim in data.dims]
    if spatial_dims:
        time_series = data.mean(dim=spatial_dims)
    else:
        time_series = data  #has one dimension already 
    
    cmap= metadata['cmap']
    if isinstance(cmap, str):
        cmap=plt.get_cmap(cmap)

    plt.figure(figsize=(10,5))
    plt.plot(time_series['time'], time_series,)
    plt.title(f"Time Series of {metadata['label']}")
    plt.xlabel('Time')
    plt.ylabel(f"{metadata['label']} [{metadata['unit']}]")
    plt.show()


def plot_maps (data: xr.DataArray, metadata: dict, time):
    
    # Select the time slice for mapping (default first time step)
    if 'time' in data.dims:
        data_slice = data.sel(time=time)
    else:
        data_slice = data

    plt.figure(figsize=(8,6))
    cmap = metadata['cmap']
    # If cmap is a string (e.g. 'viridis'), convert to plt colormap
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
        
    im = plt.imshow(data_slice, cmap=cmap)
    plt.title(f"{metadata['label']}) at time {time}")
    cbar = plt.colorbar(im)
    cbar.set_label(metadata['unit'])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def extract_and_plot_layers(filepath: Path, variable: str, mode:str, time):
    """Extracts and plots surface, bottom, and average layers of the given variable."""
    
    with xr.open_dataset(filepath) as ds: 
        data_var, metadata = check_variable(ds, variable)

        if mode == 'timeseries':
            if 'time' not in ds[variable].dims:
                raise ValueError (f"Variable '{variable} does not have the time dimension")
        
        if variable == 'chl': ######

            """ this should be able to call for the function to 
            compute the integral, pass it into a temp_file, 
            then do the plots accordingly"""
            
            chl_integrated = compute_ch_integral(filepath)
            plot_timeseries(chl_integrated, metadata)
    
        else: 
            """for any variable other than chlorophyll"""
            if mode == "timeseries": 
                plot_timeseries(data_var, metadata)
            elif mode == "maps":
                plot_maps(data_var, metadata, time)
            else: 
                raise ValueError(f"Invalid mode")


def main ():

    args = parse_args()
    output_dir = args.output_dir
    variable = args.variable
    mode = args.mode
    time = args.time

    if not output_dir.is_dir():
        raise ValueError(f"The path '{output_dir}' does not exist or is not a directory.")

    data_file_list = list(output_dir.glob("*.nc"))
    if not data_file_list:
        raise FileNotFoundError(f"No NetCDF (.nc) files found in '{output_dir}'.")

    data_file = data_file_list[0]
    LOGGER.info(f"Selected file: {data_file.name}")

    extract_and_plot_layers(filepath=data_file, variable=variable, mode=mode, time=time)

    LOGGER.info(f"Plotting completed for variable '{variable}' in mode '{mode}'")

if __name__ == '__main__':
    main()
