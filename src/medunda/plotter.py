import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import cmocean
import numpy as np
import xarray as xr

from medunda.plots import maps
from medunda.plots import timeseries
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

DEFAULT_VAR = {
    "unit": "",
    "cmap": "viridis",
}


PLOTS=[
    timeseries,
    maps,
]


def parse_args ():
    """
    parse command line arguments: 
    --input-file: path of the input file
    --variable: name of the variable to plot
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
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where the downloaded files are saved",
    )

    subparsers = parser.add_subparsers(
        title="mode",
        required=True,
        dest="mode",
        help="Sets which plot must be executed on the input file"
    )

    for mode in PLOTS:
        LOGGER.debug("Letting module %s configure its parser", mode.__name__)
        mode.configure_parser(subparsers)

    return parser.parse_args()


def check_variable (ds, var):

    if var not in ds:
        raise ValueError(f"Variable '{var}' not found in dataset")

    if var in VAR_METADATA:
        var_metadata = VAR_METADATA[var]
    else:
        var_metadata = DEFAULT_VAR
        var_metadata["label"] = var

    return ds[var], var_metadata


def plotter (filepath: Path, variable: str, mode:str, args):
    """Extracts and plots surface, bottom, and average layers of the given variable."""
    
    with xr.open_dataset(filepath) as ds: 
        data_var, metadata = check_variable(ds, variable)

        if mode == 'plotting_timeseries':
            if 'time' not in ds[variable].dims:
                raise ValueError (f"Variable '{variable} does not have the time dimension")
            else:  
                timeseries.plotting_timeseries(
                    data=data_var, 
                    metadata=metadata,
                    start_time=args.start_time,
                    end_time=args.end_time
                    )
        
        elif mode == "plotting_maps":
            maps.plotting_maps(
                data=data_var, 
                metadata=metadata, 
                time=args.time)
        
        else: 
            raise ValueError(f"Invalid mode")


def main ():

    args = parse_args()
    output_dir = args.output_dir
    variable = args.variable
    mode = args.mode
    data_file = args.input_file 

    if not data_file.exists():
        raise FileNotFoundError (f"The file '{data_file}' does not exist.")

    if data_file.suffix != ".nc":
        raise ValueError(
            f'File {data_file} is not a valid netcdf file; its suffix does '
            'not end with ".nc."'
        )

    LOGGER.info(f"Selected file: {data_file.name}")

    plotter (filepath=data_file, variable=variable, mode=mode, args=args)

    LOGGER.info(f"Plotting completed for variable '{variable}' in mode '{mode}'")

if __name__ == '__main__':
    main()
