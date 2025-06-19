import logging

import numpy as np
import pandas as pd
import xarray as xr


LOGGER = logging.getLogger(__name__)
ACTION_NAME = "extract_layer_extremes"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="extract the minimum and maximum value of a variable for each layer available in the dataset"
    )

def extract_layer_extremes (input_file, output_file):
    """Extracts the maximum and the minimum values of a variable for each year"""

    LOGGER.info(f"reading file: {input_file}")
    with xr.open_dataset(input_file) as ds:

        var_name=list(ds.data_vars)[0]
        var=ds[var_name]

        values_per_depth= []

        depths = var.depth.values
        for depth in depths: 
            values_at_depth = var.sel (depth=depth)
            min_value = float(values_at_depth.min().values)
            max_value = float(values_at_depth.max().values)

            values_per_depth.append({
                "depth": float(depth),
                "minimum value": min_value,
                "maximum value": max_value,
            })
        
        df = pd.DataFrame(values_per_depth)
        ds_xr = df.set_index("depth").to_xarray()
        ds_xr.to_netcdf(output_file) 