import logging

import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "extract_bottom"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="Extract the values of the cells on the bottom"
    )


def extract_bottom(data: xr.Dataset) -> xr.Dataset :
    LOGGER.info(f"reading the file: {data}")

    variables = {}
    for var_name in data.data_vars:
        if var_name in ["depth", "latitude", "longitude", "time"]:
            continue
        LOGGER.debug("Computing variable %s", var_name)

        fixed_time_mask = np.ma.getmaskarray(
            data[var_name][0, :, :, :].to_masked_array()
        )
        index_map = np.count_nonzero(~fixed_time_mask, axis=0)
        index_map_labeled = xr.DataArray(
            dims=["latitude", "longitude"],
            data=index_map
        )

        time_values = data['time'].values

        new_data = data[var_name][:, index_map_labeled - 1]
        
        variables[var_name] = new_data

    final_dataset = xr.Dataset(variables)
    
    return final_dataset
