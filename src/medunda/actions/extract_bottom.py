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

        mask = np.ma.getmaskarray(data[var_name].to_masked_array(copy=False))
        fixed_time_mask = mask[0, :, :, :]
        index_map = np.count_nonzero(~fixed_time_mask, axis=0)

        current_data = data[var_name]

        new_shape = current_data.shape[:1] + current_data.shape[2:]

        new_data_array = np.empty(shape=new_shape, dtype=current_data.dtype)

        for i in range(current_data.shape[2]):
            for j in range(current_data.shape[3]):
                blue_cells = index_map[i, j]

                current_value = current_data[:, blue_cells - 1, i, j]
                new_data_array[:, i, j] = current_value

        time_values = data['time'].values

        new_data = xr.DataArray(dims=["time", "latitude", "longitude"],
                                coords={"time": time_values, 
                                        "latitude": data["latitude"].values,
                                        "longitude": data["longitude"].values,
                                        },
                                data=new_data_array)
        variables[var_name] = new_data

    final_dataset = xr.Dataset(variables)
    
    return final_dataset
