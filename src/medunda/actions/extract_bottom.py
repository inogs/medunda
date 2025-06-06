import logging
from pathlib import Path

import numpy as np
import xarray as xr


LOGGER = logging.getLogger(__name__)
ACTION_NAME = "extract_bottom"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="Extract the values of the cells on the bottom"
    )


def extract_bottom(input_file: Path, output_file: Path):
    LOGGER.info(f"reading the file: {input_file}")
    with xr.open_dataset(input_file) as ds:
        var_name = list(ds.data_vars)[0]
        mask = ds[var_name].to_masked_array(copy=False).mask
        fixed_time_mask = mask[0, :, :, :]
        index_map = np.count_nonzero(~fixed_time_mask, axis=0)

        current_data = ds[var_name]

        new_shape = current_data.shape[:1] + current_data.shape[2:]

        new_data_array = np.empty(shape=new_shape, dtype=current_data.dtype)

        for i in range(current_data.shape[2]):
            for j in range(current_data.shape[3]):
                blue_cells = index_map[i, j]

                current_value = current_data[:, blue_cells - 1, i, j]
                new_data_array[:, i, j] = current_value

        new_data = xr.DataArray(dims=["time", "latitude", "longitude"],
                                data=new_data_array)
        ds[var_name] = new_data

        ds.to_netcdf(output_file)
