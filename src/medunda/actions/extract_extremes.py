import logging

import numpy as np
import pandas as pd
import xarray as xr


LOGGER = logging.getLogger(__name__)
ACTION_NAME = "extract_min_max"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="extract the minimum and maximum value of a variable of each year"
    )


def extract_min_max (input_file, output_file):
    """Extracts the maximum and the minimum values of a variable for each year"""

    LOGGER.info(f"reading file: {input_file}")
    with xr.open_dataset(input_file) as ds:

        var_name=list(ds.data_vars)[0]
        var=ds[var_name]
        
        values =[]
        years = pd.to_datetime(ds.time.values).year
        years_array = np.array(years)

        for year in sorted(list(set(years_array))) :
            indices = np.where(years_array == year)[0]
            yearly_data = var.isel(time=indices)
            
            min_value = float(yearly_data.min().item())
            max_value = float(yearly_data.max().item())

            min_indices = np.unravel_index(
                yearly_data.argmin().item(), yearly_data.shape
            )
            max_indices = np.unravel_index(
                yearly_data.argmax().item(), yearly_data.shape
            )
            depth_at_min = float(yearly_data.depth[min_indices[1]])
            depth_at_max = float(yearly_data.depth[max_indices[1]])

            values.append ({
                "year": year, 
                "minimum value": min_value,
                "depth at min": depth_at_min,
                "maximum value": max_value,
                "depth at max": depth_at_max,
                })
            
        df=pd.DataFrame(values)
        df.to_csv(output_file)
