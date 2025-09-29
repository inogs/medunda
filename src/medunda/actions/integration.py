import logging

import xarray as xr

from medunda.tools.layers import compute_layer_height

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "compute_integral"

def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="Compute the integrated depth value of all variables with a 'depth' dimension"
    )

def compute_integral (data: xr.Dataset) -> xr.Dataset:

    layer_height = compute_layer_height(data.depth.values)
    lh = xr.DataArray(layer_height, dims=['depth'])

    ds_integrated = xr.Dataset()
    
    for var_name in data.data_vars:
        var = data[var_name]

        if "depth" in var.dims:
            weighted_avg = lh * var
            ds_integrated[var_name] = weighted_avg.sum(dim='depth')

    return ds_integrated