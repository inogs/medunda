import logging

import xarray as xr

from medunda.tools.layers import compute_layer_height


LOGGER = logging.getLogger(__name__)
ACTION_NAME = "averaging_between_layers"


def configure_parser(subparsers):
    averaging_between_layers_parser = subparsers.add_parser(
        ACTION_NAME,
        help="Compute the vertical average between two specific depths"
    )
    averaging_between_layers_parser.add_argument(
        "--depth-min",
        type=float,
        required=True,
        help="minimum limit of the layer"
    )
    averaging_between_layers_parser.add_argument(
        "--depth-max",
        type=float,
        required=True,
        help="maximum limit of the layer"
    )


def averaging_between_layers (data: xr.Dataset, output_file, depth_min, depth_max):
    var_name = list(data.data_vars)[0]
    #var = ds[var_name]

    selected_layer = data[var_name].sel(depth=slice(depth_min, depth_max))
    selected_depth = selected_layer.depth.values

    layer_height = compute_layer_height (selected_depth)
    layer_height_extended = xr.DataArray (layer_height, dims=["depth"])

    mask =selected_layer.to_masked_array(copy=False).mask[0,:,:,:]
    mask_extended = xr.DataArray(
        mask,
        dims=("depth", "latitude", "longitude")
    )

    total_height = (layer_height_extended * ~mask_extended).sum(dim="depth")

    weighted_average = (selected_layer*layer_height_extended).sum(dim="depth", skipna=True) / total_height

    #output_filename = f"{var_name}_vertical_average_{depth_min}_{depth_max}.nc"
    #output_file = output_filename

    weighted_average.to_netcdf(output_file)
