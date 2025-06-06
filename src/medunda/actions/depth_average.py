import logging

import xarray as xr

from medunda.tools.layers import compute_layer_height


LOGGER = logging.getLogger(__name__)
ACTION_NAME = "compute_depth_average"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="Compute the average on all the vertical levels"
    )


def compute_depth_average(input_file, output_file):
    LOGGER.info(f"reading file:{input_file}")
    with xr.open_dataset(input_file) as ds:
        LOGGER.debug("computing_layer_height")
        layer_height = compute_layer_height(ds.depth.values)
        layer_height_extended = xr.DataArray(layer_height, dims=["depth"])

        var_name = list(ds.data_vars)[0]
        mask = ds[var_name].to_masked_array(copy=False).mask[0, :, :, :]
        mask_extended = xr.DataArray(
            mask,
            dims=("depth", "latitude", "longitude")
        )
        total_height = (layer_height_extended * ~mask_extended).sum(dim="depth")

        mean_layer = (ds * layer_height_extended).sum(dim="depth",
                                                      skipna=True) / total_height

        LOGGER.info(f"writing file: {output_file}")
        mean_layer.to_netcdf(output_file)
    LOGGER.info("done")
