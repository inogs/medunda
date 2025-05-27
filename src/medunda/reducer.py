
import argparse
import logging
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd


LOGGER = logging.getLogger()


def configure_logger():
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    LOGGER.setLevel(logging.DEBUG)

    logging.getLogger("botocore").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("h5py").setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    LOGGER.addHandler(handler)


def compute_average(input_file, output_file): 
    LOGGER.info(f"reading file:{input_file}")
    with xr.open_dataset(input_file) as ds :
        LOGGER.debug("computing_layer_height")
        layer_height= compute_layer_height(ds.depth.values)
        layer_height_extended = xr.DataArray(layer_height, dims=["depth"])
        
        var_name=list(ds.data_vars)[0]
        mask = ds[var_name].to_masked_array(copy=False).mask[0,:,:,:]
        mask_extended = xr.DataArray(
            mask,
            dims=("depth", "latitude", "longitude")
        )
        total_height = (layer_height_extended * ~mask_extended).sum(dim="depth")

        mean_layer = (ds*layer_height_extended).sum(dim="depth", skipna=True) / total_height

        LOGGER.info(f"writing file: {output_file}")
        mean_layer.to_netcdf(output_file)
    LOGGER.info("done")


def extract_layer (input_file, output_file, depth):
    LOGGER.info(f"reading the file: {input_file}")
    with xr.open_dataset(input_file) as ds :
        #print(ds['depth'].values)
        bottom_layer = ds.sel(depth=depth, method="nearest")
        LOGGER.info(f"writing the file: {output_file}")
        bottom_layer.to_netcdf(output_file)
    LOGGER.info("done")
    print(bottom_layer.coords)


def extract_bottom (input_file, output_file):
    LOGGER.info(f"reading the file: {input_file}")
    with xr.open_dataset(input_file) as ds :
        var_name=list(ds.data_vars)[0]
        mask=ds[var_name].to_masked_array(copy=False).mask
        fixed_time_mask=mask[0,:,:,:]
        index_map=np.count_nonzero(~fixed_time_mask, axis=0)
        
        current_data = ds[var_name]

        new_shape = current_data.shape[:1] + current_data.shape[2:]

        new_data_array = np.empty(shape=new_shape, dtype=current_data.dtype)

        for i in range (current_data.shape[2]):
            for j in range (current_data.shape[3]):
                blue_cells = index_map [i,j]
            

                current_value = current_data[:, blue_cells-1, i , j ]
                new_data_array[:, i, j] = current_value


        new_data = xr.DataArray(dims= ["time", "latitude", "longitude"], data=new_data_array) 
        ds[var_name] = new_data
        
        ds.to_netcdf(output_file)


def extract_surface (input_file, output_file):
    LOGGER.info(f"reading the file: {input_file}")
    with xr.open_dataset(input_file) as ds:
        surface_layer = ds.isel(depth=0)
        LOGGER.info(f"writing the file: {output_file}")
        surface_layer.to_netcdf(output_file)
        print(surface_layer)


def averaging_between_layers (input_file, output_file, depth_min, depth_max):
    with xr.open_dataset(input_file) as ds:

        var_name = list(ds.data_vars)[0]
        #var = ds[var_name]

        selected_layer = ds[var_name].sel(depth=slice(depth_min, depth_max))
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


def compute_layer_height (layer_centers): 
    layer_height=[] 
    for i in range (len(layer_centers)):
    
        if i==0: 
            layer_height.append(layer_centers[0]*2) 
        else: 
            current_layer=(layer_centers[i]-sum(layer_height[:i]))*2
            layer_height.append(current_layer)
    return np.array(layer_height) 


def extract_min_max (input_file, output_file):
    """Extracts the maximum and the minimum values of a variable for each year"""

    LOGGER.info(f"reading file: {input_file}")
    with xr.open_dataset(input_file) as ds:
        var_name=list(ds.data_vars)[0] 
        var=ds[var_name]

        values = []
        years = pd.to_datetime(ds.time.values).year 
        print(pd.to_datetime(ds.time.values))
        print(years)
        years_array = np.array(years)
        for year in sorted(list(set(years_array))) : 
            indices = np.where(years_array == year)[0]
            yearly_data = var.isel(time=indices)
            min_value = float(yearly_data.min().values)
            max_value = float(yearly_data.max().values)
            values.append ({
                "year": year,
                "minimum value": min_value,
                "maximum value": max_value,
            })
        df=pd.DataFrame(values)
        df.to_csv(output_file)


def parse_args ():
    """
    parse command line arguments: 
    """
    parser = argparse.ArgumentParser(
        description="elaborate downloaded data by performing different statistical operations"
        )

    parser.add_argument(   
        "--input-file",  
        type=Path,
        required=True,
        help="Path of the input file"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path of the output file"
    )
    subparsers = parser.add_subparsers(
        title="action",
        dest="action",
        help="Sets which operation must be executed on the input file"
    )
    subparsers.add_parser('average', help="Compute the average on all the vertical levels")
    subparsers.add_parser('extract_bottom', help="Extract the values of the cells on the bottom")
    subparsers.add_parser('extract_surface', help="Extract the values of the cells on the surface")
    subparsers.add_parser('extract_min_max', help="extract the minimum and maximum value of a variable of each year")
    extract_layer_parser = subparsers.add_parser(
        "extract_layer",
        help="Extract the values of a specific depth (in metres)"
    )
    extract_layer_parser.add_argument(
        "--depth",
        type=float,
        required=True,
        help="Depth of the layer that must be extracted"
    )

    averaging_between_layers_parser = subparsers.add_parser(
        "averaging_between_layers",
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
    
    return parser.parse_args()


def main ():
    args=parse_args()       #parse the command line arguments

    input_file = args.input_file
    output_file = args.output_file
    action = args.action

    match action:
        case "average":
            compute_average(input_file, output_file)
        case "extract_layer":
            depth = args.depth
            extract_layer(input_file, output_file, depth=depth)
        case "extract_surface":
            extract_surface(input_file, output_file)
        case "extract_bottom":
            extract_bottom(input_file, output_file)
        case "averaging_between_layers":
            depth_min= args.depth_min
            depth_max= args.depth_max
            averaging_between_layers(input_file, output_file, depth_min, depth_max)
        case "extract_min_max":
            extract_min_max(input_file, output_file)
        case _:
            raise ValueError(
                f"Unexpected action: {action}"
            )



if __name__ == "__main__":
    configure_logger()
    main()
