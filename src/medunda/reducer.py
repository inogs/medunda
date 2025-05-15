
import argparse
import logging
from pathlib import Path
import xarray as xr
import numpy as np


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
        mean_layer = ds.mean(dim="depth", skipna=True)
        LOGGER.info(f"writing file: {output_file}")
        mean_layer.to_netcdf(output_file)
    LOGGER.info("done")


def extract_layer (input_file, output_file):
    LOGGER.info(f"reading the file: {input_file}")
    with xr.open_dataset(input_file) as ds :
        #print(ds['depth'].values)
        bottom_layer = ds.isel(depth=-1)
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
    parser.add_argument(
        "--action",
        type=str,
        choices=("average", "extract_bottom", "extract_surface", "extract_layer"),
        required=True,
        help="End date of the download"
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
            extract_layer(input_file, output_file)
        case "extract_surface":
            extract_surface(input_file, output_file)
        case "extract_bottom":
            extract_bottom(input_file, output_file)
        case _:
            raise ValueError(
                f"Unexpected action: {action}"
            )



if __name__ == "__main__":
    configure_logger()
    main()
