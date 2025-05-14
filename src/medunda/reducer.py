
import argparse
import logging
from pathlib import Path
import xarray as xr

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


def extract_bottom (input_file, output_file):
    LOGGER.info(f"reading the file: {input_file}")
    with xr.open_dataset(input_file) as ds :
        #print(ds['depth'].values)
        bottom_layer = ds.isel(depth=-1)
        LOGGER.info(f"writing the file: {output_file}")
        bottom_layer.to_netcdf(output_file)
    LOGGER.info("done")
    print(bottom_layer.coords)

 
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
        choices=("average", "extract_bottom", "extract_surface"),
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
        case "extract_bottom":
            extract_bottom(input_file, output_file)
        case "extract_surface":
            extract_surface(input_file, output_file)
        case _:
            raise ValueError(
                f"Unexpected action: {action}"
            )



if __name__ == "__main__":
    configure_logger()
    main()
