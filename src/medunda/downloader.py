import argparse
import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import xarray as xr

from medunda.dataset import Dataset
from medunda.sources.cmems import VARIABLES
from medunda.tools.argparse_utils import date_from_str
from medunda.tools.file_names import get_output_filename
from medunda.tools.logging_utils import configure_logger
from medunda.tools.time_tables import split_by_month
from medunda.tools.time_tables import split_by_year
from medunda.tools.typing import VarName
from medunda.domains.domain import read_domain
from medunda.domains.domain import Domain


LOGGER = logging.getLogger()


def parse_args ():
    """
    parse command line arguments: 
    --variable: the variable to download:
    --frequency: choose the frequency of the download: monthly or daily:
    --output-dir: directory to save the download file
    """
    parser = argparse.ArgumentParser(
        description="dowload monthly or daily data for a chosen variable")

    parser.add_argument(   
        "--variables",  
        type=str,
        choices=VARIABLES,
        nargs="+",
        required=True,
        help="Name of the variable to download"
    )
    parser.add_argument(
        "--start-date",
        type=date_from_str,
        required=True,
        help="Starting date for the download (format YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=date_from_str,
        required=True,
        help="End date of the download"
    )
    parser.add_argument( 
        "--frequency",
        type=str,
        choices=["monthly", "daily"],
        required=False,
        default="monthly",
        help="frequency of the downloaded data"
    )
    parser.add_argument(
        "--domain",
        type=Path,
        required=True,
        help="Choose the domain",
    )

    parser.add_argument(
        "--split-by",
        type=str,
        required=False,
        choices=["month", "year", "whole"],
        default="whole",
        help="Choose the domain",
    )

    parser.add_argument(      #input the directory to save the file
        "--output-dir",
        type=Path,
        required=True,
        help="directory where the downloaded file will be saved",
    )
    return parser.parse_args()


def download_data (
        variables: Iterable[VarName],
        output_dir:Path,
        frequency:str,
        start:datetime,
        end:datetime,
        domain: Domain,
        split_by: str = "whole",
        ) -> dict[VarName, tuple[Path, ...]]:

    """Download data for the specified variables, frequency, and time range.
    """
    # Check if frequency is valid
    allowed_frequency = ("daily", "monthly")
    if frequency not in allowed_frequency:
        raise ValueError(f"invalid frequency")

    # Check if value of split_by is valid
    allowed_split_by = ("month", "year", "whole")
    if split_by not in allowed_split_by:
        raise ValueError(
            f'Invalid value for "split_by"; received {split_by} but the only '
            f'valid values are {allowed_split_by}'
        )
    
    # Prepare the time intervals based on the split_by parameter
    if split_by == "whole":
        time_intervals = [(start, end)]
    elif split_by == "year":
        time_intervals = split_by_year(start, end)
    elif split_by == "month":
        time_intervals = split_by_month(start, end)
    else:
        raise ValueError(f"Internal error: invalid parameter: {split_by}")
    
    # Prepare output directory if not available
    output_dir.mkdir(exist_ok=True)

    if not output_dir.is_dir():
        raise ValueError(f"Output directory {output_dir} is not a directory")

    # We want to prepare a Dataset object that will contain the path of all the
    # files that we will download.
    downloaded_files: dict[VarName, tuple[Path, ...]] = {}

    # Dataset file
    dataset_file = output_dir / f"medunda_dataset.json"
    if dataset_file.exists():
        raise IOError(
            f"Dataset file {dataset_file} already exists. "
            f"This means that the output directory {output_dir} has been "
            "already used for a previous download. Please choose a different "
            "output directory or delete the existing dataset file."
        )

    for variable in variables:
        # We save here the files that we download for this variable
        files_for_current_var: list[Path] = []

        var_output_dir = output_dir / variable / frequency

        for start_date, end_date in time_intervals:
            output_file_name = get_output_filename(
                variable=variable,
                frequency=frequency,
                start=start_date,
                end=end_date,
                domain_name=domain.name
            )

            output_file_path = var_output_dir / output_file_name
            files_for_current_var.append(output_file_path)

        downloaded_files[variable] = tuple(files_for_current_var)
    
    # Create a Dataset object that will describe the data that we are going
    # to download.
    dataset = Dataset(
        domain=domain,
        start_date=start,
        end_date=end,
        data_files=downloaded_files,
    )

    # Save the dataset information to a JSON file
    dataset_file.write_text(dataset.model_dump_json(indent=4) + "\n")

    # Download the data for each variable. We delegate the actual download
    # to the Dataset class, which will handle the downloading of the data
    # files for each variable.
    LOGGER.info("Downloading data...")
    dataset.download_data()

    return dataset.data_files


def validate_dataset(filepath, variable):        
    """Validates the dataset, by checking for:
    dimensions, variables, and depth coverage."""
    
    LOGGER.info(f"dataset validated: {filepath}")

    with xr.open_dataset(filepath) as dataset:  #open the dataset using xarray
        # check for necessary dimensions
        required_dims=["time", "depth", "latitude", "longitude"]
        
        for dim in required_dims:
            if dim not in dataset.dims:
                LOGGER.error(f"{dim}: dimension missing, validation failed")
                return False
            
        if variable not in dataset.data_vars: 
            LOGGER.error(f"{variable}: variable missing. Validation failed")
            return False
        
        if "depth" in dataset.variables: 
            depth_values = dataset["depth"].values
            LOGGER.debug(f"depth values: {depth_values}")
            if depth_values.min()<0 or depth_values.max()>800:
                print ("depth range is outside the expected bounds. Validation failed.")
                return False

    LOGGER.info ("Successful Validation")
    return True


def main ():
    configure_logger(LOGGER)

    args=parse_args()       #parse the command line arguments
    
    domain= read_domain (args.domain)
    
    downloaded_files = download_data(
        variables=args.variables,
        output_dir=args.output_dir,
        frequency=args.frequency,
        start=args.start_date,
        end=args.end_date,
        domain=domain,
        split_by=args.split_by
    )

    for variable, files_for_var in downloaded_files.items():
        for filepath in files_for_var:
            if validate_dataset(filepath, variable): 
                LOGGER.info(f"dataset validated for variable: '{variable}'")
            else:
                LOGGER.warning(f"failed dataset validation for variable:'{variable}'")
    
    


if __name__ == "__main__":
    main()
