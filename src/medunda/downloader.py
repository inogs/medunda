import argparse
import logging
import shutil
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import copernicusmarine
import xarray as xr

from medunda.sources.cmems import search_for_product
from medunda.sources.cmems import VARIABLES
from medunda.tools.argparse_utils import date_from_str
from medunda.tools.logging_utils import configure_logger
from medunda.tools.time_tables import split_by_month
from medunda.tools.time_tables import split_by_year
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
        variables: Sequence[str],
        output_dir:Path,
        frequency:str,
        start:datetime,
        end:datetime,
        domain: Domain,
        split_by: str = "whole",
        ) -> tuple[Path, ...]:

    """ Download and organize data by year, month, and day, for the chosen variables.
    Steps: 1) Search in the 'products' dictionary for the product_id related to the chosen variable
           2) Create the output directory if it does not exist
           3) Define the output file name based on the variable
           4) Call copernicusmarine.subset () using the **parameters"""

    allowed_frequency = ("daily", "monthly")
    if frequency not in allowed_frequency:
        raise ValueError(f"invalid frequency")

    allowed_split_by = ("month", "year", "whole")
    if split_by not in allowed_split_by:
        raise ValueError(
            f'Invalid value for "split_by"; received {split_by} but the only '
            f'valid values are {allowed_split_by}'
        )
    
    downloaded_files = []

    for variable in variables:
        #1. search for the product
        selected_product = search_for_product(var_name=variable, frequency=frequency)
        
        LOGGER.info(f"Downloading the variable '{variable}' from the product '{selected_product}'")

        #2. output directory if not available
        output_dir.mkdir(exist_ok=True)

        #3. define the output file name (exp: monthly.uo_1999-2023.nc)
        final_output_dir = output_dir / variable / frequency
        final_output_dir.mkdir(exist_ok=True, parents=True)

        if split_by == "whole":
            time_intervals = [(start, end)]
        elif split_by == "year":
            time_intervals = split_by_year(start, end)
        elif split_by == "month":
            time_intervals = split_by_month(start, end)
        else:
            raise ValueError(f"Internal error: invalid parameter: {split_by}")
        
        for start_date, end_date in time_intervals:

            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")
            file_time = f"{start_str}--{end_str}"

            output_filename = f"{domain.name}_{variable}_{frequency}_{file_time}.nc"
            temp_filename = output_filename + ".tmp"
            output_filepath = final_output_dir / output_filename
            temp_filepath = final_output_dir / temp_filename 
        
            LOGGER.info("Saving file %s", output_filepath)

            LOGGER.info(f"downloading '{frequency}''{variable}' from '{start}' to '{end}'")

            LOGGER.info(f"Dataset ID being used: {selected_product}")
            
            if temp_filepath.is_file():
                LOGGER.debug("%s already exists. I will delete it!", temp_filepath)
                temp_filepath.unlink()

            #4 
            copernicusmarine.subset(
                dataset_id=selected_product,
                variables=[variable],
                start_datetime=start_date,
                end_datetime=end_date,
                output_filename=str(temp_filepath),
                **domain.model_dump(exclude= {"name"})
            )

            shutil.move(temp_filepath, output_filepath)
            downloaded_files.append(output_filepath)

        return tuple(downloaded_files)


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

    for filepath in downloaded_files:
        if validate_dataset(filepath, args.variable): 
            LOGGER.info(f"dataset validated for variable: '{args.variable}'")
        else:
            LOGGER.warning(f"failed dataset validation for variable:'{args.variable}'")
    
    


if __name__ == "__main__":
    configure_logger(LOGGER)
    main()
