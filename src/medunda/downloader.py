import copernicusmarine
from datetime import datetime
import argparse
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

from domains.domain import GSA9
from sources.cmems import search_for_product
from sources.cmems import VARIABLES
from tools.argparse_utils import date_from_str


def parse_args ():
    """
    parse command line arguments: 
    --variable: the variable to download:
    --frequency: choose the frequency of the download: monthly or annualy:
    --output-dir: directory to save the download file
    """
    parser = argparse.ArgumentParser(
        description="dowload monthly data for a chosen variable")

    parser.add_argument(   
        "--variable",  
        type=str,
        choices=VARIABLES,
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
    parser.add_argument(      #input the directory to save the file
        "--output-dir",
        type=Path,
        default=Path("."),
        help="directory where the downloaded file will be saved",
    )
    return parser.parse_args()


def download_data (variable: str, output_dir:Path, frequency:str, start:datetime, end:datetime):

    """ Download and organize data by year, month and day,  for the chosen variables. 
    Steps: 1) Search in the 'products' dictionary for the product_id related to the chosen variable
           2) Create the output directory if it does not exist
           3) Define the output file name based on the variable
           4) Call copernicusmarine.subset () using the **parameters"""

    if frequency not in ["daily", "monthly"]:
        raise ValueError(f"invalid frequency")

    #1. search for product
    selected_product = search_for_product(var_name=variable)
    
    print (f"trying to download the variable '{variable}' from the product '{selected_product}'")
           
    #2. output directory if not available
    output_dir.mkdir(exist_ok=True)
    
    #3. define the output file name (exp: monthly.uo_1999-2023.nc)
    final_output_dir = output_dir / variable / frequency
    final_output_dir.mkdir(exist_ok=True, parents=True)

    output_filename = f"'{frequency}''{variable}'_'{start}'-'{end}'.nc"
    output_filepath = final_output_dir / output_filename

    print (f"downloading '{frequency}''{variable}' from '{start}' to '{end}'")

    #4 
    copernicusmarine.subset(
        dataset_id=selected_product,
        variables=[variable],
        start_datetime=start,
        end_datetime=end,
        output_filename=output_filepath,
        **GSA9.model_dump()
    )

    return (output_filepath, )


def validate_dataset(filepath, variable):        
    """Validates the dataset, by checking for:
    dimensions, variables, and depth coverage."""
    
    print(f"dataset validated: {filepath}")

    with xr.open_dataset(filepath) as dataset:  #open the dataset using xarray
        # check for necessary dimensions
        required_dims=["time", "depth", "latitude", "longitude"]
        
        for dim in required_dims:
            if dim not in dataset.dims:
                print(f"{dim}: dimension missing, validation failed")
                return False
            
        if variable not in dataset.data_vars: 
            print(f"{variable}: variable missing. Validation failed")
            return False
        
        if "depth" in dataset.variables: 
            depth_values = dataset["depth"].values
            print(f"{depth_values}")
            if depth_values.min()<0 or depth_values.max()>300:
                print ("depth range is outside the expected bounds. Validation failed.")
                return False

    print ("Successful Validation")
    return True


def main ():
    args=parse_args()       #parse the command line arguments

    downloaded_files = download_data(args.variable, args.output_dir, args.frequency, args.start_date, args.end_date)

    for filepath in downloaded_files:
        if validate_dataset(filepath, args.variable): 
            print(f"dataset validated for variable: '{args.variable}'")
        else:
            print(f"failed dataset validation for variable:'{args.variable}'")


if __name__ == "__main__":
    main()
