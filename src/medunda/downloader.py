import copernicusmarine
from datetime import datetime
import argparse
import os
import xarray as xr

from sources.cmems import PRODUCTS
from sources.cmems import VARIABLES
from tools.argparse_utils import date_from_str


start=datetime (year= 1999, month=1, day=1)
end=datetime (year= 2023, month=12, day=31)


parameters = {      
     #geographical coordinates and bathymetry 
    "minimum_latitude":41,
    "maximum_latitude":44,
    "minimum_longitude":9,
    "maximum_longitude":13,
    "minimum_depth":0,
    "maximum_depth":250,
}


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
        type=str,
        default=".",
        help="directory where the downloaded file will be saved",
    )
    return parser.parse_args()


def download_data (variable, output_dir, frequency, start, end):

    """ Download and organize data by year, month and day,  for the chosen variables. 
    Steps: 1) Search in the 'products' dictionary for the product_id related to the chosen variable
           2) Create the output directory if it does not exist
           3) Define the output file name based on the variable
           4) Call copernicusmarine.subset () using the **parameters"""
    
    #1. search for product
    selected_product=None
    for prod_id, vars_available in PRODUCTS.items():             #zip: to loop between more keys
        if variable in vars_available:
            selected_product=prod_id
            break
    if selected_product is None:
        raise ValueError (f"Variable '{variable}' is not available in the dictionnary")
    
    print (f"trying to download the variable '{variable}' from the product '{selected_product}'")
           
    #2. output directory if not available
    os.makedirs(output_dir, exist_ok=True)
    
    #3. define the output file name (exp: monthly.uo_1999-2023.nc)
    output_dir=os.path.join(output_dir, variable, frequency)
    output_filename = os.path.join(output_dir,f"'{frequency}''{variable}'_'{start}'-'{end}'.nc")


    if frequency in ["daily", "monthly"]:
        start_datetime= date_from_str(start)
        end_datetime= date_from_str(end)
    else:
        raise ValueError(f"invalid frequency")

    print (f"downloading '{frequency}''{variable}' from '{start}' to '{end}'")

    #4 
    try:
        copernicusmarine.subset(
            dataset_id=selected_product,
            variables=[variable],
            start_datetime=start,
            end_datetime=end,
            output_filename=output_filename,
            **parameters
        )
    except Exception as e:
        print(f"download failed: {e}")

def validate_dataset(filepath, variable):        
    """Validates the dataset, by checking for:
    dimensions, variables, and depth coverage."""
    
    print(f"dataset validated: {filepath}")

    dataset= xr.open_dataset(filepath)   #open the dataset using xarray

        #check for necessary dimensions
    required_dims=["time", "depth", "latitude", "longitude"]
    
    for dim in required_dims:
        if dim not in dataset.dims:
            print(f"{dim}: dimension missing, validation failed")
            dataset.close()
            return False
        
    if variable not in dataset.data_vars: 
        print(f"{variable}: variable missing. Validation failed")
        dataset.close ()
        return False
    
    if "depth" in dataset.variables: 
        depth_values = dataset["depth"].values
        print(f"{depth_values}")
        if depth_values.min()<0 or depth_values.max()>300:
            print ("depth range is outside the expected bounds. Validation failed.")
            return False
    
    print ("Successful Validation")
    dataset.close()
    return True

def main ():
    args=parse_args()       #parse the command line arguments
    download_data(args.variable, args.output_dir, args.frequency, args.start_date, args.end_date)     

    filepath = os.path.join(args.output_dir, f"{args.variable}'{args.frequency}'_'{args.start}'-'{args.end}'.nc")

    if validate_dataset(filepath, args.variable): 
        print(f"dataset validated for variable: '{args.variable}'")
    else:
        print(f"failed dataset validation for variable:'{args.variable}'")

if __name__ == "__main__":
    main()
