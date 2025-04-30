import copernicusmarine
from datetime import datetime
import argparse
import os
import xarray as xr


start=datetime (year= 1999, month=1, day=1)
end=datetime (year= 2023, month=12, day=31)

products = {     
       # -> each key represents a product and its associated with a list of available variables.
     # physical variables (MEDSEA_MULTIYEAR_PHY_006_004)
    "med-cmcc-tem-rean-m": ["thetao"],  # temperature
    "med-cmcc-cur-rean-m": ["uo"],    # current: composante zonale
    "med-cmcc-cur-rean-m": [ "vo" ],    # current: composante méridienne
    "med-cmcc-sal-rean-m" : ["so"],  # salinity

    # biogeochemical variables (MEDSEA_MULTIYEAR_BGC_006_008)
    "med-ogs-bio-rean-m": ["o2"],            # dissolved oxygen
    "med-ogs-car-rean-m":  ["ph"],          # pH
    "med-ogs-nut-rean-m" : ["no3"],          # nitrate
    "med-ogs-nut-rean-m" : ["po4"],          # phosphate
    "med-ogs-nut-rean-m": ["si"],      # silicate
    "med-ogs-pft-rean-m"  : ["chl"] ,         # chlorophylle a
}

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
        description="dowload monthly data for a chosen variable"  #change to monthly data!!
    )
    parser.add_argument(     #input variable name
        "--variable",
        type=str,
        required=True,
        help="Name of the variable to download"
    )

        #parser.add_argument(    #input the frequency: download per month or day
        #"--frequency",
        #type=str,
        #choices=["monthly", "daily"],
        #required=True,
        #help="frequency of the download")

    parser.add_argument(      #input the directory to save the file
        "--output-dir",
        type=str,
        default=".",
        help="directory where the downloaded file will be saved",
    )
    return parser.parse_args()

def download_data (variable, output_dir):

    """ Download and organize data by year, month and day,  for the chosen variables. 
    Steps: 1) Search in the 'products' dictionnary for the product_id related to the chosen variable
           2) Create the output directory if it does not exist
           3) Define the output file name based on the variable
           4) Call copernicusmarine.subset () using the **parameters"""
    
    #1. search for product
    selected_product=None
    for prod_id, vars_available in products.items():             #zip: to loop between more keys
        if variable in vars_available:
            selected_product=prod_id
            break
    if selected_product is None:
        raise ValueError (f"Variable '{variable}' is not available in the dictionnary")
    
    print (f"trying to download the variable '{variable}' from the product '{selected_product}'")
           
    #2. output directory if not available
    os.makedirs(output_dir, exist_ok=True)
    
    #3. define the output file name (exp: vo_annual.nc)
    output_filename = os.path.join(output_dir,f"{variable}_1999-2023.nc")

    print (f"dowloading '{variable} from 1999 to 2023")

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
    download_data(args.variable, args.output_dir)     

    filepath = os.path.join(args.output_dir, f"{args.variable}_1999-2023.nc")

    if validate_dataset(filepath, args.variable): 
        print(f"dataset validated for variable: '{args.variable}'")
    else:
        print(f"failed dataset validation for variable:'{args.variable}'")

if __name__ == "__main__":
    main()
