import argparse
from datetime import datetime
import yaml
import copernicusmarine
import os
import xarray as xr

start=datetime(year=1999, month=1, day=1)
end=datetime(year=2022, month=12, day=31)

products = {       #Dictionary of available products 
       # -> each key represents a product and is associated with a list of available variables.

    "cmems_mod_med_phy-cur_my_4.2km_P1Y-m": ["uo", "vo"],
    "cmems_mod_med_phy-tem_my_4.2km_P1Y-m": ["thetao"],
    "cmems_mod_med_bgc-co2_my_4.2km_P1Y-m": ["pH", "o2"],
    "cmems_mod_med_bgc-bio_my_4.2km_P1Y-m": ["chl"],
}

parameters = {         #geographical coordinates and bathymetry 
    "minimum_latitude":41,
    "maximum_latitude":44,
    "minimum_longitude":9,
    "maximum_longitude":13,
    "minimum_depth":0,
    "maximum_depth":200,
}

def parse_args ():
    """
    parse command line arguments: 
    --variable: the variable to download:
    --output-dir: directory to save the download file
    """
    parser = argparse.ArgumentParser(
        description="dowload annual data for a chosen variable"  
    )
    parser.add_argument(
        "--variable",
        type=str,
        required=True,
        help="Name of the variable to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="directory where the downloaded file will be saved",
    )
    return parser.parse_args()

def download_data (variable, output_dir):

    """ Download annual data for the chosen variables. 
    Steps: 1) Search in the 'products' dictionnary for the product_id related to the chosen variable
           2) Create the output directory if it does not exist
           3) Define the output file name based on the variable
           4) Call copernicusmarine.subset () using the **parameters"""
    
    #1. search for product
    selected_product=None
    for prod_id, vars_available in products.items():
        selected_product=prod_id
        break
    if selected_product is None:
        raise ValueError (f"Variable '{variable} is not available in the dictionnary")

    #2. output directory if not available
    os.makedirs(output_dir, exist_ok=True)
    
    #3. define the output file name (vo_annual.nc)
    output_filename = os.path.join(output_dir,f"{variable}_annual.nc")

    print (f"dowloading data for variable '{variable}")

    #4. dowloading data
    dataset = copernicusmarine.subset(
        dataset_id=selected_product,
        variables=[variable],
        output_filename=output_filename,
        **parameters,
        start_datetime=start,
        end_datetime=end,
    )
    print (f"download complete: {output_filename}")
    return dataset

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

def argument():
    # TODO: Write a sensible description of what this script does
    parser = argparse.ArgumentParser(
        description="""
        Here there is a description of this program
        """
    )
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        required=True,
        help="""
        The domain of the basin for which the data will be downloaded
        """,
    )
    #
    # parser.add_argument(
    #     "--start-date",
    #     "-s",
    #     type=str,
    #     required=True,
    #     help="""
    #     The start date of the period for which the data will be downloaded
    #     """,
    # )
    #
    # parser.add_argument(
    #     "--end-date",
    #     "-e",
    #     type=str,
    #     required=True,
    #     help="""
    #     The end date of the period for which the data will be downloaded
    #     """,
    # )
    #
    # parser.add_argument(
    #     "--output-dir",
    #     "-o",
    #     type=str,
    #     required=True,
    #     help="""
    #     The path where the data will be downloaded
    #     """,
    # )
    #
    # parser.add_argument(
    #     "--variable",
    #     "-v",
    #     type=str,
    #     required=True,
    #     help="""
    #     Which variable will be downloaded
    #     """
    # )

    return parser.parse_args()

def main():
    args = argument()

    # Read the domain file

    with open(args.domain, "r") as f:
        domain = yaml.safe_load(f)

    copernicusmarine.subset(
        "med-cmcc-tem-rean-d",
        minimum_latitude=domain["domain"]["minimum_latitude"],
        maximum_latitude=domain["domain"]["maximum_latitude"],
        minimum_longitude=domain["domain"]["minimum_longitude"],
        maximum_longitude=domain["domain"]["maximum_longitude"],
        start_datetime=datetime(2020, 1, 1),
        end_datetime=datetime(2020, 1, 31),
        variables=["thetao",],
        output_filename="/dev/shm/test.nc",
    )

    print("Everything has been downloaded")



if __name__ == "__main__":
    main()
