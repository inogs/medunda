import logging

import numpy as np
import xarray as xr
import scipy 

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "calculate_stats"

def configure_parser(subparsers):
    calculate_stats_parser = subparsers.add_parser(
        ACTION_NAME,
        help="Calculates the values of some specific statistical operations"
    )
    calculate_stats_parser.add_argument(
        "--operation",
        type=str,
        choices=["mean", "median", "variance", "quartiles", "all"],
        required=False,
        default = "all",
        help="Chose the operation(s) required"
    )

class Stats: 
    """This class provides methods to perform basic statistical calculations"""
    
    def __init__(self, data):
        self.data = data

    def mean (self):
        return np.mean(self.data, axis=0)
    def variance(self):
        return np.var(self.data, axis=0)
    def median(self):
        return np.median(self.data, axis=0) 
    # def quartiles(self): 
    #     q1 = np.percentile(self.data, 25, axis=0)
    #     q3 = np.percentile(self.data, 75, axis=0)
    #     return q1, q3

    
    def calculate (self, operation): 

        available_operations = {
            'mean': self.mean,
            'variance': self.variance,
            'median': self.median,
            #'quartiles': self.quartiles
        }

        if operation == "all":
            selected_op = available_operations.keys()
        else: 
            selected_op = [operation]

        results={}

        for operation in selected_op:
            if operation in available_operations:
                results[operation]=available_operations[operation]()
            else:
                raise ValueError(f"Unavailable operation: {operation}")
        
        return results

def calculate_stats (input_file, output_file, operation):
    """ Regroups and compute some statistical operations 
    according to the user's choice """
    
    LOGGER.info(f"reading file: {input_file}")
    with xr.open_dataset(input_file) as ds:

        var_name=list(ds.data_vars)[0]
        data=ds[var_name].values

    stats= Stats(data)
    results= stats.calculate(operation)

    ds_results = xr.Dataset() 
    for operation_name, result_array in results.items():
        ds_results[f"{var_name}_{operation_name}"]= xr.DataArray(
            data=result_array,
            dims=["depth", "latitude", "longitude"])
        
    ds_results.to_netcdf(output_file)

    return output_file 