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
        "--operations",
        type=str,
        nargs='+',
        choices=["mean", "median", "variance", "quartiles", "minimum", "maximum", "all"],
        required=False,
        default = "all",
        help="Choose the operation(s) required"
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
    def min (self):
        return np.min(self.data, axis=0)
    def max (self):
        return np.max(self.data, axis=0)
    
    def quartiles(self):
        percentiles = [5, 25, 75, 95]
        output = {
            str(k): np.percentile(self.data, k, axis=0) for k in percentiles
        }
        return output    

    
    def calculate (self, operations): 

        available_operations = {
            'mean': self.mean,
            'variance': self.variance,
            'median': self.median,
            'minimum': self.min, 
            'maximum': self.max,
            'quartiles': self.quartiles
        }
        
        if not operations:
            operations = ["all"]

        if "all" in operations:
            if len(operations) > 1: 
                raise ValueError (f"'all' cannot be used with other operations")
            selected_op = available_operations.keys() 

        else:
            if len(set(operations)) != len(operations):
                raise ValueError (f"Operations cannot be duplicated.")
            
            for op in operations:
                if op not in available_operations:
                    raise ValueError(f"Unavailable operation: {op}")
            selected_op = operations 

        results={}

        for operation in selected_op:
            results[operation]=available_operations[operation]()
            
        return results

def calculate_stats (input_file, output_file, operations):
    """ Regroups and compute some statistical operations 
    according to the user's choice """

    LOGGER.info(f"reading file: {input_file}")
    with xr.open_dataset(input_file) as ds:

        var_name=list(ds.data_vars)[0]
        data=ds[var_name].values

    stats= Stats(data)
    results= stats.calculate(operations)

    ds_results = xr.Dataset() 
    for operation_name, result_array in results.items():
        if isinstance(result_array, dict):
            for method_parameter, method_result in result_array.items():
                ds_results[f"{var_name}_{operation_name}_{method_parameter}"] = xr.DataArray(
                    data=method_result,
                    dims=["depth", "latitude", "longitude"]
                )
        else: 
            ds_results[f"{var_name}_{operation_name}"]= xr.DataArray(
                data=result_array,
                dims=["depth", "latitude", "longitude"])
        
    ds_results.to_netcdf(output_file)

    return output_file 