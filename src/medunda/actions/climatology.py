import logging
from pathlib import Path

import xarray as xr

from medunda.tools.argparse_utils import date_from_str


LOGGER = logging.getLogger(__name__)
ACTION_NAME = "climatology"


def configure_parser(subparsers):
    compute_climatology = subparsers.add_parser(
        ACTION_NAME,
        help="Compute the climatological average for a given variable over a specified time range."
    )
    compute_climatology.add_argument(
        "--variable",
        type=str,
        required=True,
        help="Choose the variable to compute the climatology on"
    )
    compute_climatology.add_argument(
        "--frequency",
        type=str,
        required=True,
        choices=["daily", "monthly","seasonally"],
        help="Choose the frequecy at which to compute the climatology"
    )
    compute_climatology.add_argument(
        "--start-date",
        type=date_from_str,
        required=False,
        help="The start date of the period. Defaults to the first year of the dataset."
    )
    compute_climatology.add_argument(
        "--end-date",
        type=date_from_str,
        required=False,
        help="The end date of climatology period. Defaults to the last year of the dataset."
    )

def configure_data_frequency (data=xr.Dataset):
    ds_title = data.attrs.get('title').lower()
    if "monthly" in ds_title:
        ds_frequency = "monthly"
    elif "daily" in ds_title:
        ds_frequency = "daily"
    else: 
        raise ValueError("Frequency cannot be determined from the title of the dataset. " \
        "Monthly or Daily datasets are required.")
    return ds_frequency

def weighted_monthly_average(group, days_in_month):
    """This function calculates the monthly weighted average
    taking in account the diffrent months lengths"""

    group_time_indices = group.time.values
    group_days_weights = days_in_month.sel(time=group_time_indices)

    total_weight = group_days_weights.sum(dim='time', skipna=True)
    weighted_sum = (group*group_days_weights).sum(dim='time', skipna=True)
    
    weighted_monthly_average = weighted_sum/total_weight
    
    return weighted_monthly_average

def climatology(data: xr.Dataset,
                output_file: Path, 
                variable:str, 
                frequency:str,
                start_date=None, 
                end_date=None,):
    
    #check the variable
    if variable not in data.data_vars:
        available_variables= list(data.data_vars.keys())
        raise ValueError(f"Variable {variable} is not found in the dataset."
                         f"Choose from the available variables: {available_variables}")

    data_var = data[variable]

    #define the period of climatology in case start_date and end_date are not specified
    if start_date is None:
        start_date = data["time"].values.min()
    if end_date is None:
        end_date = data["time"].values.max()

    data_var = data_var.sel(time=slice(start_date, end_date))
    
    ds_frequency = configure_data_frequency(data)
    
    LOGGER.info(f"Computing {type} climatology for variable: {variable}")

    ###computing climatology with monthly datasets
    if ds_frequency == "monthly":

        if frequency == "monthly":          #1
            data_var.coords["month"] = data_var["time"].dt.month
            
            months = data_var["time"].dt.month
            days_in_month = data_var["time"].dt.days_in_month

            climatology_data = data_var.groupby("month").apply(
                weighted_monthly_average,
                days_in_month=days_in_month
                )
            
        elif frequency == "daily":          #3
            raise ValueError("Impossible to compute daily climatology with monthly dataset." \
                            "A dataset with daily resolution is required.")

        elif frequency == "seasonally":     #5
            pass
    
    ###computing the climatology with daily datasets
    elif ds_frequency == "daily":
        if type == "monthly":          #2
            pass
        elif type == "daily":          #4
            pass
        elif type == "seasonally":     #6
            pass
    
    climatology = xr.Dataset({
                variable: climatology_data
            })
    
    climatology.to_netcdf(output_file)
    return climatology

    