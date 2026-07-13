import logging

from medunda.tools.argparse_utils import date_from_str
from medunda.tools.lazy_imports import xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "climatology"


def configure_parser(subparsers):
    compute_climatology = subparsers.add_parser(
        ACTION_NAME,
        help="Compute the climatological average for a given variable over a specified time range.",
    )
    compute_climatology.add_argument(
        "--variable",
        type=str,
        required=True,
        help="Choose the variable to compute the climatology on",
    )
    compute_climatology.add_argument(
        "--frequency",
        type=str,
        required=True,
        choices=["daily", "monthly", "seasonally"],
        help="Choose the frequecy at which to compute the climatology",
    )
    compute_climatology.add_argument(
        "--start-date",
        type=date_from_str,
        required=False,
        help="The start date of the period. Defaults to the first year of the dataset.",
    )
    compute_climatology.add_argument(
        "--end-date",
        type=date_from_str,
        required=False,
        help="The end date of climatology period. Defaults to the last year of the dataset.",
    )


def configure_data_frequency(data: "xr.Dataset") -> str:
    ds_title = data.attrs.get("title").lower()
    if "monthly" in ds_title:
        ds_frequency = "monthly"
    elif "daily" in ds_title:
        ds_frequency = "daily"
    else:
        raise ValueError(
            "Frequency cannot be determined from the title of the dataset. "
            "Monthly or Daily datasets are required."
        )
    return ds_frequency


SEASON_MAP = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


def weighted_monthly_average(group, days_in_month):
    """This function calculates the monthly weighted average
    taking into account the different months lengths"""

    group_time_indices = group.time.values
    group_days_weights = days_in_month.sel(time=group_time_indices)

    total_weight = group_days_weights.sum(dim="time", skipna=True)
    weighted_sum = (group * group_days_weights).sum(dim="time", skipna=True)

    weighted_monthly_average = weighted_sum / total_weight

    return weighted_monthly_average


def climatology(
    data: "xr.Dataset",
    variable: str,
    frequency: str,
    start_date=None,
    end_date=None,
) -> "xr.Dataset":
    """Compute the climatological average of a variable at a given temporal frequency.

    The climatology is computed by grouping the data by the requested temporal
    period (day-of-year, month, or season) and averaging across all years
    within the specified date range.  Both monthly and daily source datasets
    are supported; the dataset frequency is inferred automatically from the
    ``title`` global attribute of the dataset.

    When the source dataset has **monthly** resolution:

    * *monthly* climatology is computed as a year-weighted average where each
      month is weighted by its number of days.
    * *daily* climatology cannot be computed and raises a :class:`ValueError`.
    * *seasonal* climatology groups months into DJF, MAM, JJA, SON and
      averages with equal weights.

    When the source dataset has **daily** resolution:

    * *daily* climatology is the mean for each calendar day-of-year across
      all years.
    * *monthly* climatology is obtained by first computing a daily
      climatology then averaging the day-of-year bins within each calendar
      month.
    * *seasonal* climatology groups day-of-year bins into DJF, MAM, JJA,
      SON and averages with equal weights.

    Args:
        data (xr.Dataset): Input dataset.  Must have a ``time`` coordinate
            and a ``title`` global attribute containing either
            ``"monthly"`` or ``"daily"``.
        variable (str): Name of the variable to compute the climatology for.
            Must be present in ``data.data_vars``.
        frequency (str): Temporal resolution of the output climatology.  One
            of ``"daily"``, ``"monthly"``, or ``"seasonally"``.
        start_date (datetime-like, optional): Start of the reference period.
            Defaults to the first time step in the dataset.
        end_date (datetime-like, optional): End of the reference period.
            Defaults to the last time step in the dataset.

    Returns:
        xr.Dataset: Dataset containing the climatological average of
        *variable*, with the time dimension replaced by the climatological
        coordinate (``month``, ``dayofyear``, or ``season``).

    Raises:
        ValueError: If *variable* is not found in the dataset, if the
            dataset frequency cannot be determined from its ``title``
            attribute, or if a daily climatology is requested from a monthly
            dataset.
    """
    # check the variable
    if variable not in data.data_vars:
        available_variables = list(data.data_vars.keys())
        raise ValueError(
            f"Variable {variable} is not found in the dataset."
            f"Choose from the available variables: {available_variables}"
        )

    data_var = data[variable]

    # define the period of climatology in case start_date and end_date are not specified
    if start_date is None:
        start_date = data["time"].values.min()
    if end_date is None:
        end_date = data["time"].values.max()

    data_var = data_var.sel(time=slice(start_date, end_date))

    ds_frequency = configure_data_frequency(data)

    LOGGER.info(f"Computing {frequency} climatology for variable: {variable}")

    # computing climatology with monthly datasets
    if ds_frequency == "monthly":
        if frequency == "monthly":
            data_var.coords["month"] = data_var["time"].dt.month
            days_in_month = data_var["time"].dt.days_in_month
            climatology_data = data_var.groupby("month").map(
                weighted_monthly_average, days_in_month=days_in_month
            )

        elif frequency == "daily":
            raise ValueError(
                "Impossible to compute daily climatology with monthly dataset."
                "A dataset with daily resolution is required."
            )

        elif frequency == "seasonally":
            months = data_var["time"].dt.month
            seasons = months.to_pandas().map(SEASON_MAP).values
            data_var.coords["season"] = ("time", seasons)
            climatology_data = data_var.groupby("season").mean(
                "time", skipna=True
            )

    # computing the climatology with daily datasets
    elif ds_frequency == "daily":
        daily_c = data_var.groupby("time.dayofyear").mean("time", skipna=True)

        if frequency == "monthly":
            months = data_var["time"].dt.month
            month_for_each_day = months.groupby(
                data_var["time"].dt.dayofyear
            ).first()
            daily_c = daily_c.assign_coords(
                month=("dayofyear", month_for_each_day.data)
            )
            climatology_data = daily_c.groupby("month").mean()

        elif frequency == "daily":
            month_for_each_day = (
                data_var["time"]
                .dt.month.groupby(data_var["time"].dt.dayofyear)
                .first()
            )
            climatology_data = daily_c.assign_coords(
                month=("dayofyear", month_for_each_day.data)
            )

        elif frequency == "seasonally":
            month_for_each_day = (
                data_var["time"]
                .dt.month.groupby(data_var["time"].dt.dayofyear)
                .first()
            )
            daily_c = daily_c.assign_coords(
                month=("dayofyear", month_for_each_day.data)
            )
            seasons = daily_c["month"].to_pandas().map(SEASON_MAP).values
            daily_c.coords["season"] = ("dayofyear", seasons)
            climatology_data = daily_c.groupby("season").mean()

    climatology = xr.Dataset({variable: climatology_data})
    return climatology
