"""
Tools for dividing a time interval in several subintervals
"""
from datetime import datetime
from datetime import timedelta


def get_next_month(current_date: datetime):
    """
    Returns a datetime that represents the first date in the month
    that follows the one that contains `current_date`
    """
    if current_date.month == 12:
        return datetime(year=current_date.year + 1, month=1, day=1)

    return datetime(
        year=current_date.year, month=current_date.month + 1, day=1
    )

def get_next_year(current_date: datetime):
    """
    Like get_next_month, but for the years.
    """
    return datetime(
        year=current_date.year + 1, month=1, day=1
    )


def split_by_month(start_date: datetime, end_date:datetime) -> list:
    current_date = start_date
    resolution = timedelta(seconds=1)
    output = []
    while current_date <= end_date:
        next_month = get_next_month(current_date)
        output.append((current_date, next_month - resolution))
        current_date = next_month

        


def split_by_year(start_date: datetime, end_date:datetime) -> list:
    current_date = start_date
    resolution = timedelta(seconds=1)
    next_year = min(
        get_next_year(start_date) - resolution, end_date
    )
    intervals = [(current_date, next_year)]

    while intervals[-1][1] < end_date:
        current_date = get_next_year(intervals[-1][0])
        next_year = min(
            get_next_year(start_date) - resolution, end_date
        )
        intervals.append((current_date, next_year))

    return intervals
