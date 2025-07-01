"""
Tools for dividing a time interval into several subintervals.

This module provides utility functions to split a time range into intervals
of months or years, or using a custom interval function. Useful for time-based
data processing, reporting, or scheduling tasks.
"""

from collections.abc import Callable
from datetime import datetime
from datetime import timedelta
from itertools import pairwise


def get_next_month(current_date: datetime) -> datetime:
    """
    Returns the first day of the month following the month containing
    `current_date`.

    Args:
        current_date: The reference date.

    Returns:
        The first day of the next month.
    """
    if current_date.month == 12:
        return datetime(year=current_date.year + 1, month=1, day=1)

    return datetime(
        year=current_date.year, month=current_date.month + 1, day=1
    )


def get_next_year(current_date: datetime) -> datetime:
    """
    Returns the first day of the year following the year containing
    `current_date`.

    Args:
        current_date: The reference date.

    Returns:
        The first day of the next year.
    """
    return datetime(
        year=current_date.year + 1, month=1, day=1
    )


def split_into_intervals(
        start_date: datetime,
        end_date: datetime,
        get_next_point: Callable[[datetime], datetime],
        resolution: timedelta = timedelta(seconds=1)
        ) -> list[tuple[datetime, datetime]]:
    """
    Splits a time interval into subintervals using a custom function to
    determine split points.

    Args:
        start_date: The start of the interval.
        end_date: The end of the interval.
        get_next_point: Function that returns the next split point.
        resolution: The smallest unit of time to subtract from each interval's
            end (except the last). Defaults to 1 second.

    Returns:
        List of (start, end) tuples for each subinterval.

    Raises:
        ValueError: If start_date is after end_date.
    """
    if start_date > end_date:
        raise ValueError("start_date must be before end_date")

    current_date = start_date
    split_points = [current_date]

    while current_date < end_date:
        current_date = min(
            get_next_point(current_date), end_date
        )
        split_points.append(current_date)
    
    intervals = []
    for start, end in pairwise(split_points):
        interval_end = end if end == end_date else end - resolution
        intervals.append((start, interval_end))

    return intervals


def split_by_month(
        start_date: datetime,
        end_date:datetime
        ) -> list[tuple[datetime, datetime]]:
    """
    Splits a time interval into monthly subintervals.

    Args:
        start_date: The start of the interval.
        end_date: The end of the interval.

    Returns:
        List of (start, end) tuples for each month.
    """
    return split_into_intervals(
        start_date,
        end_date,
        get_next_point=get_next_month,
        resolution=timedelta(seconds=1)
    )


def split_by_year(
        start_date: datetime,
        end_date:datetime
        ) -> list[tuple[datetime, datetime]]:
    """
    Splits a time interval into yearly subintervals.

    Args:
        start_date: The start of the interval.
        end_date: The end of the interval.

    Returns:
        List of (start, end) tuples for each year.
    """
    return split_into_intervals(
        start_date,
        end_date,
        get_next_point=get_next_year,
        resolution=timedelta(seconds=1)
    )
