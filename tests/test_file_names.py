from datetime import datetime
from itertools import product as cart_prod
from pathlib import Path

from hypothesis import given
from hypothesis import strategies

from medunda.tools.file_names import from_file_path_to_time_range
from medunda.tools.file_names import get_output_filename
from datetime import datetime, time


def date_interval(min_value: datetime, max_value:datetime):
    """
    Generates an interval of two dates between a min_value and
    a max_value.

    The start date is guaranteed to be before or equal than
    the end date.

    Even if this function take into account only dates (i.e.,
    the hours, minutes, seconds and microseconds are set to 0),
    the returned datetimes are of type datetime, not date.

    Args:
        min_value: The minimum value for the start date.
        max_value: The maximum value for the end date.

    Returns:
        A tuple of two datetimes, (start_date, end_date).
    """
    min_date = datetime.date(min_value)
    max_date = datetime.date(max_value)

    @strategies.composite
    def time_interval_strategy(draw):
        d1 = draw(strategies.dates(min_value=min_date, max_value=max_date))
        d2 = draw(strategies.dates(min_value=d1, max_value=max_date))
        dt1 = datetime.combine(d1, time.min)
        dt2 = datetime.combine(d2, time.min)
        return dt1, dt2
    return time_interval_strategy


@given(
    start_end_dates=date_interval(
        min_value=datetime(1950, 1, 1),
        max_value=datetime(2100, 12, 31)
    )()
)
def test_creating_and_reading_filenames_consistency(start_end_dates):
    """
    Test that the output filename can be created and then parsed back to
    the original start and end dates.
    """
    variables = ["thetao", "ph", "so"]
    frequencies = ["daily", "monthly"]
    domain_name = "domain"
    start_date, end_date = start_end_dates

    for variable, frequency in cart_prod(variables, frequencies):
        # Generate the output filename
        output_filename = get_output_filename(
            variable,
            frequency,
            start_date,
            end_date,
            domain_name
        )

        # Create a Path object
        file_path = Path(output_filename)

        # Extract the time range from the file path
        extracted_start, extracted_end = from_file_path_to_time_range(file_path)

        # Assert that the extracted dates match the original dates
        assert extracted_start == start_date
        assert extracted_end == end_date
