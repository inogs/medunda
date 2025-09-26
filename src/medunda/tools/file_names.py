"""
In this module we store all the functions that handle, read, and modify the
names of the data files that Medunda stores
"""

import re
from datetime import datetime
from pathlib import Path

from medunda.tools.typing import VarName

FILE_NAME_MASK = re.compile(
    r"^(?P<domain>[a-zA-Z0-9-]+)_(?P<variable>[\w-]+)_"
    r"(?P<frequency>[a-zA-Z0-9-]+)_(?P<start_date>\d{4}-\d{2}-\d{2})--"
    r"(?P<end_date>\d{4}-\d{2}-\d{2})\.nc$"
)


def get_output_filename(
        variable: VarName,
        frequency: str,
        start: datetime,
        end: datetime,
        domain_name: str,
        ) -> str:
    """Generate the output filename based on the variable, frequency, start
    and end dates, and domain.

    Args:
        variable: The variable stored inside the file.
        frequency: The frequency of the data (e.g., daily, monthly).
        start: The start date of the data.
        end: The end date of the data.
        domain_name: The name of the domain.
    """
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    file_time = f"{start_str}--{end_str}"
    output_filename = f"{domain_name}_{variable}_{frequency}_{file_time}.nc"
    return output_filename


def from_file_path_to_time_range(file_path: Path) -> tuple[datetime, datetime]:
    """
    Extracts the start and end dates from a filename.

    The filename is expected to be in the format:
    "domain_variable_frequency_startdate--enddate.nc" where `startdate` and
    `enddate` are in YYYY-MM-DD format.

    Args:
        file_path: The path to the file.

    Returns:
        A tuple containing the start and end dates as datetime objects.
    """
    match = FILE_NAME_MASK.match(file_path.name)
    if not match:
        raise ValueError(
            f'Filename "{file_path.name}" does not match the expected format.'
        )
    start_date = datetime.strptime(match.group("start_date"), "%Y-%m-%d")
    end_date = datetime.strptime(match.group("end_date"), "%Y-%m-%d")

    return start_date, end_date
