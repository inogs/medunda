import argparse
import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from sys import exit as sys_exit

import xarray as xr

from medunda.components.data_files import DataFile
from medunda.components.dataset import Dataset
from medunda.components.dataset import read_dataset
from medunda.components.frequencies import Frequency
from medunda.components.variables import VariableDataset
from medunda.domains.domain import ConcreteDomain
from medunda.domains.domain import read_domain
from medunda.providers import PROVIDERS
from medunda.providers import get_provider
from medunda.tools.argparse_utils import date_from_str
from medunda.tools.file_names import get_output_filename
from medunda.tools.logging_utils import configure_logger
from medunda.tools.time_tables import split_by_month
from medunda.tools.time_tables import split_by_year
from medunda.tools.typing import VarName

if __name__ == "__main__":
    LOGGER = logging.getLogger()
else:
    LOGGER = logging.getLogger(__name__)


def configure_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """
    parse command line arguments:
    --variable: the variable to download:
    --frequency: choose the frequency of the download: monthly or daily:
    --output-dir: directory to save the download file
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description="download monthly or daily data for a chosen variable"
        )

    subparsers = parser.add_subparsers(
        title="action",
        required=True,
        dest="action",
        help="Choose if creating a new dataset or resuming the download of an existing one",
    )

    create_subparser = subparsers.add_parser(
        "create",
        help="Download data and create a Medunda dataset",
    )

    resume_subparser = subparsers.add_parser(
        "resume",
        help="Resume the download of a previously created Medunda dataset",
    )

    create_subparser.add_argument(
        "--variables",
        type=str,
        choices=VariableDataset.all_variables().get_variable_names(),
        nargs="+",
        required=True,
        help="Name of the variable to download",
    )
    create_subparser.add_argument(
        "--start-date",
        type=date_from_str,
        required=True,
        help="Starting date for the download (format YYYY-MM-DD)",
    )
    create_subparser.add_argument(
        "--end-date",
        type=date_from_str,
        required=True,
        help="End date of the download (format YYYY-MM-DD)",
    )
    create_subparser.add_argument(
        "--frequency",
        type=str,
        choices=["monthly", "daily", "yearly"],
        required=False,
        default="monthly",
        help="Frequency of the downloaded data",
    )
    create_subparser.add_argument(
        "--domain",
        type=Path,
        required=True,
        help="Choose the domain",
    )

    create_subparser.add_argument(
        "--split-by",
        type=str,
        required=False,
        choices=["month", "year", "whole"],
        default="whole",
        help="Split the downloaded dataset by month, year or download all data together",
    )

    create_subparser.add_argument(  # input the directory to save the file
        "--output-dir",
        type=Path,
        required=True,
        help="directory where the downloaded file will be saved",
    )

    create_subparser.add_argument(
        "--provider",
        type=str,
        required=False,
        choices=sorted(list(PROVIDERS.keys())),
        default="cmems_mediterranean",
        help="The provider from which to download the data (default: cmems_mediterranean)",
    )

    create_subparser.add_argument(
        "--provider-config",
        type=Path,
        required=False,
        default=None,
        help="Path to a configuration file for the selected provider",
    )

    resume_subparser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="directory where the partially downloaded dataset is saved",
    )

    return parser


def download_data(
    variables: Iterable[VarName],
    output_dir: Path,
    frequency: Frequency,
    start: datetime,
    end: datetime,
    domain: ConcreteDomain,
    split_by: str = "whole",
    provider_class_name: str = "cmems",
    provider_config: Path | None = None,
) -> dict[VarName, tuple[Path, ...]]:
    """Download data for the specified variables, frequency, and time range."""
    # Check if the value of split_by is valid
    allowed_split_by = ("month", "year", "whole")
    if split_by not in allowed_split_by:
        raise ValueError(
            f'Invalid value for "split_by"; received {split_by} but the only '
            f"valid values are {allowed_split_by}"
        )

    # Prepare the time intervals based on the split_by parameter
    if split_by == "whole":
        time_intervals = [(start, end)]
    elif split_by == "year":
        time_intervals = split_by_year(start, end)
    elif split_by == "month":
        time_intervals = split_by_month(start, end)
    else:
        raise ValueError(f"Internal error: invalid parameter: {split_by}")

    # Create the provider instance that will handle the download
    provider = get_provider(provider_class_name).create(
        config_file=provider_config
    )
    if hasattr(provider, "name"):
        provider_name = getattr(provider, "name")
    else:
        provider_name = provider_class_name

    LOGGER.info(f'Using provider "{provider_name}" for the download')
    if len(provider.available_variables(frequency)) == 0:
        raise ValueError(
            f'Provider "{provider_name}" does not provide any variable at '
            f'frequency "{frequency}".'
        )
    for variable in variables:
        if variable not in provider.available_variables(frequency):
            raise ValueError(
                f'Variable "{variable}" is not available from provider '
                f'"{provider_name}" at frequency "{frequency}"'
            )

    # Prepare the output directory if not available
    output_dir.mkdir(exist_ok=True)

    if not output_dir.is_dir():
        raise ValueError(f"Output directory {output_dir} is not a directory")

    # We want to prepare a Dataset object that will contain the path of all the
    # files that we will download.
    downloaded_files: dict[VarName, tuple[DataFile, ...]] = {}

    # Dataset file
    dataset_file = output_dir / "medunda_dataset.json"
    if dataset_file.exists():
        raise IOError(
            f"Dataset file {dataset_file} already exists. "
            f"This means that the output directory {output_dir} has been "
            "already used for a previous download. Please choose a different "
            "output directory or delete the existing dataset file."
        )

    for variable in variables:
        # We save here the files that we download for this variable
        files_for_current_var: list[DataFile] = []

        var_output_dir = Path(variable) / str(frequency)

        for start_date, end_date in time_intervals:
            output_file_name = get_output_filename(
                variable=variable,
                frequency=str(frequency),
                start=start_date,
                end=end_date,
                domain_name=domain.name,
            )

            output_file_path = var_output_dir / output_file_name
            output_file = DataFile(
                start_date=start_date,
                end_date=end_date,
                variable=variable,
                path=output_file_path,
            )
            files_for_current_var.append(output_file)

        downloaded_files[variable] = tuple(files_for_current_var)

    # Create a Dataset object that will describe the data that we are going
    # to download.
    dataset = Dataset(
        domain=domain,
        start_date=start,
        end_date=end,
        data_files=downloaded_files,
        frequency=frequency,
        provider=provider_class_name,
        provider_config=provider_config,
        main_path=output_dir.absolute(),
    )

    # Save the dataset information to a JSON file
    dataset_file.write_text(dataset.model_dump_json(indent=4) + "\n")

    # Download the data for each variable. We delegate the actual download
    # to the Dataset class, which will handle the downloading of the data
    # files for each variable.
    LOGGER.info("Downloading data...")
    dataset.download_data()

    return dataset.get_data_files()


def validate_dataset(filepath, variable, max_depth: float | None):
    """Validates the dataset by checking for:
    dimensions, variables, and depth coverage."""

    LOGGER.info(f"dataset validated: {filepath}")

    with xr.open_dataset(filepath) as dataset:  # open the dataset using xarray
        # check for necessary dimensions
        required_dims = ["time", "latitude", "longitude"]

        for dim in required_dims:
            if dim not in dataset.dims:
                LOGGER.error(f"{dim}: dimension missing, validation failed")
                return False

        if variable not in dataset.data_vars:
            LOGGER.error(f"{variable}: variable missing. Validation failed")
            return False

        if "depth" in dataset.variables:
            depth_values = dataset["depth"].values
            LOGGER.debug(f"depth values: {depth_values}")
            if (
                depth_values.min() < 0
                or max_depth is not None
                and depth_values.max() > max_depth
            ):
                print(
                    "depth range is outside the expected bounds. Validation failed."
                )
                return False

    LOGGER.info("Successful Validation")
    return True


def downloader(args):
    if args.action == "create":
        domain = read_domain(args.domain)

        downloaded_files = download_data(
            variables=args.variables,
            output_dir=args.output_dir,
            frequency=Frequency(args.frequency),
            start=args.start_date,
            end=args.end_date,
            domain=domain,
            split_by=args.split_by,
            provider_class_name=args.provider,
            provider_config=args.provider_config,
        )

        for variable, files_for_var in downloaded_files.items():
            for filepath in files_for_var:
                if validate_dataset(
                    filepath,
                    variable,
                    max_depth=domain.bounding_box.maximum_depth,
                ):
                    LOGGER.info(
                        f"dataset validated for variable: '{variable}'"
                    )
                else:
                    LOGGER.warning(
                        f"failed dataset validation for variable:'{variable}'"
                    )

    else:
        dataset = read_dataset(args.dataset_dir)

        LOGGER.info("Resuming the download...")

        dataset.download_data()

    return 0


def main():
    configure_logger(LOGGER)
    args = configure_parser().parse_args()
    return downloader(args)


if __name__ == "__main__":
    sys_exit(main())
