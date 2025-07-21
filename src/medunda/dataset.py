from collections.abc import Iterable
from datetime import datetime
from datetime import timezone
from logging import getLogger
from pathlib import Path
import shutil

import copernicusmarine
import xarray as xr
from pydantic import BaseModel

from medunda.domains.domain import Domain
from medunda.sources.cmems import search_for_product
from medunda.tools.file_names import from_file_path_to_time_range
from medunda.tools.typing import VarName


LOGGER = getLogger(__name__)


class Dataset(BaseModel):
    """
    A class to represent a dataset for a specific domain.
    
    Attributes:
        domain: The domain associated with the dataset.
        start_date: The start date of the dataset.
        end_date: The end date of the dataset.
        data_files: A dictionary mapping variable names to a tuple of files
            that contain the data for that variable.
        frequency: The frequency of the dataset.
        source: The source from which the data will be downloaded.
    """
    domain: Domain
    start_date: datetime
    end_date: datetime
    data_files: dict[VarName, tuple[Path, ...]] = {}
    frequency: str = "monthly"
    source: str = "cmems"

    def download_data(self):
        """
        Downloads the data for the dataset.
        
        This method downloads all the missing data files for the dataset.
        If a file already exists, it will not be downloaded again.
        """
        for var_name, files in self.data_files.items():
            LOGGER.debug(
                f'Downloading data for variable "%s": %s files will be '
                f'downloaded',
                var_name,
                len(files)
            )

            product_id = search_for_product(
                var_name=var_name,
                frequency=self.frequency
            )
            LOGGER.debug(
                "The product associated with variable %s is: %s",
                var_name,
                product_id
            )

            for file in files:
                if file.exists():
                    LOGGER.debug(
                        'File "%s" already exists. Skipping download.', file
                    )
                    continue

                LOGGER.info('Downloading file "%s"', file)
                
                # Ensure the parent directory exists
                file_path = file.absolute()
                file_path.parent.mkdir(parents=True, exist_ok=True)

                output_filename = file_path.name
                temp_filename = "tmp." + output_filename
                temp_file_path = file_path.parent / temp_filename

                if temp_file_path.is_file():
                    LOGGER.debug(
                        '"%s" already exists. I will delete it!',
                        temp_file_path
                    )
                    temp_file_path.unlink()

                start, end = from_file_path_to_time_range(file_path)

                # Copernicusmarine API interprets datetimes as UTC,
                # so we need to ensure that the start and end datetimes
                # are timezone-aware and set to UTC.
                start = start.replace(tzinfo=timezone.utc)
                end = end.replace(tzinfo=timezone.utc)
                copernicusmarine.subset(
                    dataset_id=product_id,
                    variables=[var_name],
                    start_datetime=start,
                    end_datetime=end,
                    output_filename=str(temp_file_path),
                    **self.domain.model_dump(exclude={"name"})
                )

                LOGGER.debug(
                    'Moving file "%s" to "%s"', temp_file_path, file_path
                )
                shutil.move(temp_file_path, file_path)
    
    def get_variables(self) -> tuple[VarName, ...]:
        """
        Returns the variable names in the dataset.
        
        Returns:
            A tuple of variable names (VarName) present in the dataset.
        """
        return tuple(self.data_files.keys())


    def get_data(self, variables: Iterable[VarName] | None = None) -> xr.Dataset:
        """
        Returns an xarray Dataset for the specified variable.
        
        Args:
            variables: An iterable of variable names (VarName) to include in the
                dataset. If None, all variables in the dataset will be included.
        
        Raises:
            ValueError: If no variables are specified.
            
        Returns:
            An xarray Dataset containing the data for the specified variables.
        """ 
        var_datasets = []

        if variables is None:
            LOGGER.debug(
                "No specific variables requested, using all variables"
            )
            variables = self.get_variables()
        variables = tuple(variables)

        if len(variables) == 0:
            raise ValueError("No variables specified for dataset retrieval.")

        for variable in variables:
            LOGGER.debug('Processing variable "%s"', variable)
            variable_data_files = self.data_files[variable]
            LOGGER.debug(
                "There are %s data files associated with variable %s",
                len(variable_data_files),
                variable
            )
            var_dataset = xr.open_mfdataset(variable_data_files)
            LOGGER.debug("Variable %s dataset opened successfully", variable)
            var_datasets.append(var_dataset)
        LOGGER.debug("Merging datasets for all variables")
        dataset_data = xr.merge(var_datasets)
        return dataset_data


def read_dataset(dataset_path: Path) -> Dataset:
    """
    Reads a dataset from the specified path.
    
    Args:
        dataset_path: The path to the dataset file.
        
    Returns:
        A Dataset object containing the data from the file.
    """
    LOGGER.info('Reading dataset from "%s"', dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f'Dataset file "{dataset_path}" does not exist.'
        )
    
    if dataset_path.is_dir():
        LOGGER.debug(
            'Dataset path "%s" is a directory, reading medunda file',
            dataset_path
        )
        dataset_path = dataset_path / "medunda_dataset.json"

    dataset = Dataset.model_validate_json(dataset_path.read_text())
    LOGGER.debug('Dataset read successfully: %s', dataset)

    return dataset
