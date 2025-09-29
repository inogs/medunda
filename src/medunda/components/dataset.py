from collections.abc import Iterable
from datetime import datetime
from logging import getLogger
from os import PathLike
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
import warnings
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import Field

from medunda.components.data_files import DataFile
from medunda.components.frequencies import Frequency
from medunda.domains.domain import Domain
from medunda.providers import get_provider
from medunda.tools.time_tables import split_by_month
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
        provider: The provider from which the data will be downloaded.
    """
    domain: Domain
    start_date: datetime
    end_date: datetime
    data_files: dict[VarName, tuple[DataFile, ...]] = {}
    frequency: Frequency = Frequency.MONTHLY
    provider: str = "cmems"
    provider_config: Path | None = None
    main_path : Path = Field(default_factory=Path, exclude=True)

    @field_validator("provider_config")
    def validate_file_size(cls, file_path):
        if file_path is not None and not file_path.exists():
            raise ValueError(f'Provider config file "{file_path}" does not exist.')
        return file_path.resolve() if file_path is not None else file_path

    def get_n_of_time_steps(self) -> int:
        """
        Returns the number of time steps in the dataset.

        This method calculates the number of time steps based on the start and
        end dates of the dataset.

        Returns:
            The number of time steps in the dataset.
        """
        if self.frequency == Frequency.MONTHLY:
            return len(split_by_month(self.start_date, self.end_date))
        elif self.frequency == Frequency.DAILY:
            # Assuming each day is a time step
            delta = self.end_date - self.start_date
            return delta.days + 1
        else:
            raise ValueError(
                f"Unsupported frequency '{self.frequency}'. "
                "Supported frequencies are 'monthly' and 'daily'."
            )

    def download_data(self):
        provider_class = get_provider(self.provider)
        provider = provider_class.create(config_file=self.provider_config)

        return provider.download_data(
            domain=self.domain,
            frequency=self.frequency,
            main_path=self.main_path,
            data_files=self.data_files
        )

    def get_variables(self) -> tuple[VarName, ...]:
        """
        Returns the variable names in the dataset.

        Returns:
            A tuple of variable names present in the dataset.
        """
        return tuple(self.data_files.keys())

    def get_mask(self) -> xr.Dataset:
        """
        Returns the mask dataset for the domain.

        The mask dataset contains the land-sea mask and other relevant
        information for the domain.

        Returns:
            An xarray Dataset containing the mask data.
        """
        LOGGER.debug("Retrieving mask for domain %s", self.domain.name)

        data = self.get_data(chunks={"time" : 1})

        vars_3d = []
        vars_2d = []
        for var_name, var_data in data.variables.items():
            if not ("latitude" in var_data.dims and "longitude" in var_data.dims):
                LOGGER.debug(
                    'Variable "%s" does not have latitude and longitude '
                    'dimensions, skipping it', var_name
                )
                continue
            LOGGER.debug(
                'Found variable "%s" with latitude and longitude',
                var_name
            )

            if "depth" in var_data.dims:
                LOGGER.debug(
                    'Variable "%s" has depth dimension, treating it '
                    'as a 3D var',
                    var_name
                )
                vars_3d.append(var_name)
            else:
                LOGGER.debug(
                    'Variable "%s" does not have depth dimension, treating it '
                    'as a 2D var',
                    var_name
                )
                vars_2d.append(var_name)

        if len(vars_2d) == 0 and len(vars_3d) == 0:
            raise ValueError(
                "No valid variables found in the dataset for mask retrieval."
            )

        if len(vars_3d) > 0:
            LOGGER.debug(
                "Using 3D variables for mask: %s", vars_3d
            )
            mask_var = vars_3d[0]
        else:
            LOGGER.debug(
                "Using 2D variables for mask: %s", vars_2d
            )
            mask_var = vars_2d[0]

        LOGGER.debug('Generating mask file from variable %s', mask_var)
        if "time" in data[mask_var].dims:
            LOGGER.debug(
                'Variable %s has a "time" dimension, using the first '
                'time step for the mask',
                mask_var
            )
            var_frame = data[mask_var].isel(time=0).drop_vars("time").compute()
        else:
            LOGGER.debug(
                'Variable "%s" does not have a time dimension, using '
                'the variable directly',
                mask_var
            )
            var_frame = data[mask_var].compute()

        # Ensure that the variable that we have read is aligned with the
        # other parts of the datasets. In other words, if we have
        # downloaded variables from different products, we must be sure
        # that we are working on the grid obtained by intersecting the
        # different products
        # var_frame, _ = xr.align(var_frame, data, join="inner")

        mask = np.ma.getmaskarray(var_frame.to_masked_array(copy=False))
        mask_dims = var_frame.dims

        # We invert (~) the mask, following the convention that True means
        # water and False means land.
        LOGGER.debug("Creating mask dataset for domain %s", self.domain.name)
        mask = xr.Dataset({"tmask": (mask_dims, ~mask)}, coords=data.coords)

        return mask

    def get_data_files(self) -> dict[VarName, tuple[Path, ...]]:
        """
        Returns the data files associated with the dataset.

        Returns:
            A dictionary mapping variable names to a tuple of file paths
        """
        return {
            var_name: tuple(self.main_path / data_file.path for data_file in data_files)
            for var_name, data_files in self.data_files.items()
        }

    def get_data(self,
                variables: Iterable[VarName] | None = None,
                chunks: dict | str | None = "auto"
        ) -> xr.Dataset:
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

        data_files = self.get_data_files()

        for variable in variables:
            LOGGER.debug('Processing variable "%s"', variable)
            variable_data_files = data_files[variable]
            LOGGER.debug(
                "There are %s data files associated with variable %s",
                len(variable_data_files),
                variable
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The specified chunks separate the stored chunks"
                )
                var_dataset = xr.open_mfdataset(
                    variable_data_files,
                    chunks=chunks
                )
            LOGGER.debug("Variable %s dataset opened successfully", variable)
            var_datasets.append(var_dataset)

        # Fixing the 0 value of the longitude! There is a subtle bug in the
        # Copernicus files: the value of the Greenwich meridian is
        # 1.4387797e-13 for the physical files and 1.4210855e-13 in the
        # biochemical ones. We need to put the same value before merging the
        # datasets, otherwise this value will be repeated twice
        for var_dataset in var_datasets:
            var_longitude = var_dataset.coords["longitude"]
            greenwich_index = np.abs(var_longitude).argmin()
            if var_longitude[greenwich_index].item() > 1e-6:
                # This dataset does not contain the Greenwich meridian.
                # We can skip this process
                continue
            new_longitude = var_longitude.values
            new_longitude[greenwich_index] = 0.
            var_longitude["longitude"] = new_longitude

        LOGGER.debug("Merging datasets for all variables")
        dataset_data = xr.merge(var_datasets, join="inner")
        return dataset_data


def read_dataset(dataset_path: PathLike | str) -> Dataset:
    """
    Reads a dataset from the specified path.

    Args:
        dataset_path: The path to the dataset file.

    Returns:
        A Dataset object containing the data from the file.
    """
    LOGGER.info('Reading dataset from "%s"', dataset_path)
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f'Dataset file "{dataset_path}" does not exist.'
        )

    if dataset_path.is_dir():
        LOGGER.debug(
            'Dataset path "%s" is a directory, reading medunda file',
            dataset_path
        )
        main_dataset_dir = dataset_path.resolve()
        dataset_path = dataset_path / "medunda_dataset.json"
    else:
        main_dataset_dir = dataset_path.parent.resolve()


    dataset_dict = yaml.safe_load(dataset_path.read_text())
    dataset = Dataset.model_validate(dataset_dict)

    dataset.main_path = main_dataset_dir

    LOGGER.debug('Dataset read successfully: %s', dataset)

    return dataset
