import logging
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal

import netCDF4
import numpy as np
import xarray as xr
import yaml
from bitsea.commons.mask import Mask

from medunda.components.data_files import DataFile
from medunda.components.frequencies import Frequency
from medunda.components.variables import Variable
from medunda.components.variables import VariableDataset
from medunda.domains.domain import Domain
from medunda.providers.provider import Provider
from medunda.tools.typing import VarName

LOGGER = logging.getLogger(__name__)

AVE_FILE_MASK = re.compile(
    r"^ave\.(?P<date>\d{8}-\d{2}:\d{2}:\d{2})\.(?P<varname>[^.]+)\.nc$"
)


def allocate_medunda_data_file(
    var_name: VarName,
    var_units: str | None,
    meshmask: xr.Dataset,
    spatial_slices: dict[Literal["depth", "latitude", "longitude"], slice],
    time_steps: np.ndarray,
    f_pointer: netCDF4.Dataset,
):
    """
    Allocates and initializes variables in an empty netCDF file.

    This function is used by the function that writes one of the data files
    retrieved by the TarArchiveProvider class to allocate and prepare a NetCDF
    file for writing. It allocates dimensions and variables in the file, sets
    the file metadata, and initializes the variable with its fill value.

    This is a utility function used by the TarArchiveProvider class. Read the
    documentation of TarArchiveProvider.download_data for a detailed context
    of how this function is used.

    Args:
        var_name: Name of the variable to be stored in the netCDF file.
        var_units: Units of the variable to be stored in the netCDF file.
            If it is None, the units will be set to "1".
        meshmask: The meshmask dataset containing spatial coordinate values.
        spatial_slices: A dictionary specifying slices for spatial dimensions
            such as latitude, longitude, and depth.
        time_steps: Array containing the temporal data to be added to the file.
        f_pointer (netCDF4.Dataset): The netCDF file pointer where the variables
            and dimensions will be created.

    Raises:
        ValueError: If there is a mismatch in coordinate dimensions or invalid
            input data.

    Returns:
        None
    """
    fill_value = np.float32(1e20)

    coords = {
        "latitude": {"units": "degrees_north", "axis": "Y"},
        "longitude": {"units": "degrees_east", "axis": "X"},
        "depth": {"units": "m", "axis": "Z", "long_name": "Depth"},
    }
    for coord in coords:
        coord_elements = meshmask[coord].values[spatial_slices[coord]]
        f_pointer.createDimension(coord, coord_elements.shape[0])
        coord_var = f_pointer.createVariable(
            coord,
            coord_elements.dtype,
            dimensions=(coord,),
        )
        coord_var[:] = coord_elements
        coord_var.units = coords[coord]["units"]
        coord_var.axis = coords[coord]["axis"]
        coord_var.long_name = coords[coord].get("long_name", coord)
        coord_var.standard_name = coord
        if coord == "depth":
            coord_var.positive = "down"

    f_pointer.createDimension("time", time_steps.shape[0])
    time_var = f_pointer.createVariable(
        "time",
        np.int32,
        dimensions=("time",),
    )
    time_var.units = "seconds since 1970-01-01"
    time_var.units_long = "Seconds Since 1970-01-01"
    time_var.calendar = "standard"
    time_var.long_name = "Time"
    time_var.standard_name = "time"
    time_var.axis = "T"
    time_var[:] = time_steps.astype("datetime64[s]").astype(np.int64)

    dtype = np.float32
    dimensions = ("time", "depth", "latitude", "longitude")

    nc_var = f_pointer.createVariable(
        var_name,
        dtype,
        dimensions=dimensions,
        fill_value=fill_value,
        zlib=True,
        complevel=6,
    )
    if var_units is None:
        nc_var.units = "1"
    else:
        nc_var.units = var_units

    return nc_var


class LocalReanalysis(Provider):
    def __init__(
        self,
        variables: dict[VarName, dict[str, str]],
        frequencies: dict[Frequency, dict],
        meshmask: Path,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ):
        self._variables = variables
        self._frequencies = frequencies
        self._meshmask = meshmask
        self._start_time = start_time
        self._end_time = end_time

    @classmethod
    def get_name(cls) -> str:
        return "local_reanalysis"

    def available_variables(self, frequency: Frequency) -> VariableDataset:
        return VariableDataset.all_variables()

    def get_meshmask(self, main_path: Path) -> xr.Dataset:
        meshmask_path = main_path / "meshmask.nc"

        if not meshmask_path.exists():
            compression = {"zlib": True, "complevel": 9}

            bitsea_meshmask = Mask.from_file(Path(self._meshmask))
            xr_meshmask = bitsea_meshmask.to_xarray()
            xr_meshmask.to_netcdf(
                meshmask_path,
                encoding={v: compression for v in xr_meshmask.data_vars},
            )
            return xr_meshmask
        else:
            return xr.load_dataset(meshmask_path)

    def download_data(
        self,
        domain: Domain,
        frequency: Frequency,
        main_path: Path,
        data_files: Mapping[VarName, tuple[DataFile, ...]],
    ) -> None:
        # Get the current meshmask
        LOGGER.debug("Getting meshmask")
        meshmask = self.get_meshmask(main_path)

        archive_path = self._frequencies[frequency]["path"]
        LOGGER.debug("Opening archive at %s", archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Archive path {archive_path} does not exist."
            )
        if not archive_path.is_dir():
            raise ValueError(
                f"Archive path {archive_path} is not a directory."
            )

        available_files: dict[str, list] = {}
        for f in archive_path.rglob("*.nc"):
            file_match = AVE_FILE_MASK.match(f.name)
            if file_match is None:
                continue
            file_var = file_match.group("varname")
            file_time = datetime.strptime(
                file_match.group("date"), "%Y%m%d-%H:%M:%S"
            )
            if file_var not in available_files:
                available_files[file_var] = []
            available_files[file_var].append((file_time, f))
        for file_list in available_files.values():
            file_list.sort(key=lambda x: x[0])

        # Here we skip files that are already downloaded; to_be_downloaded is
        # a dictionary where the keys are the variable names and the values
        # are lists of files that must be downloaded for that variable. It has
        # the same structure as data_files, but only with the files that must
        # be downloaded.
        to_be_downloaded = {}
        for var_name, var_files in data_files.items():
            to_be_downloaded[var_name] = [
                f for f in var_files if not (main_path / f.path).exists()
            ]
        total_downloads = sum(
            len(files) for files in to_be_downloaded.values()
        )
        if total_downloads == 0:
            LOGGER.info(
                "All files are already downloaded. Skipping extraction."
            )
            return
        else:
            LOGGER.info(
                "Downloading %s files from %s", total_downloads, archive_path
            )

        # Compute the spatial slices that we will use to cut the data
        bounding_box = domain.bounding_box
        spatial_slices = {
            "latitude": meshmask.indexes["latitude"].slice_indexer(
                bounding_box.minimum_latitude, bounding_box.maximum_latitude
            ),
            "longitude": meshmask.indexes["longitude"].slice_indexer(
                bounding_box.minimum_longitude, bounding_box.maximum_longitude
            ),
            "depth": meshmask.indexes["depth"].slice_indexer(
                bounding_box.minimum_depth, bounding_box.maximum_depth
            ),
        }
        LOGGER.debug("Spatial slices to be used: %s", spatial_slices)

        for var_name, var_files in to_be_downloaded.items():
            LOGGER.info("Downloading data for variable %s", var_name)

            dataset_var_name = self._variables[var_name]["dataset_name"]
            LOGGER.debug(
                "Checking files for variable %s (named %s in the dataset)",
                var_name,
                dataset_var_name,
            )

            var_input_files = available_files[dataset_var_name]
            if len(var_input_files) == 0:
                raise ValueError(
                    f"No files associated with variable {var_name}"
                )

            for output_file in var_files:
                LOGGER.info("Downloading %s", output_file.path.name)

                # Ensure that the dataset_file_path are relative to the
                # main_path of the dataset.
                if output_file.path.is_absolute():
                    output_file_path = output_file.path
                else:
                    output_file_path = main_path / output_file.path

                output_file_path.parent.parent.mkdir(exist_ok=True)
                output_file_path.parent.mkdir(exist_ok=True)

                start_date = output_file.start_date
                end_date = output_file.end_date
                input_files = [
                    v
                    for v in var_input_files
                    if v[0] >= start_date and v[0] <= end_date
                ]

                if len(input_files) == 0:
                    raise ValueError(
                        f"No input files for file {output_file.path.name}"
                    )

                LOGGER.debug(
                    "File %s will be constructed from %s input files",
                    output_file.path.name,
                    len(input_files),
                )

                time_steps = np.array(
                    [i[0] for i in input_files], dtype="datetime64[s]"
                )

                temp_output_file = (
                    output_file_path.parent / f"tmp.{output_file_path.name}"
                )

                with netCDF4.Dataset(temp_output_file, "w") as f_out:
                    nc_var = allocate_medunda_data_file(
                        var_name=var_name,
                        var_units=self._variables[var_name]["unit"],
                        meshmask=meshmask,
                        spatial_slices=spatial_slices,
                        time_steps=time_steps,
                        f_pointer=f_out,
                    )

                    for t_index in range(time_steps.shape[0]):
                        _, t_input_file = input_files[t_index]

                        slices = (
                            slice(None),
                            spatial_slices["depth"],
                            spatial_slices["latitude"],
                            spatial_slices["longitude"],
                        )

                        with netCDF4.Dataset(t_input_file, "r") as ds:
                            data = np.asarray(
                                np.ma.getdata(ds[dataset_var_name][slices]),
                                dtype=np.float32,
                            )
                        nc_var[t_index, :] = data
                temp_output_file.rename(output_file_path)

    @staticmethod
    def _read_frequencies_in_config(
        config_content: dict, variables: dict[VarName, dict]
    ) -> dict[Frequency, dict]:
        output: dict[Frequency, dict] = {}
        if "frequencies" not in config_content:
            raise ValueError(
                'The configuration file must contain a "frequencies" section.'
            )
        if not isinstance(config_content["frequencies"], list):
            raise ValueError('The "frequencies" section must be a list')

        required_keys = {"frequency", "variables", "path"}
        for frequency in config_content["frequencies"]:
            if not isinstance(frequency, Mapping):
                raise ValueError(
                    'Each frequency in the "frequencies" section must be a '
                    "dictionary with the following keys: {}. "
                    "Received: {}".format(
                        ", ".join([f for f in sorted(list(required_keys))]),
                        frequency,
                    )
                )

            missing_keys = required_keys - set(frequency.keys())
            if missing_keys:
                raise ValueError(
                    'Each frequency in the "frequencies" section must contain '
                    "the following keys: {}. Missing keys in {} ---> {}".format(
                        ", ".join([f for f in sorted(list(required_keys))]),
                        frequency,
                        ", ".join([f for f in sorted(list(missing_keys))]),
                    )
                )

            for key in frequency.keys():
                if key not in required_keys:
                    supported_keys = ", ".join(
                        [f for f in sorted(list(required_keys))]
                    )
                    raise ValueError(
                        f'Invalid key "{key}" in frequency {frequency}. '
                        f"Supported keys are: {supported_keys}."
                    )

            try:
                frequency_value = Frequency(frequency["frequency"].lower())
            except Exception as e:
                raise ValueError(
                    f"Invalid frequency '{frequency['frequency']}' in the "
                    f"section {frequency}. Supported frequencies are 'monthly' "
                    "and 'daily'."
                ) from e

            frequency_path = Path(frequency["path"])
            frequency_vars = frequency["variables"]
            if not isinstance(frequency_vars, list):
                raise ValueError(
                    f'The "variables" key in frequency {frequency} must be a '
                    f"list. Received: {frequency_vars}"
                )
            for var in frequency_vars:
                if var not in variables:
                    raise ValueError(
                        f'Variable "{var}" in frequency {frequency} is not '
                        f'defined in the "variables" section.'
                    )

            if frequency_value in output:
                raise ValueError(
                    f"Frequency '{frequency_value}' is defined multiple times."
                )

            output[frequency_value] = {
                "path": frequency_path,
                "variables": frequency_vars,
            }

        return output

    @staticmethod
    def _read_variables_in_config(
        config_content: dict,
    ) -> dict[VarName, dict[str, str]]:
        variables_config: dict[VarName, dict[str, str]] = {}
        if "variables" not in config_content:
            raise ValueError(
                'The configuration file must contain a "variables" section.'
            )
        if not isinstance(config_content["variables"], list):
            raise ValueError('The "variables" section must be a list.')

        for raw_var in config_content["variables"]:
            if not isinstance(raw_var, Mapping):
                raise ValueError(
                    'Each variable in the "variables" section must be a '
                    "dictionary with the following mandatory keys: "
                    "variable and unit. Optional keys are: dataset_name. "
                    f"Received: {raw_var}"
                )
            if "variable" not in raw_var or "unit" not in raw_var:
                raise ValueError(
                    'Each variable in the "variables" section must contain '
                    'the keys "variable" and "unit". Optional keys are: '
                    f"dataset_name. Received: {raw_var}"
                )
            for key in raw_var.keys():
                if key not in {"variable", "unit", "dataset_name"}:
                    raise ValueError(
                        f'Invalid key "{key}" in variable {raw_var}. '
                        'Supported keys are: "variable", "unit", and '
                        '"dataset_name".'
                    )
            var_name = raw_var["variable"]
            try:
                current_var = Variable.get_by_name(var_name)
            except ValueError as e:
                raise ValueError(
                    f'Invalid variable "{var_name}" in the "variables" '
                    "section. Medunda does not support this variable."
                ) from e

            variables_config[var_name] = {
                "unit": raw_var["unit"],
                "dataset_name": raw_var.get("dataset_name", current_var.name),
            }

        return variables_config

    @classmethod
    def create(cls, config_file: Path | None = None) -> "Provider":
        if config_file is None:
            raise ValueError(
                f"A {cls.__name__} configuration file must be provided."
            )

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file {config_file} does not exist."
            )

        config_content = yaml.safe_load(config_file.read_text())

        variables = cls._read_variables_in_config(config_content)

        # Validate the frequencies
        frequencies = cls._read_frequencies_in_config(
            config_content, variables
        )

        meshmask = Path(config_content["meshmask"])

        return LocalReanalysis(
            variables=variables,
            frequencies=frequencies,
            meshmask=meshmask,
            start_time=config_content.get("start_time", None),
            end_time=config_content.get("end_time", None),
        )
