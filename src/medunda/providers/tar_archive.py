import logging
import re
import tarfile
from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Queue
from pathlib import Path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import xarray as xr
import yaml
from bitsea.commons.mask import Mask
from multiprocessing import cpu_count
from multiprocessing import Manager
from multiprocessing import Pool
from multiprocessing import Process

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
TASK = tuple[Path, VarName, str, str | None, np.ndarray, list[tuple[str, Path]]]


def reader(args: tuple[Queue, str, Path, str, dict[str, slice], datetime, Path]):
    """
    Extracts a single file from a tar archive and puts it in the queue.

    This function extracts a single netcdf file from a tar archive, reads its
    content, and puts it in a queue so that it can be written to disk by
    another process.
    This function is designed to be executed into a separate process by the
    multiprocessing module.

    This is a utility function used by the TarArchiveProvider class. Read the
    documentation of TarArchiveProvider.download_data for a detailed context
    of how this function is used.

    Args:
        args: A tuple containing the following elements:
            - data_queue: Queue to put the extracted data into
            - data_file_name: Name of the file to extract
            - tar_path: Path to the tar archive
            - nc_var_name: Name of the variable to read from the netcdf file
            - slices: Slice objects to use when reading the data from the netcdf
            - time_step: Time step of the data (ignored by this function but
                saved into the queue)
            - temp_dir: Temporary directory to extract the file to
    """
    (
        data_queue,
        data_file_name,
        tar_path,
        nc_var_name,
        spatial_slices,
        time_step,
        temp_dir
    ) = args

    LOGGER.debug("Extracting %s from %s", data_file_name, tar_path)
    with tarfile.open(tar_path, 'r') as tar:
        tar.extract(data_file_name, path=temp_dir)

    nc_data_file = temp_dir / data_file_name
    LOGGER.debug("Extracted %s to %s", data_file_name,nc_data_file)

    slices = (
        slice(None),
        spatial_slices["depth"],
        spatial_slices["latitude"],
        spatial_slices["longitude"]
    )

    LOGGER.debug("Reading %s", nc_data_file)
    try:
        with netCDF4.Dataset(nc_data_file, "r") as ds:
            data = np.asarray(
                np.ma.getdata(ds[nc_var_name][slices]),
                dtype=np.float32
        )
    except Exception as e:
        raise IOError(f"Error reading file {nc_data_file}") from e
    LOGGER.debug("Read an array of shape %s", data.shape)

    LOGGER.debug("Inserting data for time %s into the queue", time_step)
    data_queue.put((time_step, data))
    LOGGER.debug("Data for time %s inserted into the queue", time_step)

    LOGGER.debug("Removing temporary file %s", nc_data_file)
    nc_data_file.unlink()

    return None


def allocate_medunda_data_file(
        var_name: VarName,
        var_units: str | None,
        meshmask: xr.Dataset,
        spatial_slices,
        time_steps: np.ndarray,
        f_pointer:netCDF4.Dataset
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
        time_steps: Array representing the temporal data to be added to the file.
        f_pointer (netCDF4.Dataset): The netCDF file pointer where the variables and
            dimensions will be created.

    Raises:
        ValueError: If there is a mismatch in coordinate dimensions or invalid input data.

    Returns:
        None
    """
    fill_value = 1e20

    coords = {
        "latitude": {
            "units": "degrees_north",
            "axis": "Y"
        },
        "longitude": {
            "units": "degrees_east",
            "axis": "X"
        },
        "depth": {
            "units": "m",
            "axis": "Z",
            "long_name": "Depth"
        }
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
    time_var[:] = time_steps.astype('datetime64[s]').astype(np.int64)

    dtype = np.float32
    dimensions = ("time", "depth", "latitude", "longitude")

    nc_var = f_pointer.createVariable(
        var_name,
        dtype,
        dimensions=dimensions,
        fill_value=fill_value,
        zlib=True,
        complevel=6
    )
    if var_units is None:
        nc_var.units = "1"
    else:
        nc_var.units = var_units


def writer(
        data_queue: Queue,
        file_path: Path,
        var_name: VarName,
        var_units: str | None,
        meshmask: xr.Dataset,
        spatial_slices: dict[str, slice],
        time_steps: np.ndarray,
):
    """
    Writes data to a Medunda NetCDF file in a time-stepped manner.

    This function handles the process of writing time-stepped data from a queue
    into a temporary NetCDF file and then renaming the temporary file to the
    specified final file path. It continues to read data from the input queue
    and writes it to the designated time-step in the file until all time steps
    have been processed. After completing the data writing, the temporary file
    is renamed to the given file path.

    This function has been designed to be used as a process in a multiprocessing
    environment. It is used by the TarArchiveProvider class to write each file
    of a Medunda dataset. This function expects that other processes will add
    data to the queue concurrently by reading the netCDF files stored inside
    the tar files.

    Note:
    The function assumes the input queue contains tuples of time steps and
    corresponding data and processes the data sequentially.

    Args:
        data_queue: Queue containing data and time steps to be written.
        file_path: Path to the final NetCDF file.
        var_name: Name of the variable to be written.
        var_units: Units of the variable. If it is `None`, the variable is
            expected to be a pure number.
        meshmask: An Xarray dataset containing the meshmask.
        spatial_slices: Dictionary specifying the spatial slices<for the data.
        time_steps: Sequence of time steps to write into the file.

    Raises:
        Any exceptions raised during the file writing process are propagated.
    """
    temp_file_path = file_path.parent / ("tmp." + file_path.name)
    with netCDF4.Dataset(temp_file_path, "w") as f:
        allocate_medunda_data_file(
            var_name, var_units, meshmask, spatial_slices, time_steps, f
        )

        processed_data = 0
        while processed_data < time_steps.shape[0]:
            LOGGER.debug("Waiting for data to be written to file")
            time_step, data = data_queue.get()
            LOGGER.debug("Got data for time %s", time_step)

            current_index = np.argmin(np.abs(time_steps - time_step))
            LOGGER.debug(
                "Index for time-step %s is %s", time_step, current_index
            )

            f.variables[var_name][current_index, :] = data
            processed_data += 1
            LOGGER.debug(
                "Data for time %s written to file %s (%s/%s time steps written)",
                time_step,
                file_path.name,
                processed_data,
                time_steps.shape[0]
            )

    temp_file_path.rename(file_path)
    LOGGER.info("File %s has been written", file_path)


def execute_writing_task(args: tuple[TASK, dict, xr.Dataset, int]):
    task, spatial_slices, meshmask, n_processors = args
    output_file_path, var_name, nc_var_name, var_units, time_steps, file_requirements = task
    LOGGER.debug("Starting writing task for file %s", output_file_path)
    if n_processors < 2:
        raise ValueError("n_processors must be at least 2")

    queue_size = min(6, n_processors - 1)

    with Manager() as manager:
        data_queue = manager.Queue(maxsize=queue_size)

        writer_process = Process(
            target=writer,
            args=(data_queue, output_file_path, var_name, var_units, meshmask, spatial_slices, time_steps)
        )
        writer_process.start()

        with TemporaryDirectory() as t:
            temp_dir = Path(t)
            LOGGER.debug(
                "Created a temporary directory %s for saving the extracted "
                "nc files for %s", temp_dir, output_file_path
            )

            reader_args = []
            for i in range(time_steps.shape[0]):
                reader_args.append(
                    (
                        data_queue,
                        file_requirements[i][0],
                        file_requirements[i][1],
                        nc_var_name,
                        spatial_slices,
                        time_steps[i],
                        temp_dir
                    )
                )

            with Pool(processes=n_processors - 1) as p:
                task_executions = p.imap_unordered(reader, reader_args)

                deque(task_executions, maxlen=0)

        writer_process.join()


class TarArchiveProvider(Provider):
    """Provider for tar archives."""

    def __init__(self, name: str,
                 variables: dict[VarName, dict[str, str]],
                 frequencies: dict[Frequency, dict],
                 source: str = "local",
                 meshmask: dict[str, str | Path] | None = None,
                 start_time: datetime | None = None,
                 end_time: datetime | None = None):
        self.name = name
        self._variables = variables
        self._frequencies = frequencies
        self._source = source
        self._meshmask = meshmask
        self._start_time = start_time
        self._end_time = end_time

    @classmethod
    def get_name(cls) -> str:
        return "tar_archive"

    def available_variables(self, frequency: Frequency) -> VariableDataset:
        if frequency not in self._frequencies:
            return VariableDataset()
        return VariableDataset(self._frequencies[frequency]["variables"])

    def get_meshmask(self, main_path: Path) -> xr.Dataset:
        meshmask_path = main_path / "meshmask.nc"

        if not meshmask_path.exists():

            compression = {"zlib": True, "complevel": 9}

            bitsea_meshmask = Mask.from_file(self._meshmask["path"])
            xr_meshmask = bitsea_meshmask.to_xarray()
            xr_meshmask.to_netcdf(
                meshmask_path,
                encoding={v: compression for v in xr_meshmask.data_vars}
            )
            return xr_meshmask
        else:
            return xr.load_dataset(meshmask_path)

    def _get_variable_archives(self, var_name: VarName, frequency: Frequency):
        """
        Retrieves a list of available netCDF files inside a tar archive.

        This method searches for tar archives within the path associated with
        the given frequency and whose name (beside the extension) is equal to
        the name of the specified variable.

        It then iterates over the files inside each tar archive and retrieves
        the list of netcdf files that the archive contains. The archive is
        supposed to be flat, i.e., it should not contain any subdirectories.
        If this is not the case, an exception is raised. Inside the tar archive,
        the name of each netCDF file is used to extract the date associated with
        the file. If a netCDF file has a name that does not match the expected
        structure or if the netCDF file refers to a different variable,
        an exception is raised.

        This method returns a dictionary that associates the name of each netCDF
        that has been found with a dictionary with two keys: "date" and "tar".
        The value of "date" is the date associated with the netCDF file, and
        the value of "tar" is the path to the tar archive where the netCDF is
        stored.

        Parameters:
            var_name (VarName): The variable name for which the archives need
                to be retrieved.
            frequency (Frequency): The frequency classifying the dataset to find
                the corresponding archive path and files.

        Returns:
            A dictionary where the keys are the file names of the netCDF files,
            and the values are dictionaries containing their dates and their
            respective tar archives.

        Raises:
            ValueError: When a tar archive contains files with incorrect naming,
                unexpected folder structure, or mismatching variable names.
        """
        dataset_name = self._variables[var_name]["dataset_name"]
        archive_path = self._frequencies[frequency]["path"]

        nc_files = {}

        tar_files = sorted(list(archive_path.rglob(f"{dataset_name}.tar")))
        for archive_path in tar_files:
            if archive_path.parent.name == "bottom":
                LOGGER.debug(f"Ignoring {archive_path} because is on bottom")
                continue
            LOGGER.debug(f"Checking content of archive {archive_path}")
            with tarfile.open(archive_path, 'r') as tar:
                tar_members = tar.getmembers()

                for member in tar_members:
                    member_path = Path(member.name)
                    if member_path.suffix != ".nc":
                        LOGGER.debug(
                            "Ignoring file %s because its suffix is not .nc",
                            member.name
                        )
                        continue
                    if member_path.parent != Path('.'):
                        raise ValueError(
                            f"Tar archive {archive_path} contains file "
                            f"'{member.name}' with subfolder path. All files "
                            f"must be in root of archive."
                        )
                    ave_match = AVE_FILE_MASK.match(member_path.name)
                    if ave_match is None:
                        raise ValueError(
                            f"Tar archive {archive_path} contains file "
                            f"'{member.name}' with invalid name. All files "
                            f"must match the pattern 'ave.<date>.<varname>.nc'."
                        )
                    if ave_match.group("varname") != dataset_name:
                        raise ValueError(
                            "Tar archive contains files with different "
                            f"variable names: found var "
                            f"{ave_match.group('varname')} from file "
                            f"{member.name}, inside archive {archive_path}"
                        )
                    ave_date = datetime.strptime(
                        ave_match.group("date"),
                        "%Y%m%d-%H:%M:%S"
                    )

                    nc_files[member.name] = {
                        "date": ave_date,
                        "tar": archive_path
                    }
        return nc_files


    def download_data(
            self,
            domain: Domain,
            frequency: Frequency,
            main_path: Path,
            data_files: Mapping[VarName, tuple[DataFile, ...]]
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
        total_downloads = sum(len(files) for files in to_be_downloaded.values())
        if total_downloads == 0:
            LOGGER.info("All files are already downloaded. Skipping extraction.")
            return
        else:
            LOGGER.info(
                "Downloading %s files from %s", total_downloads, archive_path
            )

        # Each file that must be downloaded is a "task"; we prepare all the
        # information that we need for downloading a file in advance
        tasks: list[TASK] = []

        for var_name, files in to_be_downloaded.items():
            if len(files) == 0:
                LOGGER.debug("No files to download for variable %s", var_name)
                continue
            LOGGER.debug("Preparing extraction for variable %s", var_name)
            if var_name not in self._variables:
                raise ValueError(
                    f'Variable "{var_name}" is not defined in the provider '
                    f'"{self.name}".'
                )

            # Here we retrieve the list of netCDF files that are available in
            # all the tar files for the given variable and frequency. For every
            # medunda file (dataset_file), we must check which netCDF file must
            # be copied inside this file (the netCDF files have just one
            # time step, while medunda files go from a start date to an end
            # date).
            nc_files = self._get_variable_archives(var_name, frequency)

            for dataset_file in files:
                # The time steps that this dataset_file will contain. Every time
                # we found a netCDF file whose date is between the start and
                # end dates of the dataset_file, we add it to the time steps.
                dataset_time_steps = []
                # The files that must be read to create the dataset_file. Each
                # element of this list is a tuple with the name of the netcdf
                # file and the path to the tar that stores it.
                file_requirements: list[tuple[str, Path]] = []

                LOGGER.debug("Checking dependencies of file %s", dataset_file.path)
                for nc_name in sorted(nc_files.keys(), key=lambda x: nc_files[x]["date"]):
                    file_date = nc_files[nc_name]["date"]
                    if file_date < dataset_file.start_date or file_date > dataset_file.end_date:
                        continue
                    LOGGER.debug(
                        "We must read file %s from %s to create file %s",
                        nc_name,
                        nc_files[nc_name]["tar"],
                        dataset_file.path
                    )
                    dataset_time_steps.append(file_date)
                    file_requirements.append((nc_name, nc_files[nc_name]["tar"]))

                file_time_steps = np.array(dataset_time_steps, dtype='datetime64[s]')
                LOGGER.debug(
                    "File %s will contain %s time steps",
                    dataset_file.path,
                    file_time_steps.shape[0]
                )

                # Ensure that the dataset_file_path are relative to the
                # main_path of the dataset.
                if dataset_file.path.is_absolute():
                    dataset_file_path = dataset_file.path
                else:
                    dataset_file_path = main_path / dataset_file.path

                dataset_file_path.parent.parent.mkdir(exist_ok=True)
                dataset_file_path.parent.mkdir(exist_ok=True)

                # We add the "creation of the dataset_file" task to the list of
                # the tasks to be executed.
                tasks.append(
                    (
                        dataset_file_path,
                        var_name,
                        self._variables[var_name]["dataset_name"],
                        self._variables[var_name]["unit"],
                        file_time_steps,
                        file_requirements)
                )

        # Before reading the data, we need to take into account that we do not
        # need to read all the points, but just the ones that are inside the
        # domain. To do this, we create a slice for each dimension of the mesh
        # mask that contains the indices of the points that are inside the
        # domain. We then use these slices to extract the data from the netcdf
        # files.
        data_slices = {
            "latitude": meshmask.indexes["latitude"].slice_indexer(
                domain.minimum_latitude,
                domain.maximum_latitude
            ),
            "longitude": meshmask.indexes["longitude"].slice_indexer(
                domain.minimum_longitude,
                domain.maximum_longitude
            ),
            "depth": meshmask.indexes["depth"].slice_indexer(
                domain.minimum_depth,
                domain.maximum_depth
            )
        }
        LOGGER.debug("Spatial slices to be used: %s", data_slices)

        # The last thing we need to do is decide how to parallelize the work.
        # The trivial idea could be to have one process per task, but there is
        # a problem. If the user asked to download all the timesteps of just
        # one variable and she or he decides to not split the files by years or
        # months but to keep them all in one file, then the number of tasks
        # will be 1. In this case, the algorithm will be serial, and we waste
        # the possibility of decompressing the tar files in parallel. On the
        # other hand, if the user asks for many variables and she or he decides
        # to split the files by years or months, then the number of tasks can be
        # very big and, in this case, the more efficient implementation is to
        # distribute the tasks among the available processors. Here we try to
        # guess how many tasks can be executed in parallel and how many
        # processors we can allocate for each task. In any case, each task will
        # require at least 2 processes: one for writing and one for
        # decompressing the tar files. If we have more than 2 processes per
        # task, all the processes but the first one will be used to read the
        # netCDF files from the tar archives. This happens because only one
        # process can write on the same dataset file (netCDF does not support
        # multithreading).
        n_processors = cpu_count()
        simultaneous_tasks = min(len(tasks), n_processors // 2)
        processes_per_task = max(2, n_processors // simultaneous_tasks)
        # We prefer overloading than keeping some cores idle
        if processes_per_task * simultaneous_tasks < n_processors and simultaneous_tasks < len(tasks):
            simultaneous_tasks += 1
        LOGGER.debug(
            "Running %s task simultaneously, with %s processes each",
            simultaneous_tasks,
            processes_per_task
        )

        thread_args = [
            (task, data_slices, meshmask, processes_per_task) for task in tasks
        ]
        with ThreadPoolExecutor(max_workers=simultaneous_tasks) as executor:
            deque(
                executor.map(execute_writing_task, thread_args),
                maxlen=0
            )
        LOGGER.info("All files downloaded and extracted")


    @staticmethod
    def _read_frequencies_in_config(config_content: dict, variables: dict[VarName, dict]) -> dict[Frequency, dict]:
        output: dict[Frequency, dict] = {}
        if "frequencies" not in config_content:
            raise ValueError(
                'The configuration file must contain a "frequencies" section.'
            )
        if not isinstance(config_content["frequencies"], list):
            raise ValueError(
                'The "frequencies" section must be a list'
            )

        required_keys = {"frequency", "variables", "path"}
        for frequency in config_content["frequencies"]:
            if not isinstance(frequency, Mapping):
                raise ValueError(
                    'Each frequency in the "frequencies" section must be a '
                    "dictionary with the following keys: {}. "
                    "Received: {}".format(
                        ", ". join([f for f in sorted(list(required_keys))]),
                        frequency
                    )
                )

            missing_keys = required_keys - set(frequency.keys())
            if missing_keys:
                raise ValueError(
                    'Each frequency in the "frequencies" section must contain '
                    "the following keys: {}. Missing keys in {} ---> {}".format(
                        ", ". join([f for f in sorted(list(required_keys))]),
                        frequency,
                        ", ".join([f for f in sorted(list(missing_keys))])
                    )
                )

            for key in frequency.keys():
                if key not in required_keys:
                    raise ValueError(
                        f'Invalid key "{key}" in frequency {frequency}. '
                        f'Supported keys are: {", ". join([f for f in sorted(list(required_keys))])}.'
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
                    f'The "variables" key in frequency {frequency} must be a list. Received: {frequency_vars}'
                )
            for var in frequency_vars:
                if var not in variables:
                    raise ValueError(
                        f'Variable "{var}" in frequency {frequency} is not defined in the "variables" section.'
                    )

            if frequency_value in output:
                raise ValueError(
                    f"Frequency '{frequency_value}' is defined multiple times."
                )

            output[frequency_value] = {
                "path": frequency_path,
                "variables": frequency_vars
            }

        return output


    @staticmethod
    def _read_variables_in_config(config_content: dict) -> dict[VarName, dict[str, str]]:
        variables_config: dict[VarName, dict[str, str]] = {}
        if "variables" not in config_content:
            raise ValueError(
                'The configuration file must contain a "variables" section.'
            )
        if not isinstance(config_content["variables"], list):
            raise ValueError(
                'The "variables" section must be a list.'
            )

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
                    f'dataset_name. Received: {raw_var}'
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
                "dataset_name": raw_var.get("dataset_name", current_var.name)
            }

        return variables_config


    @staticmethod
    def _read_meshmask_in_config(config_content: dict) -> dict[str, str | Path]:
        if "meshmask" not in config_content:
            raise ValueError(
                'The configuration file must contain a "meshmask" section.'
            )
        if not isinstance(config_content["meshmask"], Mapping):
            raise ValueError(
                'The "meshmask" section must be a dictionary.'
            )

        if "type" not in config_content["meshmask"]:
            raise ValueError(
                'The "meshmask" section must contain a "type" key.'
            )

        if "path" in config_content["meshmask"]:
            config_content["meshmask"]["path"] = Path(
                config_content["meshmask"]["path"]
            )

        return dict(config_content["meshmask"])


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
        frequencies = cls._read_frequencies_in_config(config_content, variables)

        meshmask = cls._read_meshmask_in_config(config_content)

        return TarArchiveProvider(
            name=config_content.get("name", "tar_archive"),
            variables=variables,
            frequencies=frequencies,
            source=config_content.get("source", "local"),
            meshmask=meshmask,
            start_time=config_content.get("start_time", None),
            end_time=config_content.get("end_time", None)
        )
