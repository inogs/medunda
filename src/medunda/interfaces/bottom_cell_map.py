import logging
import warnings
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence

import dask.dataframe
import numpy as np
import pandas as pd
import xarray as xr
from bitsea.commons.geodistances import compute_geodesic_distance
from bitsea.commons.mask import Mask
from dask.dataframe.dispatch import make_meta
from dask.delayed import Delayed
from dask.delayed import delayed

from medunda.components.dataset import Dataset
from medunda.tools.xarray_utils import DelayedDataset
from medunda.tools.xarray_utils import from_delayed
from medunda.tools.xarray_utils import to_delayed

LOGGER = logging.getLogger(__name__)


@delayed
def _extract_points(
    f: Callable,
    dataset: DelayedDataset,
    point_table: pd.DataFrame,
    depth_indices: xr.DataArray,
    indices_shift: dict,
    column_names: dict[Literal["time", "latitude", "longitude"], str],
    time_range: timedelta,
    preserve_columns: list[str],
):
    dataset = from_delayed(dataset)

    output = []

    for local_i, (global_i, point) in enumerate(point_table.iterrows()):
        point_time = point[column_names["time"]]
        LOGGER.debug(
            "Point %s is associated with time %s; collecting the model "
            "data starting from %s",
            global_i,
            point_time,
            point_time - time_range,
        )

        time_slice = slice(
            point_time - time_range,
            point_time + timedelta(milliseconds=1),
        )
        time_index_slice = dataset.indexes[column_names["time"]].slice_indexer(
            time_slice.start, time_slice.stop
        )
        LOGGER.debug(
            "The temporal slice %s corresponds to the indices %s",
            time_slice,
            time_index_slice,
        )

        dataset_selection = dict(
            time=time_index_slice,
            latitude=point["model_lat_index"] - indices_shift["latitude"],
            longitude=point["model_lon_index"] - indices_shift["longitude"],
            depth=depth_indices[local_i] - indices_shift["depth"],
        )
        LOGGER.debug(
            "Performing the following slicing on the dataset: %s",
            dataset_selection,
        )
        point_data = dataset.isel(**dataset_selection)

        point_data = point_data.rename(
            {
                "latitude": "model_latitude",
                "longitude": "model_longitude",
                "depth": "model_depth",
                "time": "model_time",
            }
        )

        point_latitude = getattr(point, column_names["latitude"])
        point_longitude = getattr(point, column_names["longitude"])
        point_time = getattr(point, column_names["time"])

        point_new_coords = {
            column_names["latitude"]: point_latitude,
            column_names["longitude"]: point_longitude,
            column_names["time"]: point_time,
            "distance": point.distance_from_model,
        }
        LOGGER.debug(
            "Assigning the following coordinates to the point %s: %s",
            (
                point[column_names["latitude"]],
                point[column_names["longitude"]],
            ),
            point_new_coords,
        )
        point_data = point_data.assign_coords(**point_new_coords)

        try:
            point_output = f(point_data)
        except Exception as e:
            LOGGER.error(
                f"Error in function {f.__name__} for point {global_i}, for which "
                f"the function received the following data: {point_data}\n\n"
                f"Error raised: \n{e}"
            )
            raise Exception(
                f"Error in function {f.__name__} for point {global_i}"
            ) from e

        if not isinstance(point_output, dict):
            raise ValueError(
                f"Function {f.__name__} must always return a dictionary. It "
                f"returned a {type(point_output)} instead."
            )

        point_as_dict = {
            p: v for p, v in point.to_dict().items() if p in preserve_columns
        }
        point_as_dict.update(point_output)

        output.append(point_as_dict)

    return output


def _merge_together(
    results: Sequence[Delayed],
    indices: list[pd.Index],
    dask_f_meta: pd.DataFrame | None,
) -> dask.dataframe.DataFrame:
    """
    Merge the results with the original table into a Dask DataFrame

    Args:
        results: List of dictionaries containing the computed results
        indices: A list of Pandas indices corresponding to the rows of the
            results
        dask_f_meta: An empty Pandas DataFrame. The columns for the
            output of this function will be copied from this input.
            If it is `None`, dask will use the structure of the
            first delayed object.

    Returns:
        A Dask DataFrame containing merged results
    """
    # Convert the list of dictionaries into a Dask DataFrame
    LOGGER.debug("Transforming all the objects into dataframes")
    delayed_dataframes = [
        delayed(pd.DataFrame)(r, index=i) for r, i in zip(results, indices)
    ]

    LOGGER.debug("Creating a dataframe with all the outputs")
    results_df = dask.dataframe.from_delayed(
        delayed_dataframes, meta=dask_f_meta, verify_meta=False
    )
    LOGGER.debug(
        "Created a dataframe with %s partitions", results_df.npartitions
    )

    return results_df


class BottomCellMap:
    """
    A class to map points in a dataset to temporal series of data from the bottom
    cell of a model grid.

    This class allows for the mapping of points defined by latitude, longitude, and time
    to the corresponding bottom cell in a model grid, taking into account the bathymetry
    and the model's spatial resolution.

    This class allows users to apply a function to the data at the bottom cell
    corresponding to each point, with the option to return the results as a Dask Delayed
    object for parallel computation or as a Pandas DataFrame for immediate use.

    The main method `map` takes a function that operates on an xarray Dataset and
    a Pandas DataFrame containing the points to be mapped. This dataframe must contain
    columns for latitude, longitude, and time, which are used to find the nearest model
    grid points and extract the corresponding data.

    Args:
        dataset (Dataset): The dataset containing the model data.
        time_range (timedelta): The time range around each point's time to consider for
            data extraction. For each point, data will be extracted starting from the
            point's time minus this range to the point's time.
        lat_column (str): The name of the column in the point table that contains latitude
            values. Defaults to "latitude".
        lon_column (str): The name of the column in the point table that contains longitude
            values. Defaults to "longitude".
        time_column (str): The name of the column in the point table that contains time
            values. Defaults to "time".
    """

    def __init__(
        self,
        dataset: Dataset,
        time_range: timedelta,
        lat_column: str = "latitude",
        lon_column: str = "longitude",
        time_column: str = "time",
    ):
        LOGGER.debug("Initializing a new %s instance", self.__class__.__name__)
        self._dataset = dataset
        self._time_range = time_range

        self._lat_column = lat_column
        self._lon_column = lon_column
        self._time_column = time_column

    def _split_into_chunks(
        self,
        lat_indices: xr.DataArray,
        lon_indices: xr.DataArray,
        lat_chunks: Sequence[int],
        lon_chunks: Sequence[int],
        times: Sequence[np.datetime64],
    ) -> tuple[tuple[tuple[slice, slice], np.ndarray], ...]:
        """
        Splits the lat_indices and lon_indices into chunks based on the provided chunk sizes.
        """
        LOGGER.debug("Longitude chunks = %s", lon_chunks)
        LOGGER.debug("Latitude chunks = %s", lat_chunks)

        lon_splits = np.cumsum(lon_chunks)[:-1]
        lat_splits = np.cumsum(lat_chunks)[:-1]
        LOGGER.debug("Longitude split = %s", lon_splits)
        LOGGER.debug("Latitude split = %s", lat_splits)

        lon_chunk_indices = np.searchsorted(
            lon_splits, lon_indices, side="right"
        )
        lat_chunk_indices = np.searchsorted(
            lat_splits, lat_indices, side="right"
        )

        start_date = np.datetime64(self._dataset.start_date, "s")
        time_range = np.timedelta64(self._time_range, "s")
        times = np.asarray(times)

        sections = []
        for lat_chunk_index in set(lat_chunk_indices):
            if lat_chunk_index == 0:
                lat_slice = slice(None, lat_splits[lat_chunk_index])
            elif lat_chunk_index == len(lat_splits):
                lat_slice = slice(lat_splits[lat_chunk_index - 1], None)
            else:
                lat_slice = slice(
                    lat_splits[lat_chunk_index - 1],
                    lat_splits[lat_chunk_index],
                )

            for lon_chunk_index in set(lon_chunk_indices):
                if lon_chunk_index == 0:
                    lon_slice = slice(None, lon_splits[lon_chunk_index])
                elif lon_chunk_index == len(lon_splits):
                    lon_slice = slice(lon_splits[lon_chunk_index - 1], None)
                else:
                    lon_slice = slice(
                        lon_splits[lon_chunk_index - 1],
                        lon_splits[lon_chunk_index],
                    )

                positions = np.nonzero(
                    np.logical_and(
                        lat_chunk_indices == lat_chunk_index,
                        lon_chunk_indices == lon_chunk_index,
                    )
                )[0]
                if positions.size == 0:
                    continue

                local_lon_indices = lon_indices[positions]
                local_lat_indices = lat_indices[positions]
                if lon_slice.start is not None:
                    assert np.min(local_lon_indices) >= lon_slice.start, (
                        f"lon_indices = {local_lon_indices}, lon_slice = {lon_slice}"
                    )
                if lon_slice.stop is not None:
                    assert np.max(local_lon_indices) < lon_slice.stop
                if lat_slice.start is not None:
                    assert np.min(local_lat_indices) >= lat_slice.start
                if lat_slice.stop is not None:
                    assert np.max(local_lat_indices) < lat_slice.stop

                current_time = start_date
                while current_time <= self._dataset.end_date:
                    inside_time_frame = np.logical_and(
                        times[positions] >= current_time,
                        times[positions] < current_time + time_range,
                    )
                    time_window = (current_time, current_time + time_range)
                    current_time += time_range

                    if not np.any(inside_time_frame):
                        continue

                    sections.append(
                        (
                            (lat_slice, lon_slice),
                            time_window,
                            positions[inside_time_frame],
                        )
                    )

        # Sort by length of sections (largest first)
        sections.sort(key=lambda x: -len(x[-1]))

        assert sum(len(s[-1]) for s in sections) == len(lat_indices), (
            "Some indices are missing after splitting into chunks."
        )

        return tuple(sections)

    def map(
        self,
        func: Callable[[xr.Dataset], dict[str, Any]],
        point_table: pd.DataFrame,
        func_meta: dict[str, np.typing.DTypeLike] | None = None,
        delayed: bool = False,
    ) -> pd.DataFrame | dask.dataframe.DataFrame:
        """
        Maps the provided function to the bottom cell data corresponding
        to each point in the point table.

        This method needs a Pandas DataFrame `point_table` that contains a
        column for latitude, a column for longitude, and a column for time.
        The name of these columns can be customized using the attributes of
        this class: `lat_column`, `lon_column`, and `time_column`.
        The output of this method is a Pandas DataFrame containing all the
        columns of the original `point_table`, plus additional columns that
        the user-defined function `func` returns.
        The `func` should accept an xarray Dataset as input and return a
        dictionary where keys are column names and values are the data for
        those columns. The Dataset that is passed to `func` will contain one
        variable for each variable in the original dataset, indexed by one
        dimension: `model_time`. The `model_time` dimension has an associated
        coordinate that contains the all the time steps of the model in the
        interval between the point's time minus `time_range` and the point's
        time (included). The values of each variable are taken from the bottom
        cell of the model grid corresponding to the point's location. The cell
        of the model is determined by the latitude and longitude of the point,
        and the bathymetry of the model grid. The corresponding cell on the
        surface is chosen as the one that is closest to the point's location,
        and that is not masked by the bathymetry (i.e., the point is not on
        land). Then, the data is extracted from the deepest cell of the model
        that contains water and that is on the same longitude and latitude
        of the surface cell.
        The Xarray Dataset contains also the following scalar coordinates:
        - latitude: the latitude of the original point.
        - longitude: the longitude of the original point.
        - time: the time of the original point.
        - `model_latitude`: the latitude of the model point corresponding
          to the bottom cell of the model grid at the point's location.
        - `model_longitude`: the longitude of the model point corresponding
          to the bottom cell of the model grid at the point's location.
        - `model_depth`: the depth of the model point corresponding to the
          bottom cell of the model grid at the point's location.
        - distance: the geodesic distance between the original point and
          the model point corresponding to the bottom cell of the model grid
          at the point's location (in meters).

        The `func` must return a dictionary whose keys can be choosen freely
        by the user, but they must not change among points. The values of the
        dictionary must be integers, floats, or strings. The values will
        be added as new columns to the output DataFrame. The output DataFrame
        will contain all the columns of the original `point_table`, plus the
        columns returned by the `func`. The names of the columns returned by
        the `func` must not conflict with the names of the columns in the
        `point_table`.

        The `func` is applied to each point in the `point_table`, and the
        results are collected in a Pandas DataFrame. If `delayed` is set to
        `True`, the method returns a Dask Delayed object that can be computed
        later. If `delayed` is set to `False`, the method computes the results
        immediately and returns a Pandas DataFrame.

        Args:
            func: A function that takes an xarray Dataset as input and
                returns a dictionary where keys are column names and values
                are the data for those columns.
            point_table: A Pandas DataFrame containing the points to be mapped.
                The DataFrame must contain columns for latitude, longitude,
                and time, which are used to find the nearest model grid points
                and extract the corresponding data.
            func_meta: Here you can specify the dtypes of the output of f. For
                example, if func returns two columns named A and B and the
                values of the column A are integers while the values of B are
                floating point numbers, f_meta must be
                {"A": int, "B", np.float32}. If it is not submitted, the code
                will try to guess an appropriate meta by executing f on the
                first point.
            delayed: If True, the method returns a Dask Delayed object that can
                be computed later. If False, the method computes the results
                immediately and returns a Pandas DataFrame.
        """
        LOGGER.debug("Applying the function to %s points...", len(point_table))

        if point_table.index.duplicated().any():
            warnings.warn(
                "point_table contains duplicated indices. Since this method "
                "alters the order of the rows, it is recommended to use "
                "an index without duplicates in order to be able to identify "
                "the original order of the points."
            )

        LOGGER.debug(
            "Reading the mask of the dataset to find the bottom cell indices"
        )
        mask = Mask.from_xarray(self._dataset.get_mask())

        LOGGER.debug(
            "Computing the bottom cell indices for the points in the point "
            "table"
        )
        bottom_index_map: np.ndarray = mask.bathymetry_in_cells() - 1

        original_columns = point_table.columns.tolist()

        def get_model_indices(row):
            """
            Given a row of the point table, this function extracts the
            latitude and longitude of the point and converts them to the
            nearest model point indices in the model grid. It returns a
            Pandas Series with the model longitude index and model latitude
            index.
            """
            lat = row[self._lat_column]
            lon = row[self._lon_column]
            # Convert the point's latitude and longitude to the nearest model
            # point indices
            model_lon_index, model_lat_index = (
                mask.convert_lon_lat_wetpoint_indices(
                    lat=lat, lon=lon, max_radius=None
                )
            )
            return pd.Series([model_lon_index, model_lat_index])

        LOGGER.debug(
            "Computing the model indices for the points in the point table..."
        )
        point_table[["model_lon_index", "model_lat_index"]] = (
            point_table.apply(get_model_indices, axis=1)
        )

        point_table["model_lat"] = mask.lat[point_table["model_lat_index"]]
        point_table["model_lon"] = mask.lon[point_table["model_lon_index"]]

        LOGGER.debug(
            "Computing the geodesic distance between the points and the "
            "model points corresponding to the bottom cell of the model grid"
        )
        point_table["distance_from_model"] = compute_geodesic_distance(
            lat1=point_table[self._lat_column],
            lon1=point_table[self._lon_column],
            lat2=point_table["model_lat"],
            lon2=point_table["model_lon"],
        )

        LOGGER.debug(
            "Extracting the bottom cell indices for all the %s points "
            "in the point table",
            len(point_table),
        )
        lat_indices = point_table["model_lat_index"].values.astype(int)
        lon_indices = point_table["model_lon_index"].values.astype(int)

        # I create an array of bottom indices
        bottom_indices = xr.DataArray(
            bottom_index_map[lat_indices, lon_indices],
            dims="points",
            name="model_depth_index",
        )

        n_time_steps = self._dataset.get_n_of_time_steps()
        LOGGER.debug(
            "The dataset has approximately %s time steps", n_time_steps
        )

        chunks: dict[str, str | int] = {
            "time": "auto",
            "depth": -1,
            "latitude": 50,
            "longitude": 50,
        }
        LOGGER.debug(
            "Reading the dataset to extract the data using the following "
            "chunks: %s",
            chunks,
        )
        data = self._dataset.get_data(chunks=chunks)
        LOGGER.debug(
            "Dataset opened! It has the following dimensions: %s",
            dict(data.sizes),
        )

        var3d = []
        for var_name in data.data_vars:
            if "depth" in data[var_name].dims:
                LOGGER.debug(
                    "Variable %s has depth dimension; selecting the bottom "
                    "cell data only",
                    var_name,
                )
                var3d.append(var_name)
            else:
                LOGGER.debug(
                    "Variable %s does not have depth dimension; using it "
                    "as is",
                    var_name,
                )

        if len(var3d) > 0:
            chunking_dataset = data[[*var3d]]
        else:
            chunking_dataset = data

        try:
            data_chunks = chunking_dataset.chunks
            LOGGER.debug(
                "The 3d variables of the dataset has the following chunks: %s",
                data_chunks,
            )
        except ValueError as e:
            LOGGER.debug(
                "Can not read the chunks of the dataset; probably it is not "
                'an homogeneous dataset; the error message was: "%s"',
                str(e),
            )
            data = data.unify_chunks()
            data_chunks = data.chunks

        # Now we need to split the points into several arrays, one for each
        # zone of the model grid
        if data_chunks is not None:
            lat_chunks = data_chunks["latitude"]
            lon_chunks = data_chunks["longitude"]
        else:
            lat_chunks = [data.sizes["latitude"]]
            lon_chunks = [data.sizes["longitude"]]

        sections = self._split_into_chunks(
            lat_indices=lat_indices,
            lon_indices=lon_indices,
            lat_chunks=lat_chunks,
            lon_chunks=lon_chunks,
            times=point_table[self._time_column],
        )

        column_names = {
            "time": self._time_column,
            "latitude": self._lat_column,
            "longitude": self._lon_column,
        }

        dask_f_meta = None
        if func_meta is not None:
            LOGGER.debug("func_meta is: %s", func_meta)
            dask_f_meta: pd.DataFrame | None = make_meta(func_meta)
            LOGGER.debug(f"func_meta has been translated as: {dask_f_meta}")

        if dask_f_meta is None:
            LOGGER.debug("We use the first point to guess the func_meta")
            test_point = point_table.iloc[0]
            time_slice = slice(
                test_point[column_names["time"]] - self._time_range,
                test_point[column_names["time"]] + timedelta(milliseconds=1),
            )
            time_index_slice = data.indexes[
                column_names["time"]
            ].slice_indexer(time_slice.start, time_slice.stop)
            lat_index = test_point["model_lat_index"]
            lon_index = test_point["model_lon_index"]
            test_dataset = data.isel(
                time=time_index_slice,
                latitude=slice(lat_index, lat_index + 1),
                longitude=slice(lon_index, lon_index + 1),
                depth=slice(
                    bottom_indices[0].item(), bottom_indices[0].item() + 1
                ),
            )
            local_shifts = {
                "latitude": lat_index,
                "longitude": lon_index,
                "depth": bottom_indices[0].item(),
            }
            meta_task = _extract_points(
                func,
                dataset=to_delayed(test_dataset),
                point_table=point_table.iloc[[0]],
                depth_indices=bottom_indices[0:1],
                indices_shift=local_shifts,
                column_names=column_names,
                time_range=self._time_range,
                preserve_columns=[],
            ).compute()

            LOGGER.debug(f"func returned the following output: {meta_task}")
            dask_f_meta = pd.DataFrame(meta_task, index=[0]).head(0)
            LOGGER.debug(
                f"This is the inferred value of func_meta: {dask_f_meta}"
            )

        delayed_computations = []
        LOGGER.debug(
            "Generating the dask graph of all the %s tasks that will be "
            "executed",
            len(point_table),
        )
        for section_slice, time_window, section in sections:
            lat_shift = (
                section_slice[0].start
                if section_slice[0].start is not None
                else 0
            )
            lon_shift = (
                section_slice[1].start
                if section_slice[1].start is not None
                else 0
            )
            min_bottom_index = bottom_indices[section].min().item()
            max_bottom_index = bottom_indices[section].max().item()
            bottom_slice = slice(min_bottom_index, max_bottom_index + 1)
            time_slice = data.indexes["time"].slice_indexer(
                time_window[0] - np.timedelta64(self._time_range, "s"),
                time_window[1],
            )
            LOGGER.debug(
                "Slicing a section in the time window %s", time_window
            )
            section_slice = dict(
                time=time_slice,
                latitude=section_slice[0],
                longitude=section_slice[1],
                depth=bottom_slice,
            )
            LOGGER.debug(
                "Slicing the dataset in the following area: %s", section_slice
            )
            point_ds = data.isel(**section_slice)

            local_shifts = {
                "latitude": lat_shift,
                "longitude": lon_shift,
                "depth": bottom_slice.start,
            }
            delayed_point_ds = to_delayed(point_ds)

            delayed_task = _extract_points(
                func,
                dataset=delayed_point_ds,
                point_table=point_table.iloc[section],
                depth_indices=bottom_indices[section],
                indices_shift=local_shifts,
                column_names=column_names,
                time_range=self._time_range,
                preserve_columns=original_columns,
            )

            delayed_computations.append(delayed_task)

        LOGGER.debug("Merging together all the points")
        final_output = _merge_together(
            delayed_computations,
            indices=[point_table.index[s] for _, _, s in sections],
            dask_f_meta=dask_f_meta,
        )

        if not delayed:
            LOGGER.debug("Starting the overall computation...")
            final_output = final_output.compute()

        LOGGER.debug(
            "Computation of %s.map completed!", self.__class__.__name__
        )
        return final_output
