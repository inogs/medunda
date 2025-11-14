import logging
from datetime import timedelta
from typing import Any
from typing import Callable
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
from medunda.tools.typing import VarName

LOGGER = logging.getLogger(__name__)


def _build_callable_wrapper(
    f: Callable[[xr.Dataset], dict[str, Any]],
) -> Callable[[np.ndarray, dict, dict[str, Any]], Delayed]:
    """
    Builds a delayed wrapper function for the provided function `f`

    This is a function used by the `BottomCellMap` class to transform the
    user-defined function `f` into a delayed function `f_wrapper` that
    can be executed in parallel. Read the documentation of the
    `BottomCellMap.map` to better understand what is `f` and the purpose
    of this function.

    This function solves the problem of passing a Dask Array as a delayed
    object to the user-defined function `f`.
    The method `BottomCellMap.map` decomposes the xarray Dataset
    into its components: time steps, coordinates, and data variables,
    where data_variables is a list of delayed objects that can be
    concatenated along the `model_time` dimension to reconstruct
    the original Dataset.
    The wrapper function `f_wrapper` takes these components and passes
    them to the user-defined function `f` after reconstructing
    the xarray Dataset from the delayed objects.
    Then it returns the result of the user-defined function `f`.

    Args:
        f: The user-defined function that takes an xarray Dataset as input
            and returns a dictionary where keys are column names and values
            are the data for those columns.

    Returns:
        A delayed function that takes time steps, coordinates, and data
        variables as input and returns the result of the user-defined function
        `f`.
    """

    @delayed
    def f_wrapper(
        time_steps: np.ndarray,
        coordinates: dict[str, Any],
        data: dict[VarName, list[np.ndarray]],
    ) -> dict[str, Any]:
        dataset_coords = {"model_time": time_steps}
        dataset_coords.update(coordinates)

        point_array = xr.Dataset(
            {d: ("model_time", np.concat(data[d])) for d in data},
            coords=dataset_coords,
        )

        return f(point_array)

    return f_wrapper


def _merge_together(
    results: Sequence[Delayed],
    dask_f_meta: pd.DataFrame | None,
    original_table: pd.DataFrame,
) -> dask.dataframe.DataFrame:
    """
    Merge the results with the original table into a Dask DataFrame

    Args:
        results: List of dictionaries containing the computed results
        dask_f_meta: An empty Pandas DataFrame. The columns for the
            output of this function will be copied from this input.
            If it is `None`, dask will use the structure of the
            first delayed object.
        original_table: Original pandas DataFrame

    Returns:
        A Dask DataFrame containing merged results
    """
    # Convert the list of dictionaries into a Dask DataFrame
    LOGGER.debug("Transforming all the objects into dataframes")
    delayed_dataframes = [
        delayed(pd.DataFrame)([r], index=[original_table.index[i]])
        for i, r in enumerate(results)
    ]

    LOGGER.debug("Creating a dataframe with all the outputs")
    results_df = dask.dataframe.from_delayed(
        delayed_dataframes, meta=dask_f_meta, verify_meta=False
    )
    LOGGER.debug(
        "Created a dataframe with %s partitions", results_df.npartitions
    )

    # Convert the original table to a Dask DataFrame
    LOGGER.debug("Transforming original inputs into dask dataframe")
    original_ddf = dask.dataframe.from_pandas(
        original_table, npartitions=results_df.npartitions
    )

    LOGGER.debug("Merging the results with the original table")
    output_ddf = dask.dataframe.merge(
        results_df, original_ddf, left_index=True, right_index=True, how="left"
    )

    return output_ddf


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

    def _generate_delayed(
        self,
        callable_wrapper: Callable[
            [np.ndarray, dict, dict[VarName, list]], Delayed
        ],
        point_dataset: xr.Dataset,
    ) -> Delayed:
        """
        Generate a delayed object that applies the callable_wrapper to a point

        This is a helper method that takes a callable_wrapper function
        and a point_dataset, and returns a delayed object that applies the
        callable_wrapper to the point_dataset. This method is invoked
        for each point in the point table when mapping the function to the
        bottom cell data (see the `map` method).

        In some sense, this method is the inverse of the
        `_build_callable_wrapper` function. While the output of the
        `_build_callable_wrapper` takes three arguments (time_steps,
        coordinates, data) and uses them to reconstruct an xarray Dataset,
        this method takes a point_dataset (which is an xarray Dataset) and
        extracts the time steps, coordinates, and data variables from it,
        and then passes them to the callable_wrapper function.

        In this way, the callable_wrapper function can be applied to the data
        of the dataset in a delayed manner, allowing for parallel computation
        of the results for each point in the point table.
        """
        coordinates = {}
        for coordinate in point_dataset.coords:
            if coordinate == "model_time":
                continue
            coord_value = point_dataset[coordinate].values
            assert np.ndim(coord_value) == 0, (
                f"Coordinate {coordinate} is not a scalar value."
            )
            coordinates[coordinate] = coord_value

        times = point_dataset["model_time"].values

        data: dict[VarName, list] = {}
        for k, v in point_dataset.data_vars.items():
            if k == "model_time":
                continue
            if k in coordinates:
                continue
            delayed_v = v.data.to_delayed()
            assert len(delayed_v.shape) == 1, (
                f"Data variable {k} is not a 1D array."
            )

            data[k] = delayed_v.tolist()

        return callable_wrapper(times, coordinates, data)

    def _split_into_chunks(
        self,
        lat_indices: xr.DataArray,
        lon_indices: xr.DataArray,
        lat_chunks: Sequence[int],
        lon_chunks: Sequence[int],
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

                sections.append(((lat_slice, lon_slice), positions))

        # Sort by length of sections (largest first)
        sections.sort(key=lambda x: -len(x[-1]))

        # Now we put the smallest section at the first place, so if we have to guess
        # the func_meta using the first point, we use a point that is in the smallest
        # section
        smallest_section = sections.pop(-1)
        sections.insert(0, smallest_section)

        assert sum(len(s[1]) for s in sections) == len(lat_indices), (
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

        This method need a Pandas DataFrame `point_table` that contains a
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
                will try to guess an approriate meta by executing f on the
                first point.
            delayed: If True, the method returns a Dask Delayed object that can
                be computed later. If False, the method computes the results
                immediately and returns a Pandas DataFrame.
        """
        LOGGER.debug("Applying the function to %s points...", len(point_table))

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
            latitude and longitude of the point, and converts them to the
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
        bottom_indices = bottom_index_map[lat_indices, lon_indices]

        # I convert the indices to xarray DataArrays
        # so that they can be used to index the xarray Dataset later
        lat_indices = xr.DataArray(
            lat_indices, dims="points", name="model_lat_index"
        )
        lon_indices = xr.DataArray(
            lon_indices, dims="points", name="model_lon_index"
        )
        bottom_indices = xr.DataArray(
            bottom_indices, dims="points", name="model_depth_index"
        )

        n_time_steps = self._dataset.get_n_of_time_steps()
        LOGGER.debug(
            "The dataset has approximately %s time steps", n_time_steps
        )

        # Here I define the chunks for the xarray Dataset
        # I use "auto" for depth, latitude, and longitude to allow
        # Dask to determine the optimal chunk size for these dimensions.
        # For time, I use "1000" if there are more than 2000 time steps
        # to avoid loading all time steps at once, which can be
        # memory-intensive.
        chunks: dict[str, str | int] = {
            "depth": "auto",
            "latitude": "auto",
            "longitude": "auto",
        }
        if n_time_steps > 500:
            chunks["time"] = "auto"
        else:
            chunks["time"] = -1  # Load all time steps at once

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

        callable_wrapper = _build_callable_wrapper(func)

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
        )

        dask_f_meta = None
        if func_meta is not None:
            LOGGER.debug("func_meta is: %s", func_meta)
            dask_f_meta: pd.DataFrame | None = make_meta(func_meta)
            LOGGER.debug(f"func_meta has been translated as: {dask_f_meta}")

        delayed_computations = []
        LOGGER.debug(
            "Generating the dask graph of all the %s tasks that will be "
            "executed",
            len(point_table),
        )
        for section_slice, section in sections:
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
            LOGGER.warning(bottom_slice)
            point_ds = data.isel(
                latitude=section_slice[0],
                longitude=section_slice[1],
                depth=bottom_slice,
            ).isel(
                latitude=lat_indices[section] - lat_shift,
                longitude=lon_indices[section] - lon_shift,
                depth=bottom_indices[section] - bottom_slice.start,
            )
            for point_indx, point_global_index in enumerate(section):
                point = point_table.iloc[point_global_index]
                point_time = getattr(point, self._time_column)

                LOGGER.debug(
                    "Point %s is associated with time %s; collecting the model "
                    "data starting from %s",
                    point_global_index,
                    point_time,
                    point_time - self._time_range,
                )
                time_slice = slice(
                    point_time - self._time_range,
                    point_time + timedelta(milliseconds=1),
                )
                time_index_slice = point_ds.indexes["time"].slice_indexer(
                    time_slice.start, time_slice.stop
                )
                LOGGER.debug(
                    "The temporal slice %s corresponds to the indices %s",
                    time_slice,
                    time_index_slice,
                )
                point_data = point_ds.isel(
                    points=point_indx, time=time_index_slice
                )

                point_data = point_data.rename(
                    {
                        "latitude": "model_latitude",
                        "longitude": "model_longitude",
                        "depth": "model_depth",
                        "time": "model_time",
                    }
                )

                point_latitude = getattr(point, self._lat_column)
                point_longitude = getattr(point, self._lon_column)
                point_time = getattr(point, self._time_column)

                point_new_coords = {
                    "latitude": point_latitude,
                    "longitude": point_longitude,
                    "time": point_time,
                    "distance": point.distance_from_model,
                }
                LOGGER.debug(
                    "Assigning the following coordinates to the point %s: %s",
                    point_indx,
                    point_new_coords,
                )
                point_data = point_data.assign_coords(**point_new_coords)

                if point_indx == 0 and dask_f_meta is None:
                    LOGGER.debug(
                        "Using this first point to guess the func_meta"
                    )
                    test_output = func(point_data.compute())
                    LOGGER.debug(
                        f"func returned the following output: {test_output}"
                    )
                    dask_f_meta = pd.DataFrame([test_output], index=[0]).head(
                        0
                    )
                    LOGGER.debug(
                        f"This is the inferred value of func_meta: {dask_f_meta}"
                    )

                LOGGER.debug(
                    "Genererating delayed task for point %s",
                    point_global_index,
                )
                delayed_task = self._generate_delayed(
                    callable_wrapper, point_data
                )
                delayed_computations.append(delayed_task)

        LOGGER.debug("Merging together all the points")
        final_output = _merge_together(
            delayed_computations,
            dask_f_meta=dask_f_meta,
            original_table=point_table[original_columns],
        )

        if not delayed:
            LOGGER.debug("Starting the overall computation...")
            final_output = final_output.compute()

        LOGGER.debug(
            "Computation of %s.map completed!", self.__class__.__name__
        )
        return final_output
