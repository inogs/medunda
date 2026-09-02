import logging
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence

import dask.dataframe
import numpy as np
import pandas as pd
from bitsea.commons.geodistances import compute_geodesic_distance
from bitsea.commons.mask import Mask
from dask.dataframe.dispatch import make_meta
from dask.delayed import Delayed
from dask.delayed import delayed

from medunda.components.geodata import GeoDataCollection
from medunda.tools.lazy_imports import xr
from medunda.tools.xarray_utils import DelayedDataset
from medunda.tools.xarray_utils import from_delayed
from medunda.tools.xarray_utils import to_delayed

LOGGER = logging.getLogger(__name__)

# If the chunks that come natively from the files on the disk are
# already smaller than these many grid points (along latitude and along
# longitude), they are used as they are. Otherwise, `BottomCellMap.map`
# rechunks the dataset along latitude and longitude into blocks of
# `TARGET_SPATIAL_CHUNK_SIZE` points, so that points scattered across the
# domain can be grouped into small, spatially localized sections instead of
# a handful of sections that span almost the whole domain.
MAX_REASONABLE_SPATIAL_CHUNK_SIZE = 200
TARGET_SPATIAL_CHUNK_SIZE = 100


@delayed
def _extract_points(
    f: Callable,
    dataset: DelayedDataset,
    point_table: pd.DataFrame,
    depth_indices: xr.DataArray | None,
    indices_shift: dict[Literal["latitude", "longitude", "depth"], int],
    column_names: dict[Literal["time", "latitude", "longitude"], str],
    time_range: timedelta,
    preserve_columns: list[str],
    has_depth: bool,
):
    """
    Extracts, for a batch of points that share the same spatial chunk
    and time window, the corresponding data from `dataset` and applies
    `f` to each of them.

    Because this function is wrapped with `dask.delayed.delayed`,
    calling it does not run any of the code below immediately: it
    returns a `Delayed` object, and the list of dictionaries described
    in "Returns" is only produced once that object is computed.

    For each row of `point_table`, this function slices `dataset`
    around the point's time and (via `indices_shift`) its position in
    the model grid, renames the sliced dimensions with a `model_`
    prefix, assigns the point's original latitude, longitude, time and
    distance from the model grid point as new scalar coordinates, and
    finally calls `f` on the resulting Dataset.

    Args:
        f: A function that takes an xarray Dataset describing one
            point's data and returns a dictionary mapping output
            column names to values.
        dataset: The dataset (wrapped as a `DelayedDataset`), already
            sliced so that it only covers the spatial chunk and time
            window relevant to `point_table`.
        point_table: The rows of the original point table that fall
            inside this dataset chunk. Must contain the
            `model_lat_index` and `model_lon_index` columns.
        depth_indices: For each row of `point_table` (in the same
            order), the absolute index, along the `depth` dimension of
            the full model grid, of the model's bottom cell at that
            point. Must be `None` when `has_depth` is `False`.
        indices_shift: The offset, along each of `latitude`,
            `longitude` and (when `has_depth` is `True`) `depth`,
            between the absolute indices in the full model grid (as
            used by `point_table` and `depth_indices`) and the indices
            of the already-sliced `dataset`.
        column_names: Maps the logical roles `"time"`, `"latitude"`
            and `"longitude"` to the actual column names used in
            `point_table`.
        time_range: For each point, data is extracted starting from
            the point's time minus `time_range` up to the point's time
            (included).
        preserve_columns: Names of the columns of `point_table` that
            must be copied, unchanged, into each output dictionary.
        has_depth: Whether `dataset` has a `depth` dimension (i.e. the
            underlying `GeoDataCollection` is 3D). When `False`,
            `depth_indices` is ignored and no depth-related selection
            or renaming is performed.

    Returns:
        A list with one dictionary per row of `point_table`, in the
        same order. Each dictionary contains the columns listed in
        `preserve_columns` plus all the keys returned by `f`.
    """
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
        time_index_slice = dataset.indexes["time"].slice_indexer(
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
        )
        if has_depth:
            dataset_selection["depth"] = (
                depth_indices[local_i] - indices_shift["depth"]
            )
        LOGGER.debug(
            "Performing the following slicing on the dataset: %s",
            dataset_selection,
        )
        point_data = dataset.isel(**dataset_selection)

        rename_map = {
            "latitude": "model_latitude",
            "longitude": "model_longitude",
            "time": "model_time",
        }
        if has_depth:
            rename_map["depth"] = "model_depth"
        point_data = point_data.rename(rename_map)

        point_latitude = point[column_names["latitude"]]
        point_longitude = point[column_names["longitude"]]
        point_time = point[column_names["time"]]

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
    Assembles the per-section results into a single Dask DataFrame.

    Args:
        results: A sequence of `Delayed` objects, as returned by
            `_extract_points`. Each one wraps, once computed, the list
            of dictionaries with the results for the points of one
            section; each such list becomes one partition of the
            output DataFrame.
        indices: For each element of `results`, the Pandas Index to
            use for the rows of the corresponding partition (i.e. the
            original index of the points in that section), so that the
            output DataFrame's index matches `point_table`'s.
        dask_f_meta: An empty Pandas DataFrame describing the columns
            (and their dtypes) of the output. If it is `None`, dask
            infers the columns and dtypes from the structure of the
            first delayed object.

    Returns:
        A Dask DataFrame with one partition per element of `results`,
        indexed according to `indices`.
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


def _get_lat_lon_chunks(
    dataset: "xr.Dataset", var3d: Sequence[str]
) -> tuple["xr.Dataset", tuple[int, ...], tuple[int, ...]]:
    """
    Returns the chunks of `dataset` along the `latitude` and
    `longitude` dimensions.

    If the dataset does not have homogeneous chunks (e.g. because 2D
    and 3D variables are chunked differently), it is unified first;
    the (possibly unified) dataset is returned together with the
    chunks, since the caller needs to keep using the same dataset that
    these chunks refer to.

    Args:
        dataset: The dataset whose spatial chunks are inspected.
        var3d: The names of the 3D (depth-dependent) variables of
            `dataset`. When there are any, the chunks are read from
            these variables only, since 2D and 3D variables may be
            chunked differently and it is the 3D ones that determine
            how the bottom cell can be sliced.

    Returns:
        A 3-tuple `(dataset, lat_chunks, lon_chunks)`, where `dataset`
        is the input dataset (unified, if that was needed), and
        `lat_chunks`/`lon_chunks` are the sizes of its chunks along
        the `latitude` and `longitude` dimensions.
    """
    if len(var3d) > 0:
        chunking_dataset = dataset[[*var3d]]
    else:
        chunking_dataset = dataset

    try:
        dataset_chunks = chunking_dataset.chunks
    except ValueError as e:
        LOGGER.debug(
            "Can not read the chunks of the dataset; probably it "
            "is not an homogeneous dataset; the error message was: "
            '"%s"',
            str(e),
        )
        dataset = dataset.unify_chunks()
        dataset_chunks = dataset.chunks

    if dataset_chunks is not None:
        return (
            dataset,
            dataset_chunks["latitude"],
            dataset_chunks["longitude"],
        )
    return (
        dataset,
        (dataset.sizes["latitude"],),
        (dataset.sizes["longitude"],),
    )


@dataclass(frozen=True, eq=False, slots=True)
class _Section:
    """
    A group of points that share the same spatial chunk and time
    window of the model dataset, as produced by
    `BottomCellMap._split_into_chunks`.

    Attributes:
        lat_slice: The slice, along `latitude`, that selects this
            section's spatial chunk from the dataset.
        lon_slice: The slice, along `longitude`, that selects this
            section's spatial chunk from the dataset.
        time_window: The start (inclusive) and end (exclusive) of
            this section's time window.
        positions: The positional indices (as used by
            `DataFrame.iloc`) of the points, among the arrays passed
            to `_split_into_chunks`, that belong to this section.
    """

    lat_slice: slice
    lon_slice: slice
    time_window: tuple[np.datetime64, np.datetime64]
    positions: np.ndarray


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
        data_collection (GeoDataCollection): The dataset containing the model data.
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
        data_collection: GeoDataCollection,
        time_range: timedelta,
        lat_column: str = "latitude",
        lon_column: str = "longitude",
        time_column: str = "time",
    ):
        """
        Initializes the mapping between points and the model's bottom
        cell, using `dataset` as the source of model data.

        See the class docstring for a description of the arguments.
        """
        LOGGER.debug("Initializing a new %s instance", self.__class__.__name__)
        self._data_collection = data_collection
        self._time_range = time_range

        self._lat_column = lat_column
        self._lon_column = lon_column
        self._time_column = time_column

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
        those columns. The Dataset passed to `func` will contain one
        variable for each variable in the original dataset, indexed by one
        dimension: `model_time`. The `model_time` dimension has an associated
        coordinate that contains all the time steps of the model in the
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
          bottom cell of the model grid at the point's location. This
          coordinate (and the `depth` dimension it comes from) is only
          present when the underlying dataset has 3D variables; if the
          dataset only contains 2D variables, `model_depth` is omitted.
        - distance: the geodesic distance between the original point and
          the model point corresponding to the bottom cell of the model grid
          at the point's location (in meters).

        The `func` must return a dictionary whose keys can be chosen freely
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
            func_meta: Here you can specify the dtypes of the output of func.
                For example, if func returns two columns named A and B and the
                values of the column A are integers while the values of B are
                floating point numbers, func_meta must be
                {"A": int, "B": np.float32}. If it is not submitted, the code
                will try to guess an appropriate meta by executing func on the
                first point.
            delayed: If True, the method returns a Dask Delayed object that can
                be computed later. If False, the method computes the results
                immediately and returns a Pandas DataFrame.

        Returns:
            If `delayed` is `False` (the default), a Pandas DataFrame
            with one row per point of `point_table`, containing all
            the columns of `point_table` plus the columns returned by
            `func`. If `delayed` is `True`, a Dask DataFrame with the
            same columns, not computed yet.
            In both cases, the rows are indexed like `point_table`, but
            may not be in the same order: if `point_table.index` has no
            duplicates, it can still be used to match each output row
            back to the corresponding input point.
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
            "Reading the mask of the GeoDataCollection to find the bottom "
            "cell indices"
        )
        xarray_mask = self._data_collection.get_mask()
        geo_data_2d = "depth" not in xarray_mask.dims
        mask = self._build_bitsea_mask(xarray_mask, geo_data_2d)

        original_columns = point_table.columns.tolist()
        self._add_model_grid_columns(point_table, mask)

        lat_indices = point_table["model_lat_index"].values.astype(int)
        lon_indices = point_table["model_lon_index"].values.astype(int)

        if geo_data_2d:
            bottom_indices = None
        else:
            bottom_indices = self._compute_bottom_indices(
                mask, lat_indices, lon_indices
            )

        data, lat_chunks, lon_chunks = self._load_rechunked_dataset(
            has_depth=bottom_indices is not None
        )

        # Now we need to split the points into several arrays, one for each
        # zone of the model grid
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

        dask_f_meta = self._resolve_func_meta(
            func_meta=func_meta,
            func=func,
            point_table=point_table,
            data=data,
            column_names=column_names,
            bottom_indices=bottom_indices,
        )

        delayed_computations = self._build_delayed_computations(
            func=func,
            point_table=point_table,
            data=data,
            sections=sections,
            column_names=column_names,
            original_columns=original_columns,
            bottom_indices=bottom_indices,
        )

        LOGGER.debug("Merging together all the points")
        final_output = _merge_together(
            delayed_computations,
            indices=[point_table.index[s.positions] for s in sections],
            dask_f_meta=dask_f_meta,
        )

        if not delayed:
            LOGGER.debug("Starting the overall computation...")
            final_output = final_output.compute()

        LOGGER.debug(
            "Computation of %s.map completed!", self.__class__.__name__
        )
        return final_output

    def _build_bitsea_mask(
        self, xarray_mask: "xr.Dataset", geo_data_2d: bool
    ) -> Mask:
        """
        Builds the bit.sea `Mask` of the model grid from the xarray
        mask of the dataset.

        `bitsea.commons.mask.Mask` always expects a `depth` dimension;
        when the underlying `GeoDataCollection` is 2D (`geo_data_2d`
        is `True`), a fake `depth` dimension with a single level is
        added first.

        Args:
            xarray_mask: The mask of the dataset, as returned by
                `self._data_collection.get_mask()`.
            geo_data_2d: Whether the underlying `GeoDataCollection` is
                2D (has no `depth` dimension).

        Returns:
            The bit.sea `Mask` describing the model grid.
        """
        if geo_data_2d:
            LOGGER.debug("The GeoDataCollection is 2D (no depth found)")
            # We add a fake "depth" coordinate with just value 1
            xarray_mask = xarray_mask.expand_dims(dim={"depth": [1.0]})
        else:
            LOGGER.debug("The GeoDataCollection is 3D (depth found)")

        mask = Mask.from_xarray(xarray_mask)
        LOGGER.debug("bit.sea mask for the domain has been computed")
        return mask

    def _add_model_grid_columns(
        self, point_table: pd.DataFrame, mask: Mask
    ) -> None:
        """
        Adds, in place, the columns describing each point's associated
        model grid point to `point_table`.

        For every row, the point's `(lat_column, lon_column)` position
        is matched to the nearest wet point of the model grid. The
        following columns are added:
            - `model_lon_index`, `model_lat_index`: the indices, along
              the `longitude` and `latitude` dimensions of the model
              grid, of the matched model point.
            - `model_lat`, `model_lon`: the latitude and longitude of
              the matched model point.
            - `distance_from_model`: the geodesic distance, in meters,
              between the point and the matched model point.

        Args:
            point_table: The point table to augment. Modified in
                place.
            mask: The bit.sea `Mask` of the model grid.
        """

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

    def _compute_bottom_indices(
        self, mask: Mask, lat_indices: np.ndarray, lon_indices: np.ndarray
    ) -> "xr.DataArray":
        """
        Computes, for each point, the absolute index of the model's
        bottom cell along the `depth` dimension.

        Args:
            mask: The bit.sea `Mask` of the model grid.
            lat_indices: For each point, the index (along `latitude`)
                of its associated model point.
            lon_indices: For each point, the index (along `longitude`)
                of its associated model point.

        Returns:
            A `points`-indexed DataArray named `model_depth_index`
            with, for each point, the index of the deepest wet cell of
            the model grid at that point's location.
        """
        LOGGER.debug(
            "Computing the bottom cell indices for the points in the point "
            "table"
        )
        bottom_index_map = mask.bathymetry_in_cells() - 1

        LOGGER.debug(
            "Extracting the bottom cell indices for all the %s points "
            "in the point table",
            len(lat_indices),
        )
        return xr.DataArray(
            bottom_index_map[lat_indices, lon_indices],
            dims="points",
            name="model_depth_index",
        )

    def _load_rechunked_dataset(
        self, has_depth: bool
    ) -> tuple["xr.Dataset", tuple[int, ...], tuple[int, ...]]:
        """
        Loads the model data and ensures it is chunked into reasonably
        small spatial blocks.

        The dataset is opened with `depth` as a single chunk (when
        `has_depth` is `True`); if its native spatial chunks (along
        `latitude`/`longitude`) are larger than
        `MAX_REASONABLE_SPATIAL_CHUNK_SIZE` points, the dataset is
        rechunked into blocks of `TARGET_SPATIAL_CHUNK_SIZE` points,
        so that `_split_into_chunks` can later group points into
        small, spatially localized sections instead of a handful of
        sections that span almost the whole domain.

        Args:
            has_depth: Whether the underlying `GeoDataCollection` is
                3D (has a `depth` dimension).

        Returns:
            A 3-tuple `(data, lat_chunks, lon_chunks)`: the (possibly
            rechunked) dataset, and the sizes of its chunks along
            `latitude` and `longitude`.
        """
        n_time_steps = self._data_collection.get_n_of_time_steps()
        LOGGER.debug(
            "The dataset has approximately %s time steps", n_time_steps
        )

        chunks: dict[str, str | int] = {"depth": -1} if has_depth else {}
        LOGGER.debug(
            "Reading the dataset to extract the data using the following "
            "chunks: %s",
            chunks,
        )
        data = self._data_collection.get_data(chunks=chunks)

        LOGGER.debug(
            "Dataset opened! It has the following dimensions: %s",
            dict(data.sizes),
        )

        var3d = [
            var_name
            for var_name in data.data_vars
            if "depth" in data[var_name].dims
        ]
        LOGGER.debug(
            "The following variables have a depth dimension: %s", var3d
        )

        data, lat_chunks, lon_chunks = _get_lat_lon_chunks(data, var3d)
        LOGGER.debug(
            "The dataset has the following native spatial chunks: "
            "latitude=%s, longitude=%s",
            lat_chunks,
            lon_chunks,
        )

        if (
            max(lat_chunks) > MAX_REASONABLE_SPATIAL_CHUNK_SIZE
            or max(lon_chunks) > MAX_REASONABLE_SPATIAL_CHUNK_SIZE
        ):
            LOGGER.debug(
                "The native spatial chunks are larger than %s points "
                "(latitude=%s, longitude=%s); rechunking latitude and "
                "longitude into blocks of %s points",
                MAX_REASONABLE_SPATIAL_CHUNK_SIZE,
                max(lat_chunks),
                max(lon_chunks),
                TARGET_SPATIAL_CHUNK_SIZE,
            )
            data = data.chunk(
                {
                    "latitude": TARGET_SPATIAL_CHUNK_SIZE,
                    "longitude": TARGET_SPATIAL_CHUNK_SIZE,
                }
            )
            data, lat_chunks, lon_chunks = _get_lat_lon_chunks(data, var3d)
        else:
            LOGGER.debug(
                "The native spatial chunks are already reasonably small; "
                "keeping them as they are"
            )

        return data, lat_chunks, lon_chunks

    def _split_into_chunks(
        self,
        lat_indices: xr.DataArray,
        lon_indices: xr.DataArray,
        lat_chunks: Sequence[int],
        lon_chunks: Sequence[int],
        times: Sequence[np.datetime64],
    ) -> tuple[_Section, ...]:
        """
        Groups points into "sections", where each section corresponds
        to a single spatial chunk (a latitude chunk combined with a
        longitude chunk) and a single time window of length
        `self._time_range`, and contains only the points that fall
        inside that chunk and window.

        This grouping allows `map` to submit one dask task per section
        (via `_extract_points`) instead of one task per point, with
        each task reading a Dataset chunk that is small and localized
        both in space (one spatial chunk, instead of a handful of
        chunks spanning almost the whole domain) and in time (one time
        window, instead of the whole duration of the dataset).

        `lat_chunks` and `lon_chunks` describe how the dataset is
        chunked along the `latitude` and `longitude` dimensions (e.g.
        the dask chunk sizes, in the order they appear in the grid);
        this method turns them into chunk boundaries and determines,
        for each point, which latitude chunk and which longitude chunk
        its `lat_indices`/`lon_indices` fall into. Points are further
        split along time into consecutive, non-overlapping windows of
        length `self._time_range`, starting at
        `self._data_collection.start_date` and ending at
        `self._data_collection.end_date`.

        Args:
            lat_indices: For each point, the index (along the
                `latitude` dimension of the model grid) of its
                associated model point.
            lon_indices: For each point, the index (along the
                `longitude` dimension of the model grid) of its
                associated model point.
            lat_chunks: The sizes of the dataset chunks along the
                `latitude` dimension, in the order they appear in the
                grid (e.g. `(100, 100, 42)`).
            lon_chunks: The sizes of the dataset chunks along the
                `longitude` dimension, in the order they appear in the
                grid.
            times: For each point, its associated time value.

        Returns:
            A tuple of `_Section` objects, one per non-empty
            combination of spatial chunk and time window, sorted by
            decreasing number of points (largest section first). See
            `_Section` for the meaning of each attribute; its
            `positions` are positions among `lat_indices`,
            `lon_indices` and `times`. Every point appears in exactly
            one section.
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

        start_date = np.datetime64(self._data_collection.start_date, "s")
        time_range = np.timedelta64(self._time_range, "s")
        times = np.asarray(times)

        sections = []
        for lat_chunk_index in set(lat_chunk_indices):
            # Now we compute a slice that, if applied on the latitude array of
            # the dataset, will return the latitude positions of the cells
            # of the current section
            if len(lat_splits) == 0:
                lat_slice = slice(None, None)
            elif lat_chunk_index == 0:
                lat_slice = slice(None, lat_splits[lat_chunk_index])
            elif lat_chunk_index == len(lat_splits):
                lat_slice = slice(lat_splits[lat_chunk_index - 1], None)
            else:
                lat_slice = slice(
                    lat_splits[lat_chunk_index - 1],
                    lat_splits[lat_chunk_index],
                )

            for lon_chunk_index in set(lon_chunk_indices):
                if len(lon_splits) == 0:
                    lon_slice = slice(None, None)
                elif lon_chunk_index == 0:
                    lon_slice = slice(None, lon_splits[lon_chunk_index])
                elif lon_chunk_index == len(lon_splits):
                    lon_slice = slice(lon_splits[lon_chunk_index - 1], None)
                else:
                    lon_slice = slice(
                        lon_splits[lon_chunk_index - 1],
                        lon_splits[lon_chunk_index],
                    )

                # Which points are in the current portion of the domain?
                positions = np.nonzero(
                    np.logical_and(
                        lat_chunk_indices == lat_chunk_index,
                        lon_chunk_indices == lon_chunk_index,
                    )
                )[0]

                # This section is empty, we can move on
                if positions.size == 0:
                    continue

                local_lon_indices = lon_indices[positions]
                local_lat_indices = lat_indices[positions]

                # We apply some assetion to check that all the points that we
                # have selected have indices that are inside our portion
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
                while current_time <= self._data_collection.end_date:
                    inside_time_frame = np.logical_and(
                        times[positions] >= current_time,
                        times[positions] < current_time + time_range,
                    )
                    time_window = (current_time, current_time + time_range)
                    current_time += time_range

                    if not np.any(inside_time_frame):
                        continue

                    sections.append(
                        _Section(
                            lat_slice=lat_slice,
                            lon_slice=lon_slice,
                            time_window=time_window,
                            positions=positions[inside_time_frame],
                        )
                    )

        # Sort by length of sections (largest first)
        sections.sort(key=lambda s: -len(s.positions))

        assert sum(len(s.positions) for s in sections) == len(lat_indices), (
            "Some indices are missing after splitting into chunks."
        )

        return tuple(sections)

    def _build_extract_points_kwargs(
        self,
        point_subset: pd.DataFrame,
        indices_shift: dict[Literal["latitude", "longitude", "depth"], int],
        column_names: dict[Literal["time", "latitude", "longitude"], str],
        preserve_columns: list[str],
        bottom_indices: "xr.DataArray | None",
        bottom_positions: slice | np.ndarray,
    ) -> dict[str, Any]:
        """
        Builds the keyword arguments shared by every call to
        `_extract_points`, except for `f` and `dataset`.

        Args:
            point_subset: The rows of the point table handled by this
                call.
            indices_shift: The offset, along each of `latitude`,
                `longitude` and (when `bottom_indices` is not `None`)
                `depth`, between the absolute model grid indices and
                the indices of the dataset chunk that will be passed
                to `_extract_points`.
            column_names: Maps the logical roles `"time"`,
                `"latitude"` and `"longitude"` to the actual column
                names used in `point_table`.
            preserve_columns: Names of the columns of `point_table`
                that must be copied, unchanged, into the output.
            bottom_indices: The absolute bottom cell index, along
                `depth`, for every point of the point table, or `None`
                when the underlying `GeoDataCollection` is 2D.
            bottom_positions: The positions of `point_subset`'s rows
                within `bottom_indices` (as used to index it). Ignored
                when `bottom_indices` is `None`.

        Returns:
            A dictionary with every keyword argument of
            `_extract_points` except `f` and `dataset`.
        """
        has_depth = bottom_indices is not None
        return dict(
            point_table=point_subset,
            depth_indices=(
                bottom_indices[bottom_positions] if has_depth else None
            ),
            indices_shift=indices_shift,
            column_names=column_names,
            time_range=self._time_range,
            preserve_columns=preserve_columns,
            has_depth=has_depth,
        )

    def _resolve_func_meta(
        self,
        func_meta: dict[str, np.typing.DTypeLike] | None,
        func: Callable[["xr.Dataset"], dict[str, Any]],
        point_table: pd.DataFrame,
        data: "xr.Dataset",
        column_names: dict[Literal["time", "latitude", "longitude"], str],
        bottom_indices: "xr.DataArray | None",
    ) -> pd.DataFrame:
        """
        Resolves the dask meta describing the columns that `map` will
        return.

        If `func_meta` is given, it is translated into an empty Pandas
        DataFrame via `dask.dataframe.dispatch.make_meta`. Otherwise,
        the meta is inferred by actually calling `func` on the first
        point of `point_table` (see `_guess_func_meta`).

        Args:
            func_meta: The user-provided dtypes for the columns
                returned by `func`, or `None` to infer them.
            func: The function that will be mapped over the points.
            point_table: The point table, already augmented with the
                `model_lat_index`/`model_lon_index` columns.
            data: The (rechunked) dataset the data will be read from.
            column_names: Maps the logical roles `"time"`,
                `"latitude"` and `"longitude"` to the actual column
                names used in `point_table`.
            bottom_indices: The absolute bottom cell index, along
                `depth`, for every point of `point_table`, or `None`
                when the underlying `GeoDataCollection` is 2D.

        Returns:
            An empty Pandas DataFrame with the columns (and dtypes)
            that the output of `map` will have.
        """
        if func_meta is not None:
            LOGGER.debug("func_meta is: %s", func_meta)
            dask_f_meta = make_meta(func_meta)
            LOGGER.debug("func_meta has been translated as: %s", dask_f_meta)
            return dask_f_meta

        return self._guess_func_meta(
            func, point_table, data, column_names, bottom_indices
        )

    def _guess_func_meta(
        self,
        func: Callable[["xr.Dataset"], dict[str, Any]],
        point_table: pd.DataFrame,
        data: "xr.Dataset",
        column_names: dict[Literal["time", "latitude", "longitude"], str],
        bottom_indices: "xr.DataArray | None",
    ) -> pd.DataFrame:
        """
        Infers the dask meta of `map`'s output by calling `func` on
        the first point of `point_table`.

        Args:
            func: The function that will be mapped over the points.
            point_table: The point table, already augmented with the
                `model_lat_index`/`model_lon_index` columns.
            data: The (rechunked) dataset the data will be read from.
            column_names: Maps the logical roles `"time"`,
                `"latitude"` and `"longitude"` to the actual column
                names used in `point_table`.
            bottom_indices: The absolute bottom cell index, along
                `depth`, for every point of `point_table`, or `None`
                when the underlying `GeoDataCollection` is 2D.

        Returns:
            An empty Pandas DataFrame with the columns and dtypes
            returned by `func` for the first point.
        """
        LOGGER.debug("We use the first point to guess the func_meta")
        test_point = point_table.iloc[0]
        time_slice = slice(
            test_point[column_names["time"]] - self._time_range,
            test_point[column_names["time"]] + timedelta(milliseconds=1),
        )
        time_index_slice = data.indexes["time"].slice_indexer(
            time_slice.start, time_slice.stop
        )
        lat_index = test_point["model_lat_index"]
        lon_index = test_point["model_lon_index"]
        test_selection = dict(
            time=time_index_slice,
            latitude=slice(lat_index, lat_index + 1),
            longitude=slice(lon_index, lon_index + 1),
        )
        local_shifts = {
            "latitude": lat_index,
            "longitude": lon_index,
        }
        if bottom_indices is not None:
            bottom_index = bottom_indices[0].item()
            test_selection["depth"] = slice(bottom_index, bottom_index + 1)
            local_shifts["depth"] = bottom_index
        test_dataset = data.isel(**test_selection)

        extract_kwargs = self._build_extract_points_kwargs(
            point_subset=point_table.iloc[[0]],
            indices_shift=local_shifts,
            column_names=column_names,
            preserve_columns=[],
            bottom_indices=bottom_indices,
            bottom_positions=slice(0, 1),
        )
        meta_task = _extract_points(
            func, dataset=to_delayed(test_dataset), **extract_kwargs
        ).compute()

        LOGGER.debug("func returned the following output: %s", meta_task)
        dask_f_meta = pd.DataFrame(meta_task, index=[0]).head(0)
        LOGGER.debug(
            "This is the inferred value of func_meta: %s", dask_f_meta
        )
        return dask_f_meta

    def _build_delayed_computations(
        self,
        func: Callable[["xr.Dataset"], dict[str, Any]],
        point_table: pd.DataFrame,
        data: "xr.Dataset",
        sections: tuple[_Section, ...],
        column_names: dict[Literal["time", "latitude", "longitude"], str],
        original_columns: list[str],
        bottom_indices: "xr.DataArray | None",
    ) -> list[Delayed]:
        """
        Builds one delayed `_extract_points` task per section.

        For each section returned by `_split_into_chunks`, this method
        slices `data` down to that section's spatial chunk and time
        window (widened by `self._time_range` at the start, since
        every point may need model data from before its own time
        window) and schedules a `_extract_points` task on it.

        Args:
            func: The function to map over the points.
            point_table: The point table, already augmented with the
                `model_lat_index`/`model_lon_index` columns.
            data: The (rechunked) dataset the data will be read from.
            sections: The sections returned by `_split_into_chunks`.
            column_names: Maps the logical roles `"time"`,
                `"latitude"` and `"longitude"` to the actual column
                names used in `point_table`.
            original_columns: The columns of the original
                `point_table`, to be preserved in the output.
            bottom_indices: The absolute bottom cell index, along
                `depth`, for every point of `point_table`, or `None`
                when the underlying `GeoDataCollection` is 2D.

        Returns:
            A list with one `Delayed` object per section, in the same
            order as `sections`.
        """
        delayed_computations = []
        LOGGER.debug(
            "Generating the dask graph of all the %s tasks that will be "
            "executed",
            len(point_table),
        )
        for section in sections:
            lat_shift = (
                section.lat_slice.start
                if section.lat_slice.start is not None
                else 0
            )
            lon_shift = (
                section.lon_slice.start
                if section.lon_slice.start is not None
                else 0
            )
            time_slice = data.indexes["time"].slice_indexer(
                section.time_window[0] - np.timedelta64(self._time_range, "s"),
                section.time_window[1],
            )
            LOGGER.debug(
                "Slicing a section in the time window %s", section.time_window
            )
            isel_selection = dict(
                time=time_slice,
                latitude=section.lat_slice,
                longitude=section.lon_slice,
            )
            local_shifts = {
                "latitude": lat_shift,
                "longitude": lon_shift,
            }
            if bottom_indices is not None:
                min_bottom_index = (
                    bottom_indices[section.positions].min().item()
                )
                max_bottom_index = (
                    bottom_indices[section.positions].max().item()
                )
                bottom_slice = slice(min_bottom_index, max_bottom_index + 1)
                isel_selection["depth"] = bottom_slice
                local_shifts["depth"] = bottom_slice.start
            LOGGER.debug(
                "Slicing the dataset in the following area: %s",
                isel_selection,
            )
            point_ds = data.isel(**isel_selection)

            extract_points_kwargs = self._build_extract_points_kwargs(
                point_subset=point_table.iloc[section.positions],
                indices_shift=local_shifts,
                column_names=column_names,
                preserve_columns=original_columns,
                bottom_indices=bottom_indices,
                bottom_positions=section.positions,
            )
            delayed_task = _extract_points(
                func, dataset=to_delayed(point_ds), **extract_points_kwargs
            )

            delayed_computations.append(delayed_task)

        return delayed_computations
