from typing import Any
from typing import Literal

import numpy as np
import xarray as xr
from dask.delayed import Delayed

DelayedDataset = dict[Literal["data", "coords", "attrs"], Any]


def to_delayed(dataset: xr.Dataset) -> DelayedDataset:
    """
    Transform a Dataset into a dictionary of delayed objects.

    The dataset is expected to have data variables with data that is stored in
    a Dask Array. Moreover, the content of each Dask Array will be converted
    into a single delayed object; this implies that this function has been
    designed to work with small datasets that can be loaded into memory.
    """
    delayed_objects: dict[Literal["data", "coords", "attrs"], Any] = {
        "data": {},
        "coords": dataset.coords,
        "attrs": dataset.attrs,
    }
    for var_name in dataset.data_vars:
        var_dataarray = dataset[var_name]
        delayed_data = var_dataarray.data.rechunk(-1).to_delayed()
        while isinstance(delayed_data, np.ndarray):
            delayed_data = delayed_data[0]
        assert isinstance(delayed_data, Delayed), (
            f"Expected Delayed object, got {type(delayed_data)}"
        )

        delayed_objects["data"][var_name] = (
            var_dataarray.dims,
            var_dataarray.attrs,
            delayed_data,
        )
    return delayed_objects


def from_delayed(delayed_objects: DelayedDataset) -> xr.Dataset:
    """
    Transform a dictionary of delayed objects into a Dataset.

    This function is the opposite of `to_delayed` and transforms a dictionary
    into an xarray Dataset. This function is designed to be called inside a
    delayed function and, therefore, it expects that the delayed objects
    produced by the `to_delayed` function have been transformed into
    numpy arrays by Dask.
    """
    data_arrays = {}
    for var_name, (dims, attrs, data) in delayed_objects["data"].items():
        data_arrays[var_name] = xr.DataArray(data, dims=dims, attrs=attrs)
    return xr.Dataset(
        data_arrays,
        coords=delayed_objects["coords"],
        attrs=delayed_objects["attrs"],
    )
