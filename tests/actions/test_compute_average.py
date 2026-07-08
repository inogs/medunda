from contextlib import nullcontext
from itertools import product as cart_prod

import numpy as np
import pytest

import medunda.tools.lazy_imports.bitsea.mask as bitsea_mask
from medunda.actions.compute_average import compute_average
from medunda.tools.lazy_imports import xarray as xr


def _build_dataset_mask(dataset, dataset_id):
    if dataset_id == "dataset4d":
        tmask = xr.Dataset(
            {"tmask": np.isfinite(dataset.so.isel(time=0, drop=True))}
        )
    elif dataset_id == "climatological_dataset":
        tmask = xr.Dataset({"tmask": np.isfinite(dataset.so)})
    else:
        tmask_data = np.isfinite(dataset.so.isel(time=0).values)
        tmask_array = xr.DataArray(
            tmask_data[None, :, :], dims=("depth", "latitude", "longitude")
        )
        tmask = xr.Dataset(
            {"tmask": tmask_array},
            coords={
                "latitude": dataset.latitude,
                "longitude": dataset.longitude,
                "depth": np.array([1.0]),
            },
        )
    return bitsea_mask.Mask.from_xarray(tmask)


def test_compute_average_on_depth(dataset, request):
    """Test the extract_surface function."""
    dataset_id = request.node.callspec.id
    mask = _build_dataset_mask(dataset, dataset_id)

    depth_weights = [1.0, 2.0, 4.0]

    if "time" in dataset.sizes:
        n_time = dataset.time.shape[0]
    else:
        n_time = 1
    n_lat = dataset.latitude.shape[0]
    n_lon = dataset.longitude.shape[0]

    vars_with_depth = []
    maps = []
    for var_name in dataset.variables:
        if var_name in ["depth", "latitude", "longitude", "time"]:
            continue
        if "depth" in dataset[var_name].dims:
            vars_with_depth.append(var_name)
        else:
            maps.append(var_name)

    if dataset_id in ("map_dataset", "dataset_at_a_specific_depth"):
        ctx = pytest.warns(UserWarning)
    else:
        ctx = nullcontext()

    with ctx:
        ds = compute_average(
            data=dataset, axes=("depth",), depth_min=None, depth_max=None
        )

    # Check that all the vars with a depth have the same mask of the first
    # layer
    for var_name in vars_with_depth:
        axis = ["time", "latitude", "longitude"]
        if "time" not in dataset[var_name].dims:
            axis.remove("time")
        assert np.all(np.isfinite(ds[var_name]).transpose(*axis) == mask[0])

        # Now we check that the average is correct. We compare the average
        # computed by xarray with a very naive algorithm that we execute for
        # every time-step.
        for t in range(n_time):
            # selection is the index of the element that we are considering
            # on the final result; if we have a climatological_dataset, i.e.,
            # a dataset without the time dimension, we do not need to slice
            # on the time (and the loop will be executed just once since
            # n_time is 1).
            if dataset_id == "climatological_dataset":
                selection = {}
            else:
                selection = {"time": t}
            for i, j in cart_prod(range(n_lat), range(n_lon)):
                selection["latitude"] = i
                selection["longitude"] = j

                # t, i, and j define a column we can compute the average on
                expected_value = ds[var_name].isel(**selection).item()

                # Now we compute the average on the depth axis; we need another
                # dict called depth_selection to select also along the depth
                # axis. In current_sum we store the weighted sum of the
                # different elements
                current_sum = 0.0
                for k in range(0, dataset.sizes["depth"]):
                    depth_selection = selection.copy()
                    depth_selection["depth"] = k
                    current_value = (
                        dataset[var_name].isel(**depth_selection).item()
                    )

                    # If we find a NaN, we stop the loop and divide by
                    # the sum of the weights up to that point
                    if np.isnan(current_value):
                        if k == 0:
                            current_sum = np.nan
                        else:
                            current_sum /= sum(depth_weights[:k])
                        break

                    current_sum += current_value * depth_weights[k]
                else:
                    # If we reach the end of the loop, we have not found a NaN,
                    # so we divide by the sum of the weights
                    current_sum /= sum(depth_weights)

                np.testing.assert_almost_equal(current_sum, expected_value)

    # Check that the variables without the "depth" axis are the same
    for var_name in maps:
        np.testing.assert_equal(dataset[var_name].values, ds[var_name].values)
