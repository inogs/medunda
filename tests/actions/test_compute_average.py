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


@pytest.mark.parametrize("average_also_on_time_axis", [True, False])
def test_compute_average_on_depth(
    dataset, request, average_also_on_time_axis: bool
):
    """
    Tests the computation of weighted averages along the depth axis for a
    given dataset.

    This function verifies the correctness of the `compute_average` function
    by performing two types of checks:
    1. Ensures that all variables containing the "depth" dimension have the
       same mask as the first layer of the dataset.
    2. Compares the results of the `compute_average` function with manually
       computed averages for each depth column using a simple algorithm that
       accounts for depth weights, handling `NaN` values appropriately.

    The variables without the "depth" dimension are also checked to ensure
    that their values remain unchanged during the computation.
    """
    dataset_id = request.node.callspec.id.split("-")[0]
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
        # Warning is raised because the depth axis is not present in the
        # dataset, and we are computing the average on it.
        ctx = pytest.warns(UserWarning)
    elif dataset_id == "climatological_dataset" and average_also_on_time_axis:
        # Warning is raised because the time axis is not present in the
        # dataset, and we are computing the average on it.
        ctx = pytest.warns(UserWarning)
    else:
        # No warning is expected
        ctx = nullcontext()

    if average_also_on_time_axis:
        average_axes = ("depth", "time")
    else:
        average_axes = ("depth",)

    with ctx:
        ds = compute_average(
            data=dataset, axes=average_axes, depth_min=None, depth_max=None
        )
    print(ds)

    # Check that all the vars with a depth have the same mask of the first
    # layer
    for var_name in vars_with_depth:
        axis = ["time", "latitude", "longitude"]
        if "time" not in dataset[var_name].dims or average_also_on_time_axis:
            axis.remove("time")
        assert np.all(np.isfinite(ds[var_name]).transpose(*axis) == mask[0])

        # Now we check that the average is correct. We compare the average
        # computed by xarray with a very naive algorithm that we execute for
        # every time-step.
        expected_values = xr.DataArray(
            data=np.zeros(
                (n_time, n_lat, n_lon), dtype=dataset[var_name].dtype
            ),
            dims=["time", "latitude", "longitude"],
        )
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

                expected_values.loc[dict(time=t, latitude=i, longitude=j)] = (
                    current_sum
                )

        if "time" in ds.dims:
            computed_values = ds[var_name].transpose(
                "time", "latitude", "longitude"
            )
        else:
            computed_values = ds[var_name].transpose("latitude", "longitude")

        if average_also_on_time_axis:
            expected_values = expected_values.mean(dim="time")
        elif "time" not in dataset[var_name].dims:
            expected_values = expected_values.isel(time=0, drop=True)

        np.testing.assert_array_almost_equal(
            computed_values, expected_values, decimal=4
        )

    # Check that the variables without the "depth" axis are the same
    if not average_also_on_time_axis or "time" not in dataset.dims:
        for var_name in maps:
            np.testing.assert_equal(
                dataset[var_name].values, ds[var_name].values
            )
    else:
        # If we average on the time axis, then the 2d maps we have compute must
        # be equal to the average in time of the original maps
        for var_name in maps:
            np.testing.assert_array_almost_equal(
                dataset[var_name].mean(dim="time").values, ds[var_name].values
            )
