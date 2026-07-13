from itertools import product as cart_prod

import numpy as np
import xarray as xr
from pytest import fixture


def generate_test_array(depth=None, latitude=None, longitude=None, time=None):
    if depth is None:
        depth = np.array([0.5, 2.0, 5.0], dtype=np.float32)

    if latitude is None:
        latitude = np.array([42, 43, 44, 45], dtype=np.float32)

    if longitude is None:
        longitude = np.array([30, 30.5, 31, 31.5, 32], dtype=np.float32)

    if time is None:
        time = np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[s]")

    var1 = np.empty(
        (len(time), len(depth), len(latitude), len(longitude)),
        dtype=np.float32,
    )
    var2 = np.empty(
        (len(time), len(depth), len(latitude), len(longitude)),
        dtype=np.float32,
    )

    return xr.Dataset(
        {
            "thetao": (("time", "depth", "latitude", "longitude"), var1),
            "so": (("time", "depth", "latitude", "longitude"), var2),
        },
        coords={
            "depth": depth,
            "latitude": latitude,
            "longitude": longitude,
            "time": time,
        },
    )


def generate_test_array_with_mask():
    data = generate_test_array()
    mask = np.zeros_like(data.thetao.isel(time=0, drop=True), dtype=bool)
    n_lat = data.sizes["latitude"]
    n_lon = data.sizes["longitude"]
    n_depth = data.sizes["depth"]

    for i, j in cart_prod(range(n_lat), range(n_lon)):
        k = i * n_lon + j
        if k % 5 == 0:
            mask[:, i, j] = True
        if k % 4 == 0:
            mask[1:, i, j] = True
        if k % 3 == 0:
            mask[2:, i, j] = True

        for v_name in data.data_vars:
            if v_name in ["depth", "latitude", "longitude", "time"]:
                continue
            for t, d in cart_prod(range(data.sizes["time"]), range(n_depth)):
                data[v_name][
                    dict(time=t, depth=d, latitude=i, longitude=j)
                ] = k + 1000 * (t + 1) * d

    mask = xr.DataArray(mask, dims=["depth", "latitude", "longitude"])

    for v_name in data.data_vars:
        if v_name in ["depth", "latitude", "longitude", "time"]:
            continue
        data[v_name] = xr.where(mask, np.nan, data[v_name])

    data["mlotst"] = data["thetao"].isel(depth=0, drop=True).copy()
    return data


@fixture
def data4d():
    return generate_test_array()


@fixture
def dataset_4d():
    return generate_test_array_with_mask()


@fixture
def map_dataset():
    return generate_test_array_with_mask().isel(depth=0, drop=True)


@fixture
def dataset_at_a_specific_depth():
    return generate_test_array_with_mask().isel(depth=2, drop=False)


@fixture
def climatological_dataset():
    generate_test_array_with_mask().isel(time=0, drop=True)


@fixture(
    params=[
        "dataset_4d",
        "map_dataset",
        "dataset_at_a_specific_depth",
        "climatological_dataset",
    ],
    ids=[
        "dataset4d",
        "map_dataset",
        "dataset_at_a_specific_depth",
        "climatological_dataset",
    ],
)
def dataset(request):
    match request.param:
        case "dataset_4d":
            return generate_test_array_with_mask()
        case "map_dataset":
            return generate_test_array_with_mask().isel(depth=0, drop=True)
        case "dataset_at_a_specific_depth":
            return generate_test_array_with_mask().isel(depth=2, drop=False)
        case "climatological_dataset":
            return generate_test_array_with_mask().isel(time=0, drop=True)
        case _:
            raise ValueError(f"Unknown dataset type: {request.param}")


__all__ = [
    "data4d",
    "dataset_4d",
    "map_dataset",
    "dataset_at_a_specific_depth",
    "climatological_dataset",
    "dataset",
]
