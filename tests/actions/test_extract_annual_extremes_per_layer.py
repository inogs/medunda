import numpy as np
import pandas as pd
from fixtures import generate_test_array

from medunda.actions.extract_annual_extremes_per_layer import (
    extract_annual_extremes_per_layer,
)


def test_extract_extremes(data4d):
    dates_str = [f"2024-{i:0>2}-01" for i in range(1, 13)]
    dates_str.extend([f"2025-{i:0>2}-01" for i in range(1, 13)])
    times = np.array(dates_str, dtype="datetime64[s]")

    data4d = generate_test_array(time=times)
    depth_levels = data4d.depth.shape[0]

    for t in range(len(dates_str)):
        for d in range(depth_levels):
            data4d.thetao.isel(time=t, depth=d)[:] = t + d
            data4d.so.isel(time=t, depth=d)[:] = 10 - t - 2 * d

    ds = extract_annual_extremes_per_layer(data=data4d)

    assert len(ds.data_vars) == 4
    assert "time" in ds.dims
    assert "year" not in ds.dims
    assert ds.sizes["time"] == 2
    assert np.issubdtype(ds.time.dtype, np.datetime64)
    pd.testing.assert_index_equal(
        ds.indexes["time"],
        pd.DatetimeIndex(["2024-01-01", "2025-01-01"], name="time"),
    )
    assert set(ds.data_vars) == {
        "thetao_min",
        "thetao_max",
        "so_min",
        "so_max",
    }
    np.testing.assert_allclose(ds["thetao_min"].isel(depth=0).values, [0, 12])
    np.testing.assert_allclose(ds["thetao_min"].isel(depth=-1).values, [2, 14])
    np.testing.assert_allclose(ds["thetao_max"].isel(depth=0).values, [11, 23])
    np.testing.assert_allclose(
        ds["thetao_max"].isel(depth=-1).values, [13, 25]
    )
    np.testing.assert_allclose(ds["so_min"].isel(depth=0).values, [-1, -13])
    np.testing.assert_allclose(ds["so_min"].isel(depth=-1).values, [-5, -17])
    np.testing.assert_allclose(ds["so_max"].isel(depth=0).values, [10, -2])
    np.testing.assert_allclose(ds["so_max"].isel(depth=-1).values, [6, -6])
