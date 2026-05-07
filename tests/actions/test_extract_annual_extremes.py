import numpy as np
import pandas as pd
from fixtures import generate_test_array

from medunda.actions.extract_annual_extremes import extract_annual_extremes


def test_extract_extremes(data4d):
    dates_str = [f"2024-{i:0>2}-01" for i in range(1, 13)]
    dates_str.extend([f"2025-{i:0>2}-01" for i in range(1, 13)])
    times = np.array(dates_str, dtype="datetime64[s]")

    data4d = generate_test_array(time=times)
    depth_levels = data4d.depth.shape[0]

    for t in range(len(dates_str)):
        for d in range(depth_levels):
            data4d.T.isel(time=t, depth=d)[:] = t + d
            data4d.S.isel(time=t, depth=d)[:] = 10 - t - 2 * d

    ds = extract_annual_extremes(data=data4d)

    assert len(ds.data_vars) == 4
    assert "time" in ds.dims
    assert "year" not in ds.dims
    assert ds.sizes["time"] == 2
    assert np.issubdtype(ds.time.dtype, np.datetime64)
    pd.testing.assert_index_equal(
        ds.indexes["time"],
        pd.DatetimeIndex(["2024-01-01", "2025-01-01"], name="time"),
    )
    assert set(ds.data_vars) == {"T_min", "T_max", "S_min", "S_max"}
    np.testing.assert_allclose(ds["T_min"].values, [0, 12])
    np.testing.assert_allclose(ds["T_max"].values, [13, 25])
    np.testing.assert_allclose(ds["S_min"].values, [-5, -17])
    np.testing.assert_allclose(ds["S_max"].values, [10, -2])
