import numpy as np

from medunda.actions.extract_bottom import extract_bottom


def test_extract_bottom(data4d):
    """Test the extract_bottom function."""

    depth_levels = data4d.depth.shape[0]
    for d in range(depth_levels):
        data4d.thetao.isel(depth=d)[:] = d
        data4d.so.isel(depth=d)[:] = 10 + d

    last_thetao_value = depth_levels - 1
    last_so_value = 10 + depth_levels - 1

    time_levels = data4d.time.shape[0]
    latitude_levels = data4d.latitude.shape[0]
    longitude_levels = data4d.longitude.shape[0]

    ds = extract_bottom(data=data4d)

    assert "thetao" in ds.data_vars, (
        "Variable 'thetao' not found in output dataset."
    )
    assert "so" in ds.data_vars, "Variable 'so' not found in output dataset."
    assert ds.thetao.shape == (
        time_levels,
        latitude_levels,
        longitude_levels,
    ), "Shape of 'thetao' variable is incorrect."
    assert ds.so.shape == (time_levels, latitude_levels, longitude_levels), (
        "Shape of 'so' variable is incorrect."
    )

    so_range = ds.so.max() - ds.so.min()
    thetao_range = ds.thetao.max() - ds.thetao.min()
    assert so_range < 1e-6, (
        "Difference between max and min of 'so' variable is incorrect."
    )
    assert thetao_range < 1e-6, (
        "Difference between max and min of 'thetao' variable is incorrect."
    )
    thetao_value = ds.thetao.max()
    so_value = ds.so.max()
    assert np.abs(thetao_value - last_thetao_value) < 1e-6, (
        "Minimum value of 'thetao' variable is incorrect."
    )
    assert np.abs(so_value - last_so_value) < 1e-6, (
        "Minimum value of 'so' variable is incorrect."
    )
