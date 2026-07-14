import numpy as np

from medunda.actions.compute_integral import integrate_between_layers


def test_integrate_between_layers(data4d):
    """Test the integrating between layers function."""

    depth_levels = data4d.depth.shape[0]
    for d in range(depth_levels):
        data4d.thetao.isel(depth=d)[:] = 3 * d
        data4d.so.isel(depth=d)[:] = 10 + 3 * d

    time_levels = data4d.time.shape[0]
    latitude_levels = data4d.latitude.shape[0]
    longitude_levels = data4d.longitude.shape[0]

    ds = integrate_between_layers(
        data=data4d,
        depth_min=0.0,
        depth_max=3.0,
    )

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
        "Difference between max and min of 'S' variable is incorrect."
    )
    assert thetao_range < 1e-6, (
        "Difference between max and min of 'S' variable is incorrect."
    )
    thetao_value = ds.thetao.max()
    so_value = ds.so.max()

    assert np.abs(thetao_value - 6.0) < 1e-6, (
        "Maximum value of 'T' variable is incorrect."
    )
    assert np.abs(so_value - 36.0) < 1e-6, (
        "Maximum value of 'S' variable is incorrect."
    )
