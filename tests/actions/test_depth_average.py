import numpy as np

from medunda.actions.depth_average import compute_depth_average


def test_depth_average(data4d):
    """Test the extract_surface function."""

    depth_levels = data4d.depth.shape[0]
    for d in range(depth_levels):
        data4d.T.isel(depth=d)[:] = 7 * d
        data4d.S.isel(depth=d)[:] = 14 + 7 * d
    
    time_levels = data4d.time.shape[0]
    latitude_levels = data4d.latitude.shape[0]
    longitude_levels = data4d.longitude.shape[0]

    ds = compute_depth_average(data=data4d)

    
    assert "T" in ds.data_vars, "Variable 'T' not found in output dataset."
    assert "S" in ds.data_vars, "Variable 'S' not found in output dataset."
    assert ds.T.shape == (time_levels, latitude_levels, longitude_levels), \
        "Shape of 'T' variable is incorrect."
    assert ds.S.shape ==  (time_levels, latitude_levels, longitude_levels), \
        "Shape of 'S' variable is incorrect."

    S_range = ds.S.max() - ds.S.min()
    T_range = ds.T.max() - ds.T.min()
    assert S_range < 1e-6, "Difference between max and min of 'S' variable is incorrect."
    assert T_range < 1e-6, "Difference between max and min of 'T' variable is incorrect."
    T_value = ds.T.max()
    S_value = ds.S.max()
    assert np.abs(T_value - 10.0) < 1e-6, "Maximum value of 'T' variable is incorrect."
    assert np.abs(S_value - 24.0) < 1e-6, "Maximum value of 'S' variable is incorrect."
