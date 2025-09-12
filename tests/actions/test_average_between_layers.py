import numpy as np

from medunda.actions.averaging_between_layers import averaging_between_layers


def test_averaging_between_layers(data4d):
    """Test the averaging between layers function."""
    
    depth_levels = data4d.depth.shape[0]
    for d in range(depth_levels):
        data4d.T.isel(depth=d)[:] = 3 * d
        data4d.S.isel(depth=d)[:] = 10 + 3 * d
    
    time_levels = data4d.time.shape[0]
    latitude_levels = data4d.latitude.shape[0]
    longitude_levels = data4d.longitude.shape[0]

    ds = averaging_between_layers(
        data=data4d,
        depth_min=0.,
        depth_max=3.,
    )

    assert "T" in ds.data_vars, "Variable 'T' not found in output dataset."
    assert "S" in ds.data_vars, "Variable 'S' not found in output dataset."
    assert ds.T.shape == (time_levels, latitude_levels, longitude_levels), \
        "Shape of 'T' variable is incorrect."
    assert ds.S.shape ==  (time_levels, latitude_levels, longitude_levels), \
        "Shape of 'S' variable is incorrect."

    S_range = ds.S.max() - ds.S.min()
    T_range = ds.T.max() - ds.T.min()
    assert S_range < 1e-6, "Difference between max and min of 'S' variable is incorrect."
    assert T_range < 1e-6, "Difference between max and min of 'S' variable is incorrect."
    T_value = ds.T.max()
    S_value = ds.S.max()
    assert np.abs(T_value - 2.0) < 1e-6, "Maximum value of 'T' variable is incorrect."
    assert np.abs(S_value - 12.0) < 1e-6, "Maximum value of 'S' variable is incorrect."
