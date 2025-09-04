from pathlib import Path

import xarray as xr

from medunda.actions.extract_surface import extract_surface


def test_extract_surface(data4d):
    """Test the extract_surface function."""
    
    depth_levels = data4d.depth.shape[0]
    for d in range(depth_levels):
        data4d.T.isel(depth=d)[:] = d
        data4d.S.isel(depth=d)[:] = 10 + d
    
    time_levels = data4d.time.shape[0]
    latitude_levels = data4d.latitude.shape[0]
    longitude_levels = data4d.longitude.shape[0]

    ds = extract_surface(data=data4d)

    assert "T" in ds.data_vars, "Variable 'T' not found in output dataset."
    assert "S" in ds.data_vars, "Variable 'S' not found in output dataset."
    assert ds.T.shape == (time_levels, latitude_levels, longitude_levels), \
        "Shape of 'T' variable is incorrect."
    assert ds.S.shape ==  (time_levels, latitude_levels, longitude_levels), \
        "Shape of 'S' variable is incorrect."
    assert ds.T.min() == 0, "Minimum value of 'T' variable is incorrect."
    assert ds.S.min() == 10, "Minimum value of 'S' variable is incorrect."
