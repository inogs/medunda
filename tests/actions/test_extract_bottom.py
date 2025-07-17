from pathlib import Path

import numpy as np
import xarray as xr

from medunda.actions.extract_bottom import extract_bottom


def test_extract_surface(data4d, tmp_path):
    """Test the extract_surface function."""
    tmp_path = Path(tmp_path)
    output_file = tmp_path / "surface_extraction.nc"
    depth_levels = data4d.depth.shape[0]
    for d in range(depth_levels):
        data4d.T.isel(depth=d)[:] = d
        data4d.S.isel(depth=d)[:] = 10 + d
    
    last_T_value = depth_levels - 1
    last_S_value = 10 + depth_levels - 1

    time_levels = data4d.time.shape[0]
    latitude_levels = data4d.latitude.shape[0]
    longitude_levels = data4d.longitude.shape[0]

    extract_bottom(
        data=data4d,
        output_file=output_file
    )

    assert output_file.exists(), "Output file was not created."

    with xr.open_dataset(output_file) as ds:
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
        assert np.abs(T_value - last_T_value) < 1e-6, "Minimum value of 'T' variable is incorrect."
        assert np.abs(S_value - last_S_value) < 1e-6, "Minimum value of 'S' variable is incorrect."
