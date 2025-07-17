from pathlib import Path

import xarray as xr

from medunda.actions.extract_surface import extract_surface


def test_extract_surface(data4d, tmp_path):
    """Test the extract_surface function."""
    tmp_path = Path(tmp_path)
    output_file = tmp_path / "surface_extraction.nc"
    depth_levels = data4d.depth.shape[0]
    for d in range(depth_levels):
        data4d.T.isel(depth=d)[:] = d
        data4d.S.isel(depth=d)[:] = 10 + d
    
    time_levels = data4d.time.shape[0]
    latitude_levels = data4d.latitude.shape[0]
    longitude_levels = data4d.longitude.shape[0]

    extract_surface(
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
        assert ds.T.min() == 0, "Minimum value of 'T' variable is incorrect."
        assert ds.S.min() == 10, "Minimum value of 'S' variable is incorrect."
