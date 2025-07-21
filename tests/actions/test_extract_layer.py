from pathlib import Path
import xarray as xr

from medunda.actions.extract_layer import extract_layer

def test_extract_layer(data4d, tmp_path):
    """Test the extract_layer function with a valid depth level."""

    output_file = Path(tmp_path) / "layer_extraction.nc"

    test_depth = float(data4d.depth.values[1])

    depth_levels = data4d.depth.shape[0]

    for d in range(depth_levels):
        data4d.T.isel(depth=d)[:] = d
        data4d.S.isel(depth=d)[:] = 10 + d

    time_levels = data4d.time.shape[0]
    latitude_levels = data4d.latitude.shape[0]
    longitude_levels = data4d.longitude.shape[0]

    extract_layer(data=data4d, output_file=output_file, depth=test_depth)

    assert output_file.exists(), "Output file was not created."

    with xr.open_dataset(output_file) as ds:

        assert "T" in ds.data_vars, "Variable 'T' not found in output dataset."
        assert "S" in ds.data_vars, "Variable 'S' not found in output dataset."

        assert ds.T.shape == (time_levels, latitude_levels, longitude_levels), \
            "Shape of 'T' variable is incorrect."
        assert ds.S.shape ==  (time_levels, latitude_levels, longitude_levels), \
            "Shape of 'S' variable is incorrect."

        assert ds.T.min() == 1, "Minimum value of 'T' variable is incorrect."
        assert ds.S.min() == 11, "Minimum value of 'S' variable is incorrect."