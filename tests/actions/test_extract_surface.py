from medunda.actions.extract_surface import extract_surface


def test_extract_surface(data4d):
    """Test the extract_surface function."""

    depth_levels = data4d.depth.shape[0]
    for d in range(depth_levels):
        data4d.thetao.isel(depth=d)[:] = d
        data4d.so.isel(depth=d)[:] = 10 + d

    time_levels = data4d.time.shape[0]
    latitude_levels = data4d.latitude.shape[0]
    longitude_levels = data4d.longitude.shape[0]

    ds = extract_surface(data=data4d)

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
    assert ds.thetao.min() == 0, (
        "Minimum value of 'thetao' variable is incorrect."
    )
    assert ds.so.min() == 10, "Minimum value of 'so' variable is incorrect."
