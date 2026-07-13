from medunda.actions.extract_layer import extract_layer


def test_extract_layer(data4d):
    """Test the extract_layer function with a valid depth level."""

    test_depth = float(data4d.depth.values[1])

    depth_levels = data4d.depth.shape[0]

    for d in range(depth_levels):
        data4d.thetao.isel(depth=d)[:] = d
        data4d.so.isel(depth=d)[:] = 10 + d

    time_levels = data4d.time.shape[0]
    latitude_levels = data4d.latitude.shape[0]
    longitude_levels = data4d.longitude.shape[0]

    ds = extract_layer(data=data4d, depth=test_depth)

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

    assert ds.thetao.min() == 1, (
        "Minimum value of 'thetao' variable is incorrect."
    )
    assert ds.so.min() == 11, "Minimum value of 'so' variable is incorrect."
