from pathlib import Path

import xarray as xr
import numpy as np

from medunda.actions.calculate_stats import calculate_stats


def test_depth_average(data4d):
    """Test the extract_surface function."""
    
    depth_levels = data4d.depth.shape[0]
    latitude_levels = data4d.latitude.shape[0]
    longitude_levels = data4d.longitude.shape[0]

    operations = ["mean", "minimum", "maximum"]

    ds= calculate_stats(
        data=data4d,
        operations=operations
    )

    for var_name in ["T", "S"]:
        for op in operations:
            output_name = f"{var_name}_{op}"
            assert output_name in ds.data_vars, f"Output variable {output_name} not found in dataset."
            assert ds[output_name].shape == (
                depth_levels,
                latitude_levels,
                longitude_levels
            ), f"Shape mismatch for {output_name}. Expected {(depth_levels, latitude_levels, longitude_levels)}, got {ds[output_name].shape}."

