from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from medunda.actions.extract_extremes import extract_min_max

from conftest import generate_test_array


def test_extract_extremes(tmp_path):
    output_file = Path(tmp_path) / "min_max_values.csv"

    dates_str = [f"2024-{i:0>2}-01" for i in range(1, 13)]
    dates_str.extend([f"2025-{i:0>2}-01" for i in range(1, 13)])
    times = np.array(dates_str, dtype="datetime64[s]")

    data4d = generate_test_array(time=times)
    depth_levels = data4d.depth.shape[0]

    test_depth = float(data4d.depth.values[1])

    for t in range(len(dates_str)):
        for d in range(depth_levels):
            data4d.T.isel(time=t, depth=d)[:] = t + d
            data4d.S.isel(time=t, depth=d)[:] = 10 - t - 2 * d

    extract_min_max(data=data4d, output_file=output_file)

    assert output_file.exists(), "Output file was not created."

    output_content = pd.read_csv(output_file)
    assert len(output_content) == 2
