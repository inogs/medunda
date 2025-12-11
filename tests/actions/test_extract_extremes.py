import numpy as np
from fixtures import generate_test_array

from medunda.actions.extract_extremes import extract_min_max


def test_extract_extremes(data4d):
    dates_str = [f"2024-{i:0>2}-01" for i in range(1, 13)]
    dates_str.extend([f"2025-{i:0>2}-01" for i in range(1, 13)])
    times = np.array(dates_str, dtype="datetime64[s]")

    data4d = generate_test_array(time=times)
    depth_levels = data4d.depth.shape[0]

    for t in range(len(dates_str)):
        for d in range(depth_levels):
            data4d.T.isel(time=t, depth=d)[:] = t + d
            data4d.S.isel(time=t, depth=d)[:] = 10 - t - 2 * d

    ds = extract_min_max(data=data4d)

    assert len(ds) == 2
