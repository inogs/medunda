import numpy as np
import xarray as xr
from pytest import fixture


def generate_test_array(depth=None, latitude=None, longitude=None, time=None):
    if depth is None:
        depth = np.array([0.5, 2., 5.], dtype=np.float32)
    
    if latitude is None:
        latitude = np.array([42, 43, 44, 45], dtype=np.float32)
    
    if longitude is None:
        longitude = np.array([30, 30.5, 31, 31.5, 32], dtype=np.float32)

    if time is None:
        time = np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[s]")

    var1 = np.empty(
        (len(time), len(depth), len(latitude), len(longitude)),
        dtype=np.float32
    )
    var2 = np.empty(
        (len(time), len(depth), len(latitude), len(longitude)),
        dtype=np.float32
    )

    return xr.Dataset(
        {
            "T": (("time", "depth", "latitude", "longitude"), var1),
            "S": (("time", "depth", "latitude", "longitude"), var2)
        },
        coords={
            "depth": depth,
            "latitude": latitude,
            "longitude": longitude,
            "time": time
        }
    )


@fixture
def data4d():
    return generate_test_array()

