from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from netCDF4 import *  # noqa: F403

_NETCDF4 = None


def __getattr__(name: str):
    global _NETCDF4
    if _NETCDF4 is None:
        import netCDF4

        _NETCDF4 = netCDF4

    return getattr(_NETCDF4, name)
