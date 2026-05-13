from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarray import DataArray  # noqa: F401
    from xarray import Dataset  # noqa: F401


def __getattr__(name: str):
    import xarray

    return getattr(xarray, name)
