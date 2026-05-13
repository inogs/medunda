from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dask.array import *  # noqa: F403


def __getattr__(name: str):
    import dask.array

    return getattr(dask.array, name)
