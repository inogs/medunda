from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dask.delayed import *  # noqa: F403


def __getattr__(name: str):
    # The attribute "delayed" of the dask package is the function
    # dask.delayed.delayed, which shadows the module with the same name.
    # Therefore, we retrieve the module using importlib.
    from importlib import import_module

    return getattr(import_module("dask.delayed"), name)
