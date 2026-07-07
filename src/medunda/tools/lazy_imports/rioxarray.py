from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rioxarray import *  # noqa: F403


def __getattr__(name: str):
    import rioxarray

    return getattr(rioxarray, name)
