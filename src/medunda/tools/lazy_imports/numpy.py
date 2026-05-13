from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import *  # noqa: F403


def __getattr__(name: str):
    import numpy

    return getattr(numpy, name)
