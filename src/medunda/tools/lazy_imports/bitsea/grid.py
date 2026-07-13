from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bitsea.commons.grid import *  # noqa: F403


def __getattr__(name: str):
    import bitsea.commons.grid

    return getattr(bitsea.commons.grid, name)
