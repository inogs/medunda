from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bitsea.basins.basin import *  # noqa: F403


def __getattr__(name: str):
    import bitsea.basins.basin

    return getattr(bitsea.basins.basin, name)
