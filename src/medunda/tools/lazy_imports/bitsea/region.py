from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bitsea.basins.region import *  # noqa: F403


def __getattr__(name: str):
    import bitsea.basins.region

    return getattr(bitsea.basins.region, name)
