from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bitsea.commons.mask import *  # noqa: F403


def __getattr__(name: str):
    import bitsea.commons.mask

    return getattr(bitsea.commons.mask, name)
