from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bitsea.commons.geodistances import *  # noqa: F403


def __getattr__(name: str):
    import bitsea.commons.geodistances

    return getattr(bitsea.commons.geodistances, name)
