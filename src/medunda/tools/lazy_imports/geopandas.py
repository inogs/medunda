from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geopandas import *  # noqa: F403


def __getattr__(name: str):
    import geopandas

    return getattr(geopandas, name)
