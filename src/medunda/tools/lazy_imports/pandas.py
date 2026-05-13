from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import *  # noqa: F403


def __getattr__(name: str):
    import pandas

    return getattr(pandas, name)
