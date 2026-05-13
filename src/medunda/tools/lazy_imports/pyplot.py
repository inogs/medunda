from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.pyplot import *  # noqa: F403


def __getattr__(name: str):
    import matplotlib.pyplot as plt

    return getattr(plt, name)
