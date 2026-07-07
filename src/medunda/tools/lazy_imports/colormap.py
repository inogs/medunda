from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.colors import Colormap  # noqa: F401


def __getattr__(name: str):
    if name == "Colormap":
        from matplotlib.colors import Colormap

        return Colormap

    if name == "get_cmap":
        from matplotlib.pyplot import get_cmap

        return get_cmap
    raise AttributeError(f"module {__name__} has no attribute {name}")
