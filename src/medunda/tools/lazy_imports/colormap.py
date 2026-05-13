from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.colors import Colormap  # noqa: F401


def __getattr__(name: str):
    if name == "Colormap":
        from matplotlib.colors import Colormap

        return Colormap
    raise AttributeError(f"module {__name__} has no attribute {name}")
