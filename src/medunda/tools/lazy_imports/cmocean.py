from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


_CMOCEAN = None


def __getattr__(name: str):
    global _CMOCEAN
    if _CMOCEAN is None:
        import cmocean as _CMOCEAN
    return getattr(_CMOCEAN, name)


def get_cmocean_map(name: str) -> "Colormap":
    global _CMOCEAN
    if _CMOCEAN is None:
        import cmocean as _CMOCEAN
    return getattr(_CMOCEAN.cm, name)
