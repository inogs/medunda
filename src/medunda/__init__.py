import warnings

from medunda.components.geodata import GeoDataCollection


def _configure_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Setting the shape on a NumPy array has been deprecated in NumPy 2\.5\..*",
        category=DeprecationWarning,
    )


_configure_warning_filters()

__all__ = ["GeoDataCollection"]
