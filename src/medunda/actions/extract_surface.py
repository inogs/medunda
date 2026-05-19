import logging

from medunda.tools.lazy_imports import xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "extract_surface"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME, help="Extract the values of the cells on the surface"
    )


def extract_surface(data: "xr.Dataset") -> "xr.Dataset":
    """Extract the surface layer (first depth level) from the dataset.

    Selects the shallowest depth level (index 0) across all variables,
    removing the ``depth`` dimension from the output dataset.

    Args:
        data (xr.Dataset): Input dataset containing a ``depth`` dimension.

    Returns:
        xr.Dataset: Dataset with the same variables as the input but with the
        ``depth`` dimension removed, containing only surface-level values.
    """
    LOGGER.info(f"reading the file: {data}")

    surface_layer = data.isel(depth=0)

    LOGGER.info("Surface layer extraction completed.")

    return surface_layer
