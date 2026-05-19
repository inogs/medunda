import logging

from medunda.tools.lazy_imports import xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "extract_layer"


def configure_parser(subparsers):
    extract_layer_parser = subparsers.add_parser(
        ACTION_NAME, help="Extract the values of a specific depth (in metres)"
    )
    extract_layer_parser.add_argument(
        "--depth",
        type=float,
        required=True,
        help="Depth of the layer that must be extracted",
    )


def extract_layer(data: "xr.Dataset", depth: float) -> "xr.Dataset":
    """Extract the layer nearest to a specified depth from the dataset.

    Uses nearest-neighbour selection along the ``depth`` coordinate, so the
    actually selected depth may differ slightly from the requested value when
    an exact match is not available in the dataset.

    Args:
        data (xr.Dataset): Input dataset containing a ``depth`` coordinate.
        depth (float): Target depth in metres.

    Returns:
        xr.Dataset: Dataset with the same variables as the input but with the
        ``depth`` dimension removed, containing values at the depth level
        closest to *depth*.
    """

    LOGGER.info(f"reading the file: {data}")

    selected_layer = data.sel(depth=depth, method="nearest")

    LOGGER.info("Layer extraction completed.")

    return selected_layer
