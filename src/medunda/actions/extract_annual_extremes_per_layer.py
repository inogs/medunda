import logging

from medunda.actions.extract_annual_extremes import _extract_annual_extremes
from medunda.tools.lazy_imports import xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "extract_annual_extremes_per_layer"


def configure_parser(subparsers):
    subparsers.add_parser(
        ACTION_NAME,
        help="extract the minimum and maximum value of a variable for each layer available in the dataset",
    )


def extract_annual_extremes_per_layer(data: "xr.Dataset") -> "xr.Dataset":
    """Extracts the annual extremes (maximum and minimum) from the dataset for each layer.

    This action is very similar to `extract_annual_extremes`, but it does not include the depth
    dimension in the reduction process. This means that the minimum and maximum values are computed
    for each depth layer separately, rather than across all depth layers. Therefore, the output
    dataset will have the same depth dimension as the input dataset, and the minimum and maximum
    values will be computed for each depth layer independently.
    """
    return _extract_annual_extremes(data=data, include_depth=False)
