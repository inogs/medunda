import logging

from medunda.components.dataset import Dataset
from medunda.plotter import VAR_METADATA

logger = logging.getLogger(__name__)


def check_units(ds: Dataset, fill_from_metadata: bool = True):
    for var_name in ds.data_vars:
        var = ds[var_name]
        units = ds.attrs.get("units", None)

        if units is None and fill_from_metadata:
            unit_meta = VAR_METADATA.get(var_name, {}).get("unit")
            if unit_meta:
                var.attrs["units"] = unit_meta
                logger.warning(
                    f"Variable {var_name} have no unit so it is filled from VAR_METADATA"
                )
            else:
                raise ValueError(f"Variable {var_name} has no units")
