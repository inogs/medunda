import logging
from collections.abc import Sequence
from typing import Any

from medunda.tools.lazy_imports import np
from medunda.tools.lazy_imports import xr

LOGGER = logging.getLogger(__name__)
ACTION_NAME = "calculate_stats"


def configure_parser(subparsers):
    calculate_stats_parser = subparsers.add_parser(
        ACTION_NAME,
        help="Calculates the values of some specific statistical operations",
    )
    calculate_stats_parser.add_argument(
        "--operations",
        type=str,
        nargs="+",
        choices=[
            "mean",
            "median",
            "variance",
            "quartiles",
            "minimum",
            "maximum",
            "all",
        ],
        required=False,
        help="Choose the operation(s) required",
    )


class Stats:
    """Provides methods to compute basic statistical operations on an array.

    All operations reduce along axis 0, which is assumed to correspond to the
    time dimension.

    Args:
        data: Numeric array-like object to analyse.
    """

    def __init__(self, data):
        self.data = data

    def mean(self):
        """Return the arithmetic mean along axis 0."""
        return np.mean(self.data, axis=0)

    def variance(self):
        """Return the variance along axis 0."""
        return np.var(self.data, axis=0)

    def median(self):
        """Return the median along axis 0."""
        return np.median(self.data, axis=0)

    def min(self):
        """Return the minimum value along axis 0."""
        return np.min(self.data, axis=0)

    def max(self):
        """Return the maximum value along axis 0."""
        return np.max(self.data, axis=0)

    def quartiles(self):
        """Return the 5th, 25th, 75th, and 95th percentiles along axis 0.

        Returns:
            dict[str, numpy.ndarray]: Dictionary mapping each percentile
            (as a string, e.g. ``"25"``) to the corresponding percentile
            array.
        """
        percentiles = [5, 25, 75, 95]
        output = {
            str(k): np.percentile(self.data, k, axis=0) for k in percentiles
        }
        return output

    def calculate(
        self, operations: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """Compute one or more statistical operations on the stored data.

        Args:
            operations (list[str] | None): Names of the operations to compute.
                Accepted values are ``"mean"``, ``"variance"``,
                ``"median"``, ``"minimum"``, ``"maximum"``, ``"quartiles"``,
                and ``"all"``.  When ``None`` or ``["all"]`` is passed, every
                available operation is computed.

        Returns:
            dict[str, Any]: Dictionary mapping each operation name to its
            result.

        Raises:
            ValueError: If ``"all"`` is combined with other operation names,
                if duplicate operation names are provided, or if an unknown
                operation name is requested.
        """
        available_operations = {
            "mean": self.mean,
            "variance": self.variance,
            "median": self.median,
            "minimum": self.min,
            "maximum": self.max,
            "quartiles": self.quartiles,
        }

        if not operations:
            operations = ["all"]

        if "all" in operations:
            if len(operations) > 1:
                raise ValueError("'all' cannot be used with other operations")
            selected_op = available_operations.keys()

        else:
            if len(set(operations)) != len(operations):
                raise ValueError("Operations cannot be duplicated.")
            selected_op = operations

        for op in selected_op:
            if op not in available_operations:
                raise ValueError(f"Unavailable operation: {op}")

        results = {}

        for operation in selected_op:
            results[operation] = available_operations[operation]()

        return results


def calculate_stats(data: "xr.Dataset", operations) -> "xr.Dataset":
    """Compute statistical operations on each variable in the dataset.

    For each variable in the input dataset (excluding coordinate-like
    variables such as ``depth``, ``latitude``, ``longitude``, and ``time``),
    the requested statistical operations are computed over all dimensions and
    stored as new variables in the output dataset.  Output variable names
    follow the pattern ``{variable}_{operation}``; quartile outputs use
    ``{variable}_quartiles_{percentile}``.

    Args:
        data (xr.Dataset): Input dataset containing the variables to analyse.
        operations (list[str] | None): Statistical operations to compute.
            Accepted values are ``"mean"``, ``"median"``, ``"variance"``,
            ``"quartiles"``, ``"minimum"``, ``"maximum"``, and ``"all"``.
            Pass ``None`` or ``["all"]`` to compute every available
            operation.

    Returns:
        xr.Dataset: Dataset whose variables are the results of the requested
        statistical operations.
    """

    ds_results = xr.Dataset()
    for var_name in data.data_vars:
        if var_name in ("depth", "latitude", "longitude", "time"):
            continue

        data_array = data[var_name].values

        stats = Stats(data_array)
        results = stats.calculate(operations)

        for operation_name, result_array in results.items():
            if isinstance(result_array, dict):
                for method_parameter, method_result in result_array.items():
                    ds_results[
                        f"{var_name}_{operation_name}_{method_parameter}"
                    ] = xr.DataArray(
                        data=method_result,
                        dims=["depth", "latitude", "longitude"],
                    )
            else:
                ds_results[f"{var_name}_{operation_name}"] = xr.DataArray(
                    data=result_array, dims=["depth", "latitude", "longitude"]
                )

    return ds_results
