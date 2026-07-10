from numpy.typing import ArrayLike

import medunda.tools.lazy_imports.bitsea.geodistances as bitsea_geodistances
from medunda.tools.lazy_imports import numpy as np


def compute_layer_height(layer_centers: ArrayLike) -> "np.ndarray":
    """Computes layer thicknesses from cell center depths.

    Calculates the thickness of each layer in a vertical column based on the
    depths of cell centers.

    Args:
        layer_centers: One-dimensional array containing the depths of layer
            centers in ascending order.

    Returns:
        Array of the same length as `layer_centers` containing the computed
        thickness of each layer.
    """
    level_boundaries = bitsea_geodistances.extend_from_average(
        layer_centers, 0, 0.0
    )
    return level_boundaries[1:] - level_boundaries[:-1]
