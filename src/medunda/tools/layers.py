import numpy as np
from numpy.typing import ArrayLike


def compute_layer_height(layer_centers: ArrayLike) -> np.ndarray:
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
    layer_height = []
    for i in range(len(layer_centers)):
        if i == 0:
            layer_height.append(layer_centers[0] * 2)
        else:
            current_layer = (layer_centers[i] - sum(layer_height[:i])) * 2
            layer_height.append(current_layer)
    return np.array(layer_height)
