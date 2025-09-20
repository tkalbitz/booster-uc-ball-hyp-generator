"""Linear scaling utilities for numerical values and tensors."""

from typing import overload

import torch


@overload
def scale(x: float, from_min: float, from_max: float, to_min: float, to_max: float) -> float: ...
@overload
def scale(x: torch.Tensor, from_min: float, from_max: float, to_min: float, to_max: float) -> torch.Tensor: ...
def scale(
    x: float | torch.Tensor, from_min: float, from_max: float, to_min: float, to_max: float
) -> float | torch.Tensor:
    """Scale a value from one range to another using linear interpolation.

    Transforms values from the range [from_min, from_max] to [to_min, to_max].
    This is commonly used for normalizing data or converting between different
    coordinate systems or value ranges.

    Args:
        x: The value(s) to scale - can be a single float or torch.Tensor
        from_min: Minimum value of the input range
        from_max: Maximum value of the input range
        to_min: Minimum value of the output range
        to_max: Maximum value of the output range

    Returns:
        Scaled value(s) in the target range, same type as input

    Example:
        >>> scale(5.0, 0.0, 10.0, 0.0, 100.0)  # Scale 5 from [0,10] to [0,100]
        50.0
        >>> scale(torch.tensor([2.0, 4.0]), 0.0, 10.0, -1.0, 1.0)  # Scale to [-1,1]
        tensor([-0.6, -0.2])
    """
    if isinstance(x, torch.Tensor):
        return ((to_max - to_min) * (x - from_min)) / (from_max - from_min) + to_min
    else:
        return ((to_max - to_min) * (float(x) - from_min)) / (from_max - from_min) + to_min
