import torch

# Our output is from [-1,1]. So we use input_width/2 to cover the whole image.
# We want to also find objects where the center is a bit outside of the picture so we scale the number by a factor of 1.2
from config import patch_width, patch_height

output_width: float = (patch_width/2)*1.2
output_height: float = (patch_height/2)*1.2


def scale(x: float | torch.Tensor, from_min: float, from_max: float, to_min: float, to_max: float) -> float | torch.Tensor:
    return ((to_max-to_min)*(x-from_min)) / (from_max-from_min) + to_min


def scale_x(x: float | torch.Tensor) -> float | torch.Tensor:
    return scale(x, -output_width, output_width, -1, 1)


def unscale_x(x: float | torch.Tensor) -> float | torch.Tensor:
    return scale(x, -1, 1, -output_width, output_width)


def scale_y(x: float | torch.Tensor) -> float | torch.Tensor:
    return scale(x, -output_height, output_height, -1, 1)


def unscale_y(x: float | torch.Tensor) -> float | torch.Tensor:
    return scale(x, -1, 1, -output_height, output_height)
