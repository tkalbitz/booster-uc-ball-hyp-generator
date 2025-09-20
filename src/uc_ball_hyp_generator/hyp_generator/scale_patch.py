# Our output is from [-1,1]. So we use input_width/2 to cover the whole image.
# We want to also find objects where the center is a bit outside of the picture
# so we scale the number by a factor of 1.2
import torch

from uc_ball_hyp_generator.hyp_generator.config import patch_height, patch_width
from uc_ball_hyp_generator.utils.scale import scale

_patch_output_width: float = (patch_width / 2) * 1.05
_patch_output_height: float = (patch_height / 2) * 1.05
_radius_output: float = 100


def scale_patch_x(x: float | torch.Tensor) -> float | torch.Tensor:
    return scale(x, -_patch_output_width, _patch_output_width, -1, 1)


def unscale_patch_x(x: float | torch.Tensor) -> float | torch.Tensor:
    return scale(x, -1, 1, -_patch_output_width, _patch_output_width)


def scale_patch_y(x: float | torch.Tensor) -> float | torch.Tensor:
    return scale(x, -_patch_output_height, _patch_output_height, -1, 1)


def unscale_patch_y(x: float | torch.Tensor) -> float | torch.Tensor:
    return scale(x, -1, 1, -_patch_output_height, _patch_output_height)


def scale_radius(x: float | torch.Tensor) -> float | torch.Tensor:
    return scale(x, 0, _radius_output, -1, 1)


def unscale_radius(x: float | torch.Tensor) -> float | torch.Tensor:
    return scale(x, -1, 1, 0, _radius_output)
