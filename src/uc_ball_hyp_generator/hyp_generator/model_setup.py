"""Model initialization and setup utilities."""

import os
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.nn import HuberLoss

from uc_ball_hyp_generator.hyp_generator.config import patch_height, patch_width
from uc_ball_hyp_generator.hyp_generator.model import get_ball_hyp_model
from uc_ball_hyp_generator.hyp_generator.scale_patch import unscale_patch_x, unscale_patch_y, unscale_radius
from uc_ball_hyp_generator.utils.flops import get_flops
from uc_ball_hyp_generator.utils.logger import get_logger

_logger = get_logger(__name__)


class BallSmoothL1Loss(HuberLoss):
    def __init__(self, reduction: str = "mean", delta: float = 1.0) -> None:
        super().__init__(reduction=reduction, delta=delta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_t_x = unscale_patch_x(torch.tanh(y_true[:, 0]))
        y_t_y = unscale_patch_y(torch.tanh(y_true[:, 1]))
        y_p_x = unscale_patch_x(torch.tanh(y_pred[:, 0]))
        y_p_y = unscale_patch_y(torch.tanh(y_pred[:, 1]))

        y_t_xy = torch.stack([y_t_x, y_t_y], dim=1)
        y_p_xy = torch.stack([y_p_x, y_p_y], dim=1)

        return super().forward(y_p_xy, y_t_xy)


class DistanceLoss(HuberLoss):
    """Custom distance loss for ball detection."""

    def __init__(self) -> None:
        super().__init__()
        # Pre-compute constant tensor to avoid creating it in forward pass
        self.register_buffer("log_two", torch.log(torch.tensor(2.0)))
        self.register_buffer("confidence_scale", torch.tensor(3.0))

    def _logcosh(self, x: torch.Tensor) -> torch.Tensor:
        """Stable log-cosh function."""
        return x + torch.nn.functional.softplus(-2.0 * x) - torch.as_tensor(self.log_two)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_true 0 = x, 1 = y, 2 diametre
        # Vectorized unscaling operations - ensure tensor output
        y_t_x = unscale_patch_x(torch.tanh(y_true[:, 0]))
        y_t_y = unscale_patch_y(torch.tanh(y_true[:, 1]))
        y_p_x = unscale_patch_x(torch.tanh(y_pred[:, 0]))
        y_p_y = unscale_patch_y(torch.tanh(y_pred[:, 1]))

        y_t_r = unscale_radius(torch.tanh(y_true[:, 2]))
        y_p_r = unscale_radius(torch.tanh(y_pred[:, 2]))

        y_t_xy = torch.stack([y_t_x, y_t_y], dim=1)
        y_p_xy = torch.stack([y_p_x, y_p_y], dim=1)

        y_r = y_t_r - y_p_r

        xy = self._logcosh(torch.abs(y_p_xy - y_t_xy))
        r = self._logcosh(torch.abs(y_r))
        e = torch.exp(self.confidence_scale / y_true[:, 3])

        xy = xy * e.unsqueeze(1)
        r = r * e.unsqueeze(1)
        return torch.mean(r) + torch.mean(xy)


def create_model(compile_model: bool = True) -> tuple[torch.nn.Module, str, str]:
    """Create and initialize the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_ball_hyp_model(patch_height, patch_width)
    model = model.to(device)

    # Apply PyTorch 2.0 compilation for better performance
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default", fullgraph=False)  # type: ignore[assignment]
            _logger.info("Model compiled with torch.compile for better performance")
        except Exception as e:
            _logger.warning("Failed to compile model: %s", e)
            _logger.warning("Continuing with uncompiled model")

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            sys.exit(1)
        else:
            model_name = sys.argv[1]
    else:
        model_name = "yuv_" + time.strftime("%Y-%m-%d-%H-%M-%S")

    model_dir = "model/" + model_name
    os.makedirs(model_dir)

    for f in Path(os.path.realpath(__file__)).parent.glob("*.py"):
        shutil.copy2(str(f.absolute()), model_dir)

    log_dir = "./logs/" + model_name
    os.makedirs(log_dir)

    torch.save(model.state_dict(), os.path.join(model_dir, "model_weights.pth"))

    try:
        flops = get_flops(model, (3, patch_height, patch_width))
        _logger.info("Model has %.2f MFlops", flops / 1e6)
    except Exception as e:
        _logger.warning("Could not calculate FLOPs: %s", e)

    return model, model_dir, log_dir


def compile_existing_model(model: torch.nn.Module, mode: str = "default") -> torch.nn.Module:
    """Compile an existing model for better performance.

    Args:
        model: The model to compile
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')

    Returns:
        Compiled model or original model if compilation fails
    """
    if hasattr(torch, "compile"):
        try:
            compiled_model = torch.compile(model, mode=mode, fullgraph=False)  # type: ignore[assignment]
            _logger.info("Model compiled with torch.compile (mode: %s)", mode)
            return compiled_model  # type: ignore[return-value]
        except Exception as e:
            _logger.warning("Failed to compile model: %s", e)
            _logger.warning("Returning uncompiled model")
            return model
    else:
        _logger.warning("torch.compile not available, returning uncompiled model")
        return model
