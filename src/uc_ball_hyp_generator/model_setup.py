"""Model initialization and setup utilities."""

import os
import shutil
import sys
import time
from pathlib import Path
from typing import TextIO

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from uc_ball_hyp_generator import models as models
from uc_ball_hyp_generator.config import patch_height, patch_width
from uc_ball_hyp_generator.logger import get_logger
from uc_ball_hyp_generator.scale import unscale_x, unscale_y
from uc_ball_hyp_generator.utils import get_flops

_logger = get_logger(__name__)


class DistanceLoss(nn.Module):
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
        # Vectorized unscaling operations - ensure tensor output
        y_t_x = unscale_x(y_true[:, 0])
        y_t_y = unscale_y(y_true[:, 1])
        y_p_x = unscale_x(y_pred[:, 0])
        y_p_y = unscale_y(y_pred[:, 1])

        # Convert to tensors if they aren't already
        if not isinstance(y_t_x, torch.Tensor):
            y_t_x = torch.tensor(y_t_x, device=y_true.device, dtype=y_true.dtype)
        if not isinstance(y_t_y, torch.Tensor):
            y_t_y = torch.tensor(y_t_y, device=y_true.device, dtype=y_true.dtype)
        if not isinstance(y_p_x, torch.Tensor):
            y_p_x = torch.tensor(y_p_x, device=y_pred.device, dtype=y_pred.dtype)
        if not isinstance(y_p_y, torch.Tensor):
            y_p_y = torch.tensor(y_p_y, device=y_pred.device, dtype=y_pred.dtype)

        y_t_xy = torch.stack([y_t_x, y_t_y], dim=1)
        y_p_xy = torch.stack([y_p_x, y_p_y], dim=1)

        r = self._logcosh(y_p_xy - y_t_xy)
        e = torch.exp(self.confidence_scale / y_true[:, 2])
        r = r * e.unsqueeze(1)
        return torch.mean(r)


def create_model(compile_model: bool = True) -> tuple[torch.nn.Module, str, str]:
    """Create and initialize the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.create_network_v2(patch_height, patch_width)
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


def create_training_components(
    model: torch.nn.Module, model_dir: str, log_dir: str
) -> tuple[torch.optim.Optimizer, torch.nn.Module, torch.optim.lr_scheduler.ReduceLROnPlateau, SummaryWriter, TextIO]:
    """Create optimizer, loss function, and logging components."""
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    criterion = DistanceLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=15, min_lr=1e-7)

    writer = SummaryWriter(log_dir)

    csv_file = open(os.path.join(model_dir, "training.csv"), "w")
    csv_file.write("epoch,loss,val_loss,accuracy,val_accuracy,found_balls,val_found_balls\\n")

    return optimizer, criterion, scheduler, writer, csv_file


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
