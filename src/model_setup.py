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

import models
from logger import get_logger
from utils import get_flops
from config import patch_height, patch_width
from scale import unscale_x, unscale_y

_logger = get_logger(__name__)


class DistanceLoss(nn.Module):
    """Custom distance loss for ball detection."""
    
    def __init__(self) -> None:
        super().__init__()
    
    def _logcosh(self, x: torch.Tensor) -> torch.Tensor:
        """Stable log-cosh function."""
        return x + torch.nn.functional.softplus(-2. * x) - torch.log(torch.tensor(2.))
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_t = torch.stack([torch.as_tensor(unscale_x(y_true[:,0])), torch.as_tensor(unscale_y(y_true[:,1]))], dim=1)
        y_p = torch.stack([torch.as_tensor(unscale_x(y_pred[:,0])), torch.as_tensor(unscale_y(y_pred[:,1]))], dim=1)
        
        r = self._logcosh(y_p - y_t)
        e = torch.exp(3. / y_true[:,2])
        r = r * e.unsqueeze(1)
        return torch.mean(r)


def create_model() -> tuple[torch.nn.Module, str, str]:
    """Create and initialize the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.create_network_v2(patch_height, patch_width)
    model = model.to(device)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            sys.exit(1)
        else:
            model_name = sys.argv[1]
    else:
        model_name = "yuv_" + time.strftime("%Y-%m-%d-%H-%M-%S")
    
    model_dir = "model/" + model_name
    os.makedirs(model_dir)

    for f in Path(os.path.realpath(__file__)).parent.glob("*.py"):
        shutil.copy2(str(f.absolute()), model_dir)
    
    log_dir = './logs/' + model_name
    os.makedirs(log_dir)
    
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_weights.pth'))
    
    try:
        flops = get_flops(model, (3, patch_height, patch_width))
        _logger.info("Model has %.2f MFlops", flops / 1e6)
    except Exception as e:
        _logger.warning("Could not calculate FLOPs: %s", e)

    return model, model_dir, log_dir


def create_training_components(model: torch.nn.Module, model_dir: str, log_dir: str) -> tuple[torch.optim.Optimizer, torch.nn.Module, torch.optim.lr_scheduler.ReduceLROnPlateau, SummaryWriter, TextIO]:
    """Create optimizer, loss function, and logging components."""
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    criterion = DistanceLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=15, min_lr=1e-7
    )
    
    writer = SummaryWriter(log_dir)
    
    csv_file = open(os.path.join(model_dir, 'training.csv'), 'w')
    csv_file.write('epoch,loss,val_loss,accuracy,val_accuracy,found_balls,val_found_balls\\n')
    
    return optimizer, criterion, scheduler, writer, csv_file