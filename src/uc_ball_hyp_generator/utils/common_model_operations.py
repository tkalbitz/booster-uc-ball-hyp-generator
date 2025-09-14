"""Common model operations for handling compiled models and state dict cleaning."""

from pathlib import Path

import torch
from torch import device
from torch.nn import Module

from uc_ball_hyp_generator.utils.logger import get_logger

_logger = get_logger(__name__)


def clean_compiled_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove _orig_mod. prefixes from compiled model state dict.

    Args:
        state_dict: State dict potentially containing _orig_mod. prefixes

    Returns:
        Cleaned state dict without compile prefixes
    """
    # Check if this is a compiled model (has _orig_mod. prefixes)
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        _logger.info("Detected compiled model, removing _orig_mod. prefixes")
        # Remove _orig_mod. prefixes from compiled model
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                cleaned_key = key[len("_orig_mod.") :]
                cleaned_state_dict[cleaned_key] = value
            else:
                cleaned_state_dict[key] = value
        return cleaned_state_dict
    return state_dict


def load_model_with_clean_state_dict(model: Module, model_weights_path: Path, map_device: device) -> Module:
    """Load model weights with cleaning of compiled state dict prefixes.

    Args:
        model: Model instance to load weights into
        model_weights_path: Path to model weights file
        map_device: Device to map the model to

    Returns:
        Model with loaded weights
    """
    # Load state dict and handle torch.compile prefixes
    state_dict = torch.load(model_weights_path, map_location=map_device)
    cleaned_state_dict = clean_compiled_state_dict(state_dict)
    model.load_state_dict(cleaned_state_dict)
    model.to(map_device)
    model.eval()
    return model
