"""Shared utility functions for hypothesis generator and classifier."""

from pathlib import Path

import torch
from torch import Tensor, device

from uc_ball_hyp_generator.hyp_generator.config import (
    img_scaled_height,
    img_scaled_width,
    patch_height,
    patch_width,
    scale_factor,
)
from uc_ball_hyp_generator.hyp_generator.model import get_ball_hyp_model
from uc_ball_hyp_generator.hyp_generator.scale_patch import unscale_patch_x, unscale_patch_y, unscale_radius
from uc_ball_hyp_generator.utils.logger import get_logger

_logger = get_logger(__name__)


def create_random_hpatch(
    scaled_image: Tensor, scaled_bbox: tuple[float, float, float, float]
) -> tuple[Tensor, tuple[int, int]]:
    """Extract a random safe crop from scaled image containing the ball.

    Args:
        scaled_image: Down-scaled image tensor of shape (C, H, W)
        scaled_bbox: Bounding box in scaled image coordinates (x1, y1, x2, y2)

    Returns:
        Tuple of (hpatch tensor, hpatch_position) where hpatch_position is (start_x, start_y)
    """
    x1_scaled, y1_scaled, x2_scaled, y2_scaled = scaled_bbox

    # Calculate ball center position
    center_x_scaled = (x1_scaled + x2_scaled) / 2
    center_y_scaled = (y1_scaled + y2_scaled) / 2

    # Minimum distance ball center should stay from patch edges
    min_center_margin_pixels = 1

    # Calculate valid crop region ensuring ball center stays at least min_center_margin_pixels from patch edges
    # For x: ball center must be at least min_center_margin_pixels from left/right patch edges
    # Latest crop start: ensures center is at least min_center_margin_pixels from left patch edge
    max_start_x = min(img_scaled_width - patch_width, int(center_x_scaled - min_center_margin_pixels))
    # Earliest crop start: ensures center is at least min_center_margin_pixels from right patch edge
    min_start_x = max(0, int(center_x_scaled + min_center_margin_pixels - patch_width))

    # For y: ball center must be at least min_center_margin_pixels from top/bottom patch edges
    max_start_y = min(img_scaled_height - patch_height, int(center_y_scaled - min_center_margin_pixels))
    min_start_y = max(0, int(center_y_scaled + min_center_margin_pixels - patch_height))

    # Ensure valid bounds
    if min_start_x > max_start_x:
        min_start_x = max_start_x = max(0, min(img_scaled_width - patch_width, int(center_x_scaled - patch_width / 2)))
    if min_start_y > max_start_y:
        min_start_y = max_start_y = max(
            0, min(img_scaled_height - patch_height, int(center_y_scaled - patch_height / 2))
        )

    # Random crop within valid bounds
    start_x = (
        int(torch.randint(min_start_x, max_start_x + 1, (1,)).item()) if min_start_x < max_start_x else int(min_start_x)
    )
    start_y = (
        int(torch.randint(min_start_y, max_start_y + 1, (1,)).item()) if min_start_y < max_start_y else int(min_start_y)
    )

    # Extract patch
    end_x = start_x + patch_width
    end_y = start_y + patch_height
    hpatch = scaled_image[:, start_y:end_y, start_x:end_x]

    return hpatch, (start_x, start_y)


def transform_hyp_output_to_original_coords(
    prediction: Tensor, hpatch_upper_left_corner: tuple[int, int]
) -> tuple[float, float, float]:
    """Convert hypothesis model output to original image coordinates.

    Args:
        prediction: Raw 3-element prediction tensor (pred_x, pred_y, pred_r) from hyp_model
        hpatch_position: Top-left position of hpatch in scaled image (start_x, start_y)

    Returns:
        Tuple of (center_x, center_y, diameter) in original image coordinates
    """
    # Apply tanh activation to predictions
    pred_x, pred_y, pred_r = torch.tanh(prediction)

    # Convert [-1, 1] predictions to patch-relative pixel coordinates
    patch_rel_x = unscale_patch_x(pred_x.item())
    patch_rel_y = unscale_patch_y(pred_y.item())
    radius = unscale_radius(pred_r.item())

    # Convert patch-relative to scaled image coordinates
    hpatch_start_x, hpatch_start_y = hpatch_upper_left_corner
    scaled_center_x = hpatch_start_x + patch_width / 2 + patch_rel_x
    scaled_center_y = hpatch_start_y + patch_height / 2 + patch_rel_y

    # Scale up to original image coordinates
    original_center_x = scaled_center_x * scale_factor
    original_center_y = scaled_center_y * scale_factor
    original_diameter = radius * 2 * scale_factor

    return original_center_x, original_center_y, original_diameter


def load_ball_hyp_model(model_weights_path: Path, device: device) -> torch.nn.Module:
    """Load the ball hypothesis model."""

    current_model_path = model_weights_path

    _logger.info("Loading ball hypothesis model from: %s", current_model_path)

    model = get_ball_hyp_model(patch_height, patch_width)

    # Load state dict and handle torch.compile prefixes
    state_dict = torch.load(current_model_path, map_location=device)

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
        state_dict = cleaned_state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _logger.info("Model loaded successfully on device: %s", device)

    return model
