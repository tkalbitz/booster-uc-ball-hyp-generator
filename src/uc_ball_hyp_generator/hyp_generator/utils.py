"""Shared utility functions for hypothesis generator and classifier."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, device

from uc_ball_hyp_generator.hyp_generator.config import (
    patch_height,
    patch_width,
    scale_factor,
)
from uc_ball_hyp_generator.hyp_generator.model import get_ball_hyp_model
from uc_ball_hyp_generator.hyp_generator.scale_patch import unscale_patch_x, unscale_patch_y, unscale_radius
from uc_ball_hyp_generator.utils.common_model_operations import load_model_with_clean_state_dict
from uc_ball_hyp_generator.utils.logger import get_logger

_logger = get_logger(__name__)


@dataclass
class BallHypothesis:
    """Dataclass representing a ball hypothesis with center coordinates and diameter."""

    center_x: float
    center_y: float
    diameter: float


def _get_boundary_aware_patch_bounds(
    center_x: float, center_y: float, img_width: int, img_height: int
) -> tuple[int, int]:
    """Calculate patch bounds ensuring patch stays within image boundaries.

    If patch extends beyond image border, the outer edge aligns with the border
    and start position is calculated backwards.

    Args:
        center_x: Target center x coordinate
        center_y: Target center y coordinate
        img_width: Image width
        img_height: Image height

    Returns:
        Tuple of (start_x, start_y) for patch extraction
    """
    # Initial patch position based on center
    start_x = int(center_x - patch_width / 2)
    start_y = int(center_y - patch_height / 2)

    # Adjust if patch extends beyond right/bottom borders
    if start_x + patch_width > img_width:
        start_x = img_width - patch_width
    if start_y + patch_height > img_height:
        start_y = img_height - patch_height

    # Ensure start positions are non-negative
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    return start_x, start_y


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
    img_height, img_width = scaled_image.shape[1], scaled_image.shape[2]

    # Calculate ball center position
    center_x_scaled = (x1_scaled + x2_scaled) / 2
    center_y_scaled = (y1_scaled + y2_scaled) / 2

    # Minimum distance ball center should stay from patch edges
    min_center_margin_pixels = 1

    # Calculate valid crop region ensuring ball center stays at least min_center_margin_pixels from patch edges
    # For x: ball center must be at least min_center_margin_pixels from left/right patch edges
    # Latest crop start: ensures center is at least min_center_margin_pixels from left patch edge
    max_start_x = min(img_width - patch_width, int(center_x_scaled - min_center_margin_pixels))
    # Earliest crop start: ensures center is at least min_center_margin_pixels from right patch edge
    min_start_x = max(0, int(center_x_scaled + min_center_margin_pixels - patch_width))

    # For y: ball center must be at least min_center_margin_pixels from top/bottom patch edges
    max_start_y = min(img_height - patch_height, int(center_y_scaled - min_center_margin_pixels))
    min_start_y = max(0, int(center_y_scaled + min_center_margin_pixels - patch_height))

    # Ensure valid bounds - if image is too small, use boundary-aware positioning
    if min_start_x > max_start_x:
        min_start_x = max_start_x = max(0, min(img_width - patch_width, int(center_x_scaled - patch_width / 2)))
    if min_start_y > max_start_y:
        min_start_y = max_start_y = max(0, min(img_height - patch_height, int(center_y_scaled - patch_height / 2)))

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

    model = load_model_with_clean_state_dict(model, current_model_path, device)

    _logger.info("Model loaded successfully on device: %s", device)

    return model


def run_ball_hyp_model(model: torch.nn.Module, scaled_yuv_image: Tensor) -> list[BallHypothesis]:
    """Run ball hypothesis model on scaled YUV image by processing patches in batch.

    Args:
        model: Loaded ball hypothesis model
        scaled_yuv_image: Scaled image tensor in YUV color space with shape (3, H, W)

    Returns:
        List of BallHypothesis objects with center coordinates and diameter in original image coordinates
    """
    img_height, img_width = scaled_yuv_image.shape[1], scaled_yuv_image.shape[2]

    # Calculate number of patches dynamically based on image dimensions
    n_patches_w = (img_width + patch_width - 1) // patch_width
    n_patches_h = (img_height + patch_height - 1) // patch_height

    # Calculate all patch positions and bounds in vectorized manner
    patch_positions = []
    patches_list = []

    for i in range(n_patches_w):
        for j in range(n_patches_h):
            # Calculate center position for this patch
            center_x = i * patch_width + patch_width / 2
            center_y = j * patch_height + patch_height / 2

            # Use boundary-aware patch bounds
            start_x, start_y = _get_boundary_aware_patch_bounds(center_x, center_y, img_width, img_height)

            # Extract patch from image
            end_x = start_x + patch_width
            end_y = start_y + patch_height
            patch = scaled_yuv_image[:, start_y:end_y, start_x:end_x]

            patches_list.append(patch)
            patch_positions.append((start_x, start_y))

    # Vectorize batch creation using torch.stack for GPU efficiency
    batch = torch.stack(patches_list, dim=0)

    # Run model on entire batch
    with torch.no_grad():
        if batch.is_cuda:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(batch.to(memory_format=torch.channels_last))
        else:
            outputs = model(batch)

    # Process all outputs
    hypotheses = []
    for idx, output in enumerate(outputs):
        center_x, center_y, diameter = transform_hyp_output_to_original_coords(output, patch_positions[idx])
        hypotheses.append(BallHypothesis(center_x=center_x, center_y=center_y, diameter=diameter))

    return hypotheses
