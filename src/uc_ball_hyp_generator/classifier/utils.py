"""Utility functions for ball classifier."""

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from uc_ball_hyp_generator.utils.logger import get_logger

if TYPE_CHECKING:
    from uc_ball_hyp_generator.hyp_generator.utils import BallHypothesis
else:
    BallHypothesis = object

from uc_ball_hyp_generator.classifier.config import CLASSIFIER_DILATION_FACTOR, CPATCH_SIZE
from uc_ball_hyp_generator.classifier.model import get_ball_classifier_model
from uc_ball_hyp_generator.utils.common_model_operations import load_model_with_clean_state_dict

_logger = get_logger(__name__)


def run_ball_classifier_model(
    model: torch.nn.Module, yuv_image: Tensor, ball_hyps: list[BallHypothesis]
) -> list[float]:
    """Run ball classifier model on patches extracted around ball hypotheses.

    Args:
        model: Loaded ball classifier model
        yuv_image: YUV image tensor of shape (3, H, W)
        ball_hyps: List of ball hypotheses with center coordinates and diameter

    Returns:
        List of probability scores for each hypothesis (probability of being a ball)
    """
    if not ball_hyps:
        return []

    img_height, img_width = yuv_image.shape[1], yuv_image.shape[2]

    # Pre-allocate batch tensor
    batch = torch.zeros((len(ball_hyps), 3, CPATCH_SIZE, CPATCH_SIZE), device=yuv_image.device, dtype=yuv_image.dtype)

    # Extract patches for each hypothesis
    for idx, hyp in enumerate(ball_hyps):
        # Calculate desired patch size based on dilated diameter
        desired_patch_size = int(CLASSIFIER_DILATION_FACTOR * hyp.diameter)
        # Ensure at least 1 pixel
        desired_patch_size = max(1, desired_patch_size)

        # Ensure desired patch size doesn't exceed image dimensions
        desired_patch_size = min(desired_patch_size, img_width, img_height)

        # Calculate half size for cropping
        half_size = desired_patch_size // 2

        # Calculate patch center coordinates
        center_x, center_y = hyp.center_x, hyp.center_y

        # Calculate initial patch boundaries
        start_x = int(center_x - half_size)
        start_y = int(center_y - half_size)

        # Adjust start positions to ensure patch stays within image boundaries
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        if start_x + desired_patch_size > img_width:
            start_x = img_width - desired_patch_size
        if start_y + desired_patch_size > img_height:
            start_y = img_height - desired_patch_size

        # Calculate final patch boundaries
        end_x = start_x + desired_patch_size
        end_y = start_y + desired_patch_size

        # Extract patch from image
        patch: Tensor = yuv_image[:, start_y:end_y, start_x:end_x]

        # Resize patch to CPATCH_SIZE using interpolation
        if patch.shape[1] != CPATCH_SIZE or patch.shape[2] != CPATCH_SIZE:
            patch = F.interpolate(
                patch.unsqueeze(0), size=(CPATCH_SIZE, CPATCH_SIZE), mode="bilinear", align_corners=False
            ).squeeze(0)

        batch[idx] = patch

    # Run model on batch
    with torch.no_grad():
        outputs: Tensor = model(batch)

    # Apply sigmoid to get probabilities
    probabilities = outputs.squeeze(1).cpu().tolist()

    return probabilities


def load_ball_classifier_model(model_weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the ball classifier model.

    Args:
        model_weights_path: Path to the model weights file
        device: Device to load the model on

    Returns:
        Loaded ball classifier model
    """
    model = get_ball_classifier_model()
    model = load_model_with_clean_state_dict(model, model_weights_path, device)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    _logger.info("Classifier model compiled with torch.compile for better inference performance")

    return model
