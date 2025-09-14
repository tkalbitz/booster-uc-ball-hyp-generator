"""Ball detection adapter for naoteamhtwk_machinelearning_visualizer.

This adapter loads a trained PyTorch ball detection model and runs inference on images,
splitting them into patches as during training and returning VisualizationResult objects
with ball detection annotations using EllipseShape.
"""

import logging
from pathlib import Path

import kornia
import torch
import torchvision.transforms.v2 as transforms_v2  # type: ignore[import-untyped]
from naoteamhtwk_machinelearning_visualizer.core.shapes import Annotation, EllipseShape, Point, VisualizationResult
from torchvision.io import ImageReadMode, decode_image  # type: ignore[import-untyped]

from uc_ball_hyp_generator.classifier.utils import load_ball_classifier_model, run_ball_classifier_model
from uc_ball_hyp_generator.hyp_generator.config import (
    img_scaled_height,
    img_scaled_width,
    patch_height,
    patch_width,
)
from uc_ball_hyp_generator.hyp_generator.utils import BallHypothesis, load_ball_hyp_model, run_ball_hyp_model

_logger = logging.getLogger(__name__)

CLASSIFIER_AVAILABLE = True

# Global model instance to avoid reloading
_device: torch.device | None = None
_hyp_model: torch.nn.Module | None = None
_hyp_model_path: Path | None = None
_classifier_model_path: Path | None = None
_classifier_model: torch.nn.Module | None = None


def _preprocess_image(image_path: str) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Load and preprocess image for model inference."""
    # Load image using torchvision.io.decode_image
    image_tensor = decode_image(image_path, mode=ImageReadMode.RGB)

    # Get original size before scaling
    original_height, original_width = image_tensor.shape[1], image_tensor.shape[2]
    original_size = (original_width, original_height)  # (width, height)

    # Convert from uint8 [0, 255] to float32 [0, 1]
    image_tensor_float = transforms_v2.ToDtype(torch.float32, scale=True)(image_tensor)

    # Create scaled image
    scaled_image = transforms_v2.Resize((img_scaled_height, img_scaled_width), antialias=True)(image_tensor_float)

    # Convert scaled image to YUV using Kornia
    scaled_yuv = kornia.color.rgb_to_yuv(scaled_image.unsqueeze(0)).squeeze(0)  # (C, H, W) in YUV

    return image_tensor_float, scaled_yuv, original_size


def _convert_hypothesis_to_annotation(
    hyp: BallHypothesis, original_size: tuple[int, int], prob: float | None = None
) -> Annotation:
    """Convert ball hypothesis to visualization annotation."""
    orig_size_x, orig_size_y = original_size

    # Convert to relative coordinates (0-1) for the visualizer
    rel_x = hyp.center_x / orig_size_x
    rel_y = hyp.center_y / orig_size_y

    # Clamp to valid range
    rel_x = max(0.0, min(1.0, rel_x))
    rel_y = max(0.0, min(1.0, rel_y))

    # Define ellipse size (relative to image size)
    ellipse_width = hyp.diameter / orig_size_x
    ellipse_height = hyp.diameter / orig_size_y

    # Create ellipse shape centered at predicted position
    center = Point(x=rel_x, y=rel_y)
    shape = EllipseShape(center=center, width=ellipse_width, height=ellipse_height)

    # Set color based on classifier probability if available
    if prob is not None:
        # Interpolate between red (0) and green (1)
        red = int(255 * (1 - prob))
        green = int(255 * prob)
        blue = 0
        color = f"#{red:02x}{green:02x}{blue:02x}"
    else:
        color = "#FF4500"  # Orange color for ball

    # Create annotation
    annotation = Annotation(
        shape=shape,
        text="ball",
        accuracy=prob,
        color=color,
        outline_color="#FFFFFF",  # White outline
    )

    return annotation


def adapter(image_paths: list[str]) -> list[VisualizationResult]:
    """Process images with ball detection model and return visualization results.

    Args:
        image_paths: List of image file paths to process

    Returns:
        List of VisualizationResult objects with ball detection annotations
    """
    global _hyp_model, _hyp_model_path, _classifier_model, _classifier_model_path
    _logger.info("Processing %d images with patch-based ball detection model", len(image_paths))

    cur_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not _hyp_model and _hyp_model_path:
        _hyp_model = load_ball_hyp_model(_hyp_model_path, cur_device)
        _logger.info("Loaded hypothesis model from: %s", _hyp_model_path)

        if _classifier_model_path:
            _classifier_model = load_ball_classifier_model(_classifier_model_path, cur_device)
            _logger.info("Loaded classifier model from: %s", _classifier_model_path)

    results: list[VisualizationResult] = []

    if not _hyp_model:
        _logger.warning("Hypothesis model not loaded. Skipping image processing.")
        return results

    for image_path in image_paths:
        try:
            _logger.info("Processing image: %s", image_path)

            # Preprocess image
            original_image, yuv_image, original_size = _preprocess_image(image_path)
            yuv_image = yuv_image.to(cur_device)

            # Run ball hypothesis model
            hypotheses = run_ball_hyp_model(_hyp_model, yuv_image)

            # Run classifier if enabled
            probs = None
            if _classifier_model and hypotheses:
                # Convert original image to YUV for classifier
                original_image_yuv = kornia.color.rgb_to_yuv(original_image.unsqueeze(0)).squeeze(0).to(cur_device)
                probs = run_ball_classifier_model(_classifier_model, original_image_yuv, hypotheses)

            # Convert hypotheses to annotations
            annotations = []
            for idx, hyp in enumerate(hypotheses):
                # Filter out hypotheses that are outside the image bounds
                if 0 <= hyp.center_x <= original_size[0] and 0 <= hyp.center_y <= original_size[1] and hyp.diameter > 0:
                    # Get probability if available
                    prob = probs[idx] if probs is not None else None
                    annotation = _convert_hypothesis_to_annotation(hyp, original_size, prob)
                    annotations.append(annotation)

            result = VisualizationResult(
                image_path=Path(image_path).resolve(), annotations=annotations, result_image=None
            )
            results.append(result)

        except Exception as e:
            _logger.error("Failed to process %s: %s", image_path, e)
            # Create empty result for failed image
            result = VisualizationResult(image_path=Path(image_path).resolve(), annotations=[], result_image=None)
            results.append(result)

    _logger.info("Successfully processed %d images", len(results))
    return results


def get_adapter_info() -> dict[str, str]:
    """Get information about this ball detection adapter.

    Returns:
        Dictionary containing adapter metadata
    """
    return {
        "name": "UC Ball Detection (Patch-based)",
        "version": "1.0.0",
        "description": "Ball detection using trained PyTorch model with patch-based processing for UC ball hypothesis generation",
        "supported_formats": "JPEG, PNG, and other PIL-supported formats",
        "model_type": "NetworkV2 (Custom CNN)",
        "classes": "ball",
        "requirements": "torch>=2.8.0, PIL, numpy",
        "environment": "Set BALL_MODEL_PATH to point to the .pth model file",
        "patch_info": f"Processes images as {img_scaled_width // patch_width}x{img_scaled_height // patch_height} grid of {patch_width}x{patch_height} patches",
    }
