"""Ball detection adapter for naoteamhtwk_machinelearning_visualizer.

This adapter loads a trained PyTorch ball detection model and runs inference on images,
splitting them into patches as during training and returning VisualizationResult objects
with ball detection annotations using EllipseShape.
"""

import logging
import os
from pathlib import Path

import kornia
import torch
import torchvision.transforms.v2 as transforms_v2  # type: ignore[import-untyped]
from naoteamhtwk_machinelearning_visualizer.core.shapes import Annotation, EllipseShape, Point, VisualizationResult
from torchvision.io import ImageReadMode, decode_image  # type: ignore[import-untyped]

from uc_ball_hyp_generator.hyp_generator.config import (
    img_scaled_height,
    img_scaled_width,
    patch_height,
    patch_width,
)
from uc_ball_hyp_generator.hyp_generator.utils import BallHypothesis, load_ball_hyp_model, run_ball_hyp_model

_logger = logging.getLogger(__name__)

# Global model instance to avoid reloading
_model: torch.nn.Module | None = None
_device: torch.device | None = None
_model_path: str | None = None


def _get_model_path() -> str:
    """Get the model path from environment variable or use default."""
    model_path = os.environ.get("BALL_MODEL_PATH")
    if model_path is None:
        msg = "BALL_MODEL_PATH environment variable must be set to the .pth model file"
        raise RuntimeError(msg)

    if not Path(model_path).exists():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)

    return model_path


def _load_model() -> tuple[torch.nn.Module, torch.device]:
    """Load the ball detection model."""
    global _model, _device, _model_path

    current_model_path = _get_model_path()

    if _model is None or _model_path != current_model_path:
        _device = torch.device("cpu")
        if torch.cuda.is_available():
            _device = torch.device("cuda")

        _model = load_ball_hyp_model(Path(current_model_path), _device)
        _model_path = current_model_path

    return _model, _device


def _preprocess_image(image_path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load and preprocess image for model inference."""
    # Load image using torchvision.io.decode_image
    image_tensor = decode_image(image_path, mode=ImageReadMode.RGB)

    # Get original size before scaling
    original_height, original_width = image_tensor.shape[1], image_tensor.shape[2]
    original_size = (original_width, original_height)  # (width, height)

    # Convert from uint8 [0, 255] to float32 [0, 1] and resize
    transform = transforms_v2.Compose(
        [
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Resize((img_scaled_height, img_scaled_width), antialias=True),
        ]
    )

    processed_image = transform(image_tensor)  # (C, H, W) in RGB [0,1]

    # Convert RGB to YUV using Kornia
    yuv_image = kornia.color.rgb_to_yuv(processed_image.unsqueeze(0)).squeeze(0)  # (C, H, W) in YUV

    return yuv_image, original_size


def _convert_hypothesis_to_annotation(hyp: BallHypothesis, original_size: tuple[int, int]) -> Annotation:
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

    # Create annotation
    annotation = Annotation(
        shape=shape,
        text="ball",
        accuracy=None,
        color="#FF4500",  # Orange color for ball
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
    _logger.info("Processing %d images with patch-based ball detection model", len(image_paths))

    try:
        model, device = _load_model()
    except Exception as e:
        _logger.error("Failed to load model: %s", e)
        # Return empty results for all images
        return [
            VisualizationResult(image_path=Path(path).resolve(), annotations=[], result_image=None)
            for path in image_paths
        ]

    results: list[VisualizationResult] = []

    for image_path in image_paths:
        try:
            _logger.debug("Processing image: %s", image_path)

            # Preprocess image
            yuv_image, original_size = _preprocess_image(image_path)
            yuv_image = yuv_image.to(device)

            # Run ball hypothesis model
            hypotheses = run_ball_hyp_model(model, yuv_image)

            # Convert hypotheses to annotations
            annotations = []
            for hyp in hypotheses:
                # Filter out hypotheses that are outside the image bounds
                if 0 <= hyp.center_x <= original_size[0] and 0 <= hyp.center_y <= original_size[1] and hyp.diameter > 0:
                    annotation = _convert_hypothesis_to_annotation(hyp, original_size)
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
