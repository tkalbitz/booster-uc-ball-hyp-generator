"""Ball detection adapter for naoteamhtwk_machinelearning_visualizer.

This adapter loads a trained PyTorch ball detection model and runs inference on images,
splitting them into patches as during training and returning VisualizationResult objects
with ball detection annotations using EllipseShape.
"""

import logging
import os
from dataclasses import dataclass
from math import tanh
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
    scale_factor_f,
)
from uc_ball_hyp_generator.hyp_generator.model import get_ball_hyp_model
from uc_ball_hyp_generator.hyp_generator.scale_patch import unscale_patch_x, unscale_patch_y, unscale_radius

_logger = logging.getLogger(__name__)

# Global model instance to avoid reloading
_model = None
_device = None
_model_path = None


@dataclass
class PreprocessedImage:
    """Container for preprocessed image data with patches and metadata."""

    patch_batch: torch.Tensor  # (num_patches, C, H, W) tensor of all image patches
    original_size: tuple[int, int]  # (width, height) of original image
    patch_positions: list[tuple[int, int]]  # List of (start_x, start_y) for each patch


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
        _logger.info("Loading ball detection model from: %s", current_model_path)

        _device = torch.device("cpu")
        if torch.cuda.is_available():
            _device = torch.device("cuda")

        _model = get_ball_hyp_model(patch_height, patch_width)

        # Load state dict and handle torch.compile prefixes
        state_dict = torch.load(current_model_path, map_location=_device)

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

        _model.load_state_dict(state_dict)
        _model.to(_device)
        _model.eval()
        _model_path = current_model_path

        _logger.info("Model loaded successfully on device: %s", _device)

    return _model, _device


def _preprocess_image(image_path: str) -> PreprocessedImage:
    """Load and preprocess image for model inference, splitting into patches."""
    # Load image using torchvision.io.decode_image as in dataset_handling.py
    image_tensor = decode_image(image_path, mode=ImageReadMode.RGB)

    # Get original size before scaling
    original_height, original_width = image_tensor.shape[1], image_tensor.shape[2]
    original_size = (original_width, original_height)  # (width, height)

    # Convert from uint8 [0, 255] to float32 [0, 1] and resize as in dataset_handling.py
    transform = transforms_v2.Compose(
        [
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Resize((img_scaled_height, img_scaled_width), antialias=True),
        ]
    )

    processed_image = transform(image_tensor)  # (C, H, W)

    # Convert RGB to YUV using Kornia as in dataset_handling.py
    yuv_tensor = kornia.color.rgb_to_yuv(processed_image.unsqueeze(0)).squeeze(0)  # (C, H, W)

    # Split image into patches
    patches_y = img_scaled_height // patch_height
    patches_x = img_scaled_width // patch_width

    patches: list[torch.Tensor] = []
    patch_positions: list[tuple[int, int]] = []

    for py in range(patches_y):
        for px in range(patches_x):
            start_y = py * patch_height
            end_y = start_y + patch_height
            start_x = px * patch_width
            end_x = start_x + patch_width

            # Extract patch from CHW tensor
            patch = yuv_tensor[:, start_y:end_y, start_x:end_x]  # (C, H, W)
            patches.append(patch)

            # Store patch position for coordinate conversion
            patch_positions.append((start_x, start_y))

    # Stack all patches into a batch
    patch_batch = torch.stack(patches)  # (num_patches, C, H, W)

    return PreprocessedImage(patch_batch=patch_batch, original_size=original_size, patch_positions=patch_positions)


def _postprocess_predictions(predictions: torch.Tensor, preprocessed: PreprocessedImage) -> list[Annotation]:
    """Convert model predictions to visualization annotations."""
    annotations: list[Annotation] = []

    _logger.info("Original size %d/%d.", preprocessed.original_size[0], preprocessed.original_size[1])

    orig_size_x, orig_size_y = preprocessed.original_size[0], preprocessed.original_size[1]

    # Filter predictions to only include likely ball detections
    # We'll use a threshold approach - only show patches where the model is confident
    for i, (pred_x, pred_y, pred_r) in enumerate(predictions):
        patch_start_x, patch_start_y = preprocessed.patch_positions[i]
        # Unscale the coordinates (convert from model output space to patch coordinates)
        ball_x_patch = float(unscale_patch_x(tanh(pred_x.item())))
        ball_y_patch = float(unscale_patch_y(tanh(pred_y.item())))
        ball_r = float(unscale_radius(torch.tanh(pred_r)))

        # Convert from patch-local coordinates to scaled image coordinates
        ball_x_scaled = ball_x_patch + patch_width / 2 + patch_start_x
        ball_y_scaled = ball_y_patch + patch_height / 2 + patch_start_y
        ball_d = ball_r * 2

        # Convert from scaled image coordinates to original image coordinates
        ball_x_orig = ball_x_scaled * scale_factor_f
        ball_y_orig = ball_y_scaled * scale_factor_f
        ball_d_orig = ball_d * scale_factor_f

        # Convert to relative coordinates (0-1) for the visualizer
        rel_x = ball_x_orig / preprocessed.original_size[0]  # width
        rel_y = ball_y_orig / preprocessed.original_size[1]  # height

        # Clamp to valid range
        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))

        # Only add annotation if the predicted position is within the patch bounds
        # This helps filter out patches without balls
        if 0 <= ball_x_orig <= orig_size_x and 0 <= ball_y_orig <= orig_size_y:
            # Define ellipse size (relative to image size)
            ellipse_width = ball_d_orig / preprocessed.original_size[0]
            ellipse_height = ball_d_orig / preprocessed.original_size[1]

            # Create ellipse shape centered at predicted position
            center = Point(x=rel_x, y=rel_y)
            shape = EllipseShape(center=center, width=ellipse_width, height=ellipse_height)

            # Create annotation with confidence-based coloring
            # Use different colors for patches to help debugging if needed
            annotation = Annotation(
                shape=shape,
                text=f"ball_p[{i // 4}:{i % 4}]",  # Include patch index for debugging
                accuracy=None,  # This model doesn't provide confidence scores
                color="#FF4500",  # Orange color for ball
                outline_color="#FFFFFF",  # White outline
            )

            annotations.append(annotation)

    return annotations


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

            # Preprocess image into patches
            preprocessed = _preprocess_image(image_path)
            patch_batch = preprocessed.patch_batch.to(device)

            # Run inference on all patches
            with torch.no_grad():
                predictions = model(patch_batch)

            # Postprocess predictions to annotations
            annotations = _postprocess_predictions(predictions, preprocessed)

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
