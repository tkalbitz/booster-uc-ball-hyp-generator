"""Ball detection adapter for naoteamhtwk_machinelearning_visualizer.

This adapter loads a trained PyTorch ball detection model and runs inference on images,
returning VisualizationResult objects with ball detection annotations using EllipseShape.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from naoteamhtwk_machinelearning_visualizer.core.shapes import Annotation, EllipseShape, Point, VisualizationResult
from PIL import Image

import uc_ball_hyp_generator.models as models
from uc_ball_hyp_generator.config import patch_height, patch_width
from uc_ball_hyp_generator.scale import unscale_x, unscale_y

_logger = logging.getLogger(__name__)

# Global model instance to avoid reloading
_model = None
_device = None
_model_path = None


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

        _model = models.create_network_v2(patch_height, patch_width)
        
        # Load state dict and handle torch.compile prefixes
        state_dict = torch.load(current_model_path, map_location=_device)
        
        # Check if this is a compiled model (has _orig_mod. prefixes)
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            _logger.info("Detected compiled model, removing _orig_mod. prefixes")
            # Remove _orig_mod. prefixes from compiled model
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("_orig_mod."):
                    cleaned_key = key[len("_orig_mod."):]
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


def _preprocess_image(image_path: str) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load and preprocess image for model inference."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    # Resize to model input size
    image_resized = image.resize((patch_width, patch_height), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize
    image_array = np.array(image_resized, dtype=np.float32) / 255.0

    # Convert RGB to YUV (as expected by the model)
    image_yuv = _rgb_to_yuv(image_array)

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image_yuv).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    return image_tensor, original_size


def _rgb_to_yuv(rgb_image: np.ndarray) -> np.ndarray:
    """Convert RGB image to YUV color space."""
    # YUV conversion matrix
    rgb_to_yuv_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001]
    ])

    # Reshape image for matrix multiplication
    h, w, c = rgb_image.shape
    rgb_flat = rgb_image.reshape(-1, 3)

    # Apply conversion
    yuv_flat = rgb_flat @ rgb_to_yuv_matrix.T

    # Reshape back and normalize YUV
    yuv_image = yuv_flat.reshape(h, w, 3)
    yuv_image[:, :, 0] = yuv_image[:, :, 0]  # Y channel (0-1)
    yuv_image[:, :, 1] = yuv_image[:, :, 1] + 0.5  # U channel (-0.5 to 0.5) -> (0-1)
    yuv_image[:, :, 2] = yuv_image[:, :, 2] + 0.5  # V channel (-0.5 to 0.5) -> (0-1)

    return yuv_image.astype(np.float32)


def _postprocess_prediction(prediction: torch.Tensor, original_size: tuple[int, int]) -> list[Annotation]:
    """Convert model prediction to visualization annotations."""
    annotations = []

    # Extract prediction values
    pred_x = prediction[0].item()
    pred_y = prediction[1].item()

    # Unscale the coordinates
    ball_x_patch = float(unscale_x(pred_x))
    ball_y_patch = float(unscale_y(pred_y))

    # Convert from patch coordinates to relative coordinates (0-1)
    # The unscale functions return patch coordinates relative to patch center
    rel_x = (ball_x_patch + patch_width / 2) / patch_width
    rel_y = (ball_y_patch + patch_height / 2) / patch_height

    # Clamp to valid range
    rel_x = max(0.0, min(1.0, rel_x))
    rel_y = max(0.0, min(1.0, rel_y))

    # Define ellipse size (relative to image size)
    ellipse_width = 0.02  # 2% of image width
    ellipse_height = 0.02  # 2% of image height

    # Create ellipse shape centered at predicted position
    center = Point(x=rel_x, y=rel_y)
    shape = EllipseShape(center=center, width=ellipse_width, height=ellipse_height)

    # Create annotation
    annotation = Annotation(
        shape=shape,
        text="ball",
        accuracy=None,  # This model doesn't provide confidence scores
        color="#FF4500",  # Orange color for ball
        outline_color="#FFFFFF"  # White outline
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
    _logger.info("Processing %d images with ball detection model", len(image_paths))

    try:
        model, device = _load_model()
    except Exception as e:
        _logger.error("Failed to load model: %s", e)
        # Return empty results for all images
        return [
            VisualizationResult(image_path=Path(path).resolve(), annotations=[], result_image=None)
            for path in image_paths
        ]

    results = []

    for image_path in image_paths:
        try:
            _logger.debug("Processing image: %s", image_path)

            # Preprocess image
            input_tensor, original_size = _preprocess_image(image_path)
            input_tensor = input_tensor.to(device)

            # Run inference
            with torch.no_grad():
                prediction = model(input_tensor)

            # Postprocess prediction
            annotations = _postprocess_prediction(prediction[0], original_size)

            result = VisualizationResult(
                image_path=Path(image_path).resolve(),
                annotations=annotations,
                result_image=None
            )
            results.append(result)

        except Exception as e:
            _logger.error("Failed to process %s: %s", image_path, e)
            # Create empty result for failed image
            result = VisualizationResult(
                image_path=Path(image_path).resolve(),
                annotations=[],
                result_image=None
            )
            results.append(result)

    _logger.info("Successfully processed %d images", len(results))
    return results


def get_adapter_info() -> dict[str, str]:
    """Get information about this ball detection adapter.

    Returns:
        Dictionary containing adapter metadata
    """
    return {
        "name": "UC Ball Detection",
        "version": "1.0.0",
        "description": "Ball detection using trained PyTorch model for UC ball hypothesis generation",
        "supported_formats": "JPEG, PNG, and other PIL-supported formats",
        "model_type": "NetworkV2 (Custom CNN)",
        "classes": "ball",
        "requirements": "torch>=2.8.0, PIL, numpy",
        "environment": "Set BALL_MODEL_PATH to point to the .pth model file"
    }