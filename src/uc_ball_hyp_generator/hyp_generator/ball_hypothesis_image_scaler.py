"""Image scaling and caching functionality for hypothesis generator."""

from pathlib import Path

import blake3
import torch
import torchvision.transforms.v2 as transforms_v2
from torch import Tensor
from torchvision.io import ImageReadMode, decode_image

from uc_ball_hyp_generator.hyp_generator.config import scale_factor
from uc_ball_hyp_generator.utils.logger import get_logger

_logger = get_logger(__name__)


class BallHypothesisImageScaler:
    """Handles image loading, scaling, and caching for hypothesis generator."""

    def __init__(self) -> None:
        """Initialize the image scaler with cache directory."""
        self._cache_dir = Path.home() / ".cache" / "uc_ball_hyp_generator" / "scaled_ball_hypothesis_images"

    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key from image path and scaling parameters."""
        cache_input = f"{image_path}_{scale_factor}"
        return blake3.blake3(cache_input.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given cache key."""
        subdir = self._cache_dir / cache_key[:2]
        return subdir / f"{cache_key}.pt"

    def load_and_scale(self, image_path: str) -> Tensor:
        """Load image and scale it down by scale_factor with caching.

        Args:
            image_path: Path to the image file

        Returns:
            Scaled image tensor in RGB format with shape [3, scaled_height, scaled_width]
        """
        cache_key = self._get_cache_key(image_path)
        cache_path = self._get_cache_path(cache_key)

        # Try to load from cache first
        if cache_path.exists():
            try:
                cached_data = torch.load(cache_path, weights_only=False)
                return cached_data["scaled_image"]
            except (OSError, KeyError, RuntimeError):
                _logger.warning("Failed to load scaled image from cache for %s", image_path)
                pass

        # Cache miss - load and process image
        original_image = decode_image(image_path, mode=ImageReadMode.RGB)
        original_height, original_width = original_image.shape[1], original_image.shape[2]

        # Calculate scaled dimensions
        scaled_width = original_width // scale_factor
        scaled_height = original_height // scale_factor

        # Convert from uint8 [0, 255] to float32 [0, 1] and resize
        transform = transforms_v2.Compose(
            [
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Resize((scaled_height, scaled_width), antialias=True),
            ]
        )

        scaled_image = transform(original_image)

        # Save to cache
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"scaled_image": scaled_image}, cache_path)
        except OSError:
            pass

        return scaled_image
