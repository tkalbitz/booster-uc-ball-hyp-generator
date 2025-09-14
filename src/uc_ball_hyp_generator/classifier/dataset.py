"""Ball classifier dataset for generating positive and negative cpatch samples."""

import random
from pathlib import Path

import blake3
import kornia
import torch
import torchvision.transforms.v2 as transforms_v2  # type: ignore[import-untyped]
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image, write_jpeg  # type: ignore[import-untyped]

from uc_ball_hyp_generator.classifier.config import CLASSIFIER_DILATION_FACTOR, CPATCH_SIZE
from uc_ball_hyp_generator.hyp_generator.ball_hypothesis_image_scaler import BallHypothesisImageScaler
from uc_ball_hyp_generator.hyp_generator.config import scale_factor
from uc_ball_hyp_generator.hyp_generator.utils import create_random_hpatch, transform_hyp_output_to_original_coords


class BallClassifierDataset(Dataset[tuple[Tensor, Tensor]]):
    """Ball classifier dataset for binary classification of image patches."""

    def __init__(
        self,
        positive_images: list[str],
        positive_labels: list[tuple[int, int, int, int]],
        negative_images: list[str],
        hyp_model: Module,
        is_training: bool,
    ) -> None:
        """Initialize the ball classifier dataset.

        Args:
            positive_images: List of file paths to images containing balls
            positive_labels: List of ground-truth bounding boxes (x1, y1, x2, y2)
            negative_images: List of file paths to images confirmed to have no balls
            hyp_model: Pre-trained hypothesis generator model in eval mode
        """
        self.positive_images = positive_images
        self.positive_labels = positive_labels
        self.negative_images = negative_images
        self.is_training = is_training
        self.hyp_model = hyp_model
        self.hyp_model.eval()
        self.hyp_model_device = next(self.hyp_model.parameters()).device
        self.hyp_image_scaler = BallHypothesisImageScaler()
        self.jpg_cache_dir = Path.home() / ".cache" / "uc_ball_hyp_generator" / "jpgs"
        self.jpg_cache_dir.mkdir(parents=True, exist_ok=True)

        # Data augmentation transforms
        self._brightness_jitter = transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self._horizontal_flip = transforms_v2.RandomHorizontalFlip(p=0.5)

    def __len__(self) -> int:
        """Return dataset size with 1:3 positive to negative ratio."""
        return len(self.positive_images) * 4

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a cpatch and its label.

        Args:
            idx: Dataset index

        Returns:
            Tuple of (cpatch tensor, label tensor) where label is [1.0] for ball, [0.0] for no ball
        """
        if idx % 4 == 0:
            return self._generate_positive_sample()
        return self._generate_negative_sample()

    def _generate_positive_sample(self) -> tuple[Tensor, Tensor]:
        """Generate a positive cpatch sample using hypothesis-based approach."""
        # Randomly select a positive image and its label
        pos_idx = random.randrange(len(self.positive_images))
        image_path = self.positive_images[pos_idx]
        bbox = self.positive_labels[pos_idx]

        # Load and scale the image
        original_image, scaled_image = self._load_and_scale_image(image_path)
        original_height, original_width = original_image.shape[1], original_image.shape[2]

        # Scale bbox to match scaled image coordinates
        scaled_bbox = (
            bbox[0] / scale_factor,
            bbox[1] / scale_factor,
            bbox[2] / scale_factor,
            bbox[3] / scale_factor,
        )

        # Create hpatch that contains the ball
        hpatch, hpatch_position = create_random_hpatch(scaled_image, scaled_bbox)

        # Convert to YUV and add batch dimension for model
        hpatch_yuv = kornia.color.rgb_to_yuv(hpatch.unsqueeze(0))

        # Feed through hypothesis model
        with torch.no_grad():
            prediction = self.hyp_model(hpatch_yuv).squeeze(0)

        # Transform prediction to original image coordinates
        center_x, center_y, diameter = transform_hyp_output_to_original_coords(prediction, hpatch_position)

        # Extract cpatch from original image
        cpatch = self._extract_cpatch(original_image, center_x, center_y, diameter)

        return cpatch, torch.tensor([1.0])

    def _generate_negative_sample(self) -> tuple[Tensor, Tensor]:
        """Generate a negative cpatch sample using false positive hypothesis."""
        # Randomly select a negative image
        neg_idx = random.randrange(len(self.negative_images))
        image_path = self.negative_images[neg_idx]

        # Load and scale the image
        original_image, scaled_image = self._load_and_scale_image(image_path)
        original_height, original_width = original_image.shape[1], original_image.shape[2]

        # Extract a completely random hpatch from scaled image
        hpatch, hpatch_x, hpatch_y = self._extract_random_hpatch(scaled_image)
        hpatch_position = (hpatch_x, hpatch_y)

        # Convert to YUV and add batch dimension for model
        hpatch_yuv = kornia.color.rgb_to_yuv(hpatch.unsqueeze(0))
        hpatch = hpatch.to(self.hyp_model_device)
        # Feed through hypothesis model
        with torch.no_grad():
            prediction = self.hyp_model(hpatch_yuv).squeeze(0)

        # Transform prediction to original image coordinates
        center_x, center_y, diameter = transform_hyp_output_to_original_coords(prediction, hpatch_position)

        # Extract cpatch from original image
        cpatch = self._extract_cpatch(original_image, center_x, center_y, diameter)

        return cpatch, torch.tensor([0.0])

    def _get_jpg_cache_path(self, image_path: str) -> Path:
        """Get cache path for JPEG version of image using blake3 hash."""
        abs_path = str(Path(image_path).resolve())
        hash_obj = blake3.blake3(abs_path.encode())
        hash_hex = hash_obj.hexdigest()
        subdir = self.jpg_cache_dir / hash_hex[:2]
        return subdir / f"{hash_hex}.jpg"

    def _load_and_scale_image(self, image_path: str) -> tuple[Tensor, Tensor]:
        """Load image and return both original and scaled versions."""
        # Check if we should use cached JPEG for PNG files
        if image_path.lower().endswith(".png"):
            jpg_path = self._get_jpg_cache_path(image_path)
            if jpg_path.exists():
                # Load from JPEG cache
                original_image = decode_image(str(jpg_path), mode=ImageReadMode.RGB)
            else:
                # Load PNG, convert to JPEG and cache
                original_image = decode_image(image_path, mode=ImageReadMode.RGB)
                jpg_path.parent.mkdir(parents=True, exist_ok=True)
                write_jpeg(original_image, str(jpg_path), quality=95)
        else:
            # Load original image directly
            original_image = decode_image(image_path, mode=ImageReadMode.RGB)

        original_image = transforms_v2.ToDtype(torch.float32, scale=True)(original_image)
        scaled_image = self.hyp_image_scaler.load_and_scale(image_path)

        return original_image, scaled_image

    def _extract_random_hpatch(self, scaled_image: Tensor) -> tuple[Tensor, int, int]:
        """Extract a random hpatch from scaled image."""
        from uc_ball_hyp_generator.hyp_generator.config import patch_height, patch_width

        _, height, width = scaled_image.shape
        max_start_x = max(0, width - patch_width)
        max_start_y = max(0, height - patch_height)

        start_x = random.randint(0, max_start_x)
        start_y = random.randint(0, max_start_y)

        end_x = start_x + patch_width
        end_y = start_y + patch_height

        return scaled_image[:, start_y:end_y, start_x:end_x], start_x, start_y

    def _extract_cpatch(self, original_image: Tensor, center_x: float, center_y: float, diameter: float) -> Tensor:
        """Extract and process cpatch from original image."""
        # Calculate crop size with 1.2x diameter
        crop_size = max(1, int(diameter * CLASSIFIER_DILATION_FACTOR))
        half_crop = crop_size // 2

        # Calculate crop bounds
        x1 = int(center_x - half_crop)
        y1 = int(center_y - half_crop)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # Clamp to image boundaries
        _, img_height, img_width = original_image.shape
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(x1 + 1, min(x2, img_width))
        y2 = max(y1 + 1, min(y2, img_height))

        # Extract crop
        crop = original_image[:, y1:y2, x1:x2]

        # Resize to CPATCH_SIZE
        resize_transform = transforms_v2.Resize((CPATCH_SIZE, CPATCH_SIZE), antialias=True)
        crop_resized = resize_transform(crop)

        # Apply data augmentation
        if self.is_training:
            crop_resized = self._brightness_jitter(crop_resized)
            crop_resized = self._horizontal_flip(crop_resized)

        # Convert to YUV
        cpatch_yuv = kornia.color.rgb_to_yuv(crop_resized)

        return cpatch_yuv
