"""Point-Based Image Patch Dataset for PyTorch with GPU acceleration."""

import multiprocessing
from pathlib import Path

import blake3
import kornia
import torch
import torchvision.transforms.v2 as transforms_v2  # type: ignore[import-untyped]
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image  # type: ignore[import-untyped]

from uc_ball_hyp_generator.config import (
    img_scaled_height,
    img_scaled_width,
    patch_height,
    patch_width,
    path_count_h,
    path_count_w,
    scale_factor,
)
from uc_ball_hyp_generator.utils.scale_patch import scale_patch_x, scale_patch_y

# Cache directory for preprocessed tensors
_cache_dir = Path.home() / ".cache" / "uc_ball_hyp_generator" / "tensors"


def _get_cache_key(image_path: str, bbox: tuple[int, int, int, int]) -> str:
    """Generate cache key from image path and processing parameters."""
    cache_input = (
        f"{image_path}_{img_scaled_width}_{img_scaled_height}_{scale_factor}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
    )
    return blake3.blake3(cache_input.encode()).hexdigest()


def _get_cache_path(cache_key: str) -> Path:
    """Get cache file path for given cache key."""
    _cache_dir.mkdir(parents=True, exist_ok=True)
    return _cache_dir / f"{cache_key}.pt"


class BallDataset(Dataset[tuple[Tensor, Tensor]]):
    """Point-Based Image Patch Dataset with GPU acceleration."""

    def __init__(
        self,
        images: list[str],
        labels: list[tuple[int, int, int, int]],
        training: bool = True,
        device: torch.device | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            images: List of image paths in RGB format
            labels: List of bounding boxes as (x1, y1, x2, y2)
            training: Whether dataset is for training or test mode
            device: Device for computations, defaults to cuda if available
        """
        self.images = images
        self.labels = labels
        self.training = training
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._brightness_jitter = transforms_v2.ColorJitter(brightness=0.15)

        # For test mode: precompute and cache all patches
        if not training:
            self._test_patches: list[Tensor] = []
            self._test_points: list[Tensor] = []
            self._precompute_test_data()

    def _precompute_test_data(self) -> None:
        """Precompute all test patches and points for GPU caching."""
        for idx in range(len(self.images)):
            patch, point = self._process_test_sample(idx)
            self._test_patches.append(patch)
            self._test_points.append(point)

    def _process_test_sample(self, idx: int) -> tuple[Tensor, Tensor]:
        """Process single test sample without caching to GPU."""
        image_path = self.images[idx]
        bbox = self.labels[idx]

        # Load and scale image with cached center point and diameter
        image, center_x, center_y, diameter = self._load_and_scale_image(image_path, bbox)

        # Determine which patch contains the center
        patch_x = int(center_x // patch_width)
        patch_y = int(center_y // patch_height)

        # Clamp to valid patch indices
        patch_x = max(0, min(patch_x, path_count_w - 1))
        patch_y = max(0, min(patch_y, path_count_h - 1))

        # Extract patch
        start_x = patch_x * patch_width
        start_y = patch_y * patch_height
        end_x = start_x + patch_width
        end_y = start_y + patch_height

        patch = image[:, start_y:end_y, start_x:end_x]

        # Convert RGB to YUV
        patch_yuv = kornia.color.rgb_to_yuv(patch.unsqueeze(0)).squeeze(0)

        # Calculate absolute position in patch
        abs_x_in_patch = center_x - start_x
        abs_y_in_patch = center_y - start_y

        # Convert to position relative to patch center
        x_relative_to_center = abs_x_in_patch - patch_width / 2
        y_relative_to_center = abs_y_in_patch - patch_height / 2

        # Apply scaling functions
        point_x_scaled = scale_patch_x(x_relative_to_center)
        point_y_scaled = scale_patch_y(y_relative_to_center)

        point = torch.tensor([point_x_scaled, point_y_scaled, diameter], dtype=torch.float32)

        return patch_yuv, point

    def _load_and_scale_image(
        self, image_path: str, bbox: tuple[int, int, int, int]
    ) -> tuple[Tensor, float, float, float]:
        """Load image using torchvision.io.decode_image, scale down, and return with center point and diameter."""
        cache_key = _get_cache_key(image_path, bbox)
        cache_path = _get_cache_path(cache_key)

        # Try to load from cache
        if cache_path.exists():
            try:
                cached_data = torch.load(cache_path, weights_only=True)
                return (
                    cached_data["image_tensor"],
                    cached_data["center_x"],
                    cached_data["center_y"],
                    cached_data["diameter"],
                )
            except (OSError, KeyError):
                pass

        # Load and process image
        image_tensor = decode_image(image_path, mode=ImageReadMode.RGB)

        # Convert from uint8 [0, 255] to float32 [0, 1] and resize
        transform = transforms_v2.Compose(
            [
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Resize((img_scaled_height, img_scaled_width), antialias=True),
            ]
        )

        processed_image = transform(image_tensor)

        # Calculate scaled bbox and center point
        x1_scaled = float(bbox[0]) / scale_factor
        y1_scaled = float(bbox[1]) / scale_factor
        x2_scaled = float(bbox[2]) / scale_factor
        y2_scaled = float(bbox[3]) / scale_factor

        center_x = (x2_scaled + x1_scaled) / 2.0
        center_y = (y2_scaled + y1_scaled) / 2.0

        # Calculate diameter
        dx = x2_scaled - x1_scaled
        dy = y2_scaled - y1_scaled
        diameter = float(torch.sqrt(torch.tensor(dx * dx + dy * dy)).item())

        # Save to cache
        try:
            torch.save(
                {
                    "image_tensor": processed_image,
                    "scaled_bbox": (x1_scaled, y1_scaled, x2_scaled, y2_scaled),
                    "center_x": center_x,
                    "center_y": center_y,
                    "diameter": diameter,
                },
                cache_path,
            )
        except OSError:
            pass

        return processed_image, center_x, center_y, diameter

    def _get_safe_crop_bounds(
        self, center_x: float, center_y: float, bbox: tuple[int, int, int, int]
    ) -> tuple[int, int]:
        """Calculate safe crop bounds ensuring bbox stays inside patch with safety margin."""

        # Scale bbox to downscaled coordinates
        x1_scaled = bbox[0] / scale_factor
        y1_scaled = bbox[1] / scale_factor
        x2_scaled = bbox[2] / scale_factor
        y2_scaled = bbox[3] / scale_factor

        # Minimum distance ball center should stay from patch edges (configurable)
        min_center_margin_pixels = 1

        # Calculate ball center position
        center_x_scaled = (x1_scaled + x2_scaled) / 2
        center_y_scaled = (y1_scaled + y2_scaled) / 2

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
            min_start_x = max_start_x = max(
                0, min(img_scaled_width - patch_width, int(center_x_scaled - patch_width / 2))
            )
        if min_start_y > max_start_y:
            min_start_y = max_start_y = max(
                0, min(img_scaled_height - patch_height, int(center_y_scaled - patch_height / 2))
            )

        # Random crop within valid bounds
        start_x = (
            int(torch.randint(min_start_x, max_start_x + 1, (1,)).item())
            if min_start_x < max_start_x
            else int(min_start_x)
        )
        start_y = (
            int(torch.randint(min_start_y, max_start_y + 1, (1,)).item())
            if min_start_y < max_start_y
            else int(min_start_y)
        )

        return start_x, start_y

    def _process_training_sample(self, idx: int) -> tuple[Tensor, Tensor]:
        """Process training sample with GPU-accelerated random crop and augmentation."""
        image_path = self.images[idx]
        bbox = self.labels[idx]

        # Load and scale image with cached center point and diameter
        image, center_x, center_y, diameter = self._load_and_scale_image(image_path, bbox)
        # image = image.to(self.device)

        # Get safe crop bounds
        start_x, start_y = self._get_safe_crop_bounds(center_x, center_y, bbox)
        end_x = start_x + patch_width
        end_y = start_y + patch_height

        # Use simple tensor crop for now (GPU accelerated through tensor operations)
        patch = image[:, start_y:end_y, start_x:end_x]

        # Brightness augmentation
        patch = self._brightness_jitter(patch)

        # Calculate absolute position in patch
        abs_x_in_patch = center_x - start_x
        abs_y_in_patch = center_y - start_y

        # Random horizontal flip augmentation
        if torch.rand(1).item() < 0.5:
            # Horizontal flip - flip along width dimension (dim=2 for CHW format)
            patch = torch.flip(patch, dims=[2])
            # Adjust absolute position: x becomes (patch_width - x)
            abs_x_in_patch = patch_width - abs_x_in_patch

        # Convert to position relative to patch center
        x_relative_to_center = abs_x_in_patch - patch_width / 2
        y_relative_to_center = abs_y_in_patch - patch_height / 2

        # Apply scaling functions
        point_x_scaled = scale_patch_x(x_relative_to_center)
        point_y_scaled = scale_patch_y(y_relative_to_center)

        # Convert RGB to YUV using Kornia
        patch_yuv = kornia.color.rgb_to_yuv(patch.unsqueeze(0)).squeeze(0)

        point = torch.tensor([point_x_scaled, point_y_scaled, diameter], dtype=torch.float32)

        return patch_yuv, point

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get dataset item."""
        if self.training:
            return self._process_training_sample(idx)
        else:
            # Return cached test data
            return self._test_patches[idx], self._test_points[idx]


def create_dataset(
    images: list[str],
    labels: list[tuple[int, int, int, int]],
    batch_size: int,
    device: torch.device,
    trainset: bool = True,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Create PyTorch DataLoader for the dataset."""
    dataset: Dataset[tuple[Tensor, Tensor]] = BallDataset(images, labels, trainset, device)
    return DataLoader[tuple[Tensor, Tensor]](
        dataset,
        batch_size=batch_size,
        shuffle=trainset,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )


# ============================================================================
# SPECIFICATION AND IMPLEMENTATION REQUIREMENTS
# ============================================================================
"""
Point-Based Image Patch Dataset (PyTorch) – Complete Specification

1. INPUT DATA
* images: list[str] => A list of image paths in RGB format e.g. png or jpg
* labels: list[tuple[int, int, int, int]] => A bounding box for each image in format x1, y1, x2, y2

CONFIG PARAMETERS (from config.py):
```
scale_factor: int = 4
scale_factor_f: float = float(scale_factor)
path_count_w: int = 4
path_count_h: int = 4
patch_width: int = 640 // scale_factor // path_count_w
patch_height: int = 480 // scale_factor // path_count_h
img_scaled_width: int = 640 // scale_factor
img_scaled_height: int = 480 // scale_factor
```

The center of the box must be calculated as point (x, y)
The image must be loaded and downscaled by `scale_factor`

2. DATASET STRUCTURE

Training Mode:
- Random Safe Crop (GPU-accelerated)
- Crop size: patch_width×patch_height pixels
- Ball center must stay at least min_center_margin_pixels from any patch edge (configurable, default: 1 pixel)
- Cropping ensures ball center never appears at patch borders, maintaining consistent training data
- Use GPU-accelerated operations for vectorized computation
- Points adjusted relative to the crop and scaled using scale_x/scale_y functions

Training Augmentations:
- Brightness augmentation: Use torchvision.transforms.v2.ColorJitter(brightness=0.15)
- Horizontal flip augmentation: 50% probability with point coordinate adjustment

Color Conversion:
- RGB → YUV444

Output:
- Patch shape: [3, patch_height, patch_width]: YUV: [B,3,H,W]
- Fully GPU-accelerated where possible

Test Mode:
- Deterministic Crop
- Divide scaled down image into grid of patch_width×patch_height patches (path_count_w×path_count_h)
- Crop the patch where the point (x, y) would fall
- Point coordinates adjusted relative to the patch and scaled using scale_x/scale_y functions

Color Conversion:
- RGB → YUV444

Caching:
- Cache all test patches and points in GPU VRAM for fast evaluation
- No random augmentation

3. COLOR SPACE
RGB → YUV444 use GPU optimized conversion from Kornia (kornia.color.rgb_to_yuv)

4. IMPLEMENTATION REQUIREMENTS
- Use PyTorch and Kornia for GPU acceleration
- Prefer torch.randint or torch.rand for random crop computation
- Keep point coordinates normalized to [0,1] relative to patch in both modes
- Training: DataLoader can handle batch sizes >1
- Test: precompute & cache all patches and points on GPU for fastest access
- Code should be modern, readable, and maintainable, following PyTorch best practices

5. OUTPUTS
- Patch tensor: [C,H,W] in YUV444 format
- Point tensor: [x_scaled, y_scaled, diameter] where x_scaled, y_scaled are in [-1,1] range from scale_x/scale_y functions

6. INTERFACE REQUIREMENTS
- Inherit from torch.utils.data.Dataset
- Constructor signature: __init__(images, labels, training: bool = True, device: torch.device | None = None)
- __getitem__ returns: tuple[Tensor, Tensor] where second tensor is [x_scaled, y_scaled, diameter]
- Training vs test mode determined by boolean parameter
- Device parameter for explicit GPU control
- For test mode: preload ALL patches at initialization for GPU caching
- For training mode: no preloading due to augmentation

7. EDGE CASE HANDLING
- Center-based margin enforcement: ball center stays at least min_center_margin_pixels from patch edges
- When margin constraints cannot be satisfied (image too small): fallback to centering ball in patch
- Minimum center margin is configurable via min_center_margin_pixels parameter in _get_safe_crop_bounds
- User ensures image is always big enough (no special case handling needed)
- In test mode: simple grid division, use patch containing bbox center
- Wherever the center of the bbox falls is the patch to use in test mode

8. AUGMENTATION DETAILS
- Brightness augmentation: ColorJitter with brightness=0.15 (fixed, not configurable)
- Horizontal flip augmentation: 50% probability random horizontal flip with point coordinate adjustment
- Point coordinate adjustment for horizontal flip: abs_x_in_patch = patch_width - abs_x_in_patch (before scale_x)
- Applied only in training mode, not in test mode

9. ADDITIONAL CLARIFICATIONS FROM IMPLEMENTATION
- Image loading: Use torchvision.io.decode_image with ImageReadMode.RGB for direct RGB tensor loading
- Color conversion: Use kornia.color.rgb_to_yuv for GPU-accelerated conversion
- Random crop bounds: Use torch.randint for GPU-friendly random number generation
- Safe crop calculation: Ensure ball center stays at least min_center_margin_pixels from patch edges
- Center-based cropping: Calculate ball center position and ensure minimum distance from all patch boundaries
- Test mode caching: Store tensors directly on specified device for immediate access
- GPU acceleration: Leverage tensor operations on device for maximum performance
- Image processing caching: Cache downscaled images and computed center points for performance

10. IMAGE LOADING SPECIFICATION
Use torchvision.io.decode_image() for image loading:
- Function: torchvision.io.decode_image(input, mode=ImageReadMode.RGB)
- Output: Tensor[image_channels, image_height, image_width] in uint8 [0, 255]
- Supported formats: JPEG, PNG, GIF, WebP
- Mode: Use ImageReadMode.RGB for consistent 3-channel RGB output
- Post-processing: Convert from uint8 [0, 255] to float32 [0, 1] using transforms_v2.ToDtype(torch.float32, scale=True)
- Benefits: Direct tensor output, no PIL dependency, better performance, native PyTorch integration

11. CACHING SYSTEM SPECIFICATION
Image processing results are cached for performance:

Cache Key Generation:
- Use blake3 hash of: image_path, img_scaled_width, img_scaled_height, scale_factor, bbox coordinates
- Format: f"{image_path}_{img_scaled_width}_{img_scaled_height}_{scale_factor}_{x1}_{y1}_{x2}_{y2}"
- Each unique image+bbox+scale combination gets separate cache entry

Cache Storage:
- Location: Path.home() / ".cache" / "uc_ball_hyp_generator" / "tensors"
- Format: torch.save() with .pt extension
- Data structure: {
    "image_tensor": processed_image,           # Tensor[3, img_scaled_height, img_scaled_width] float32
    "scaled_bbox": (x1, y1, x2, y2),         # tuple[float, float, float, float] scaled coordinates
    "center_x": center_x,                     # float - final center position
    "center_y": center_y,                     # float - final center position
    "diameter": diameter,                     # float - bbox diagonal length scaled by scale_factor
  }

Cache Management:
- Lazy directory creation (created when first cache access needed)
- Cache validation: existence check sufficient (no timestamp validation)
- Error handling: on cache load failure, recompute and save to cache
- Method signature: _load_and_scale_image(image_path: str, bbox: tuple[int,int,int,int]) -> tuple[Tensor, float, float, float]

Performance Benefits:
- Eliminates repeated image decoding and scaling operations
- Eliminates repeated bbox center point and diameter calculations
- Shared cache between training epochs and test mode initialization
- Significant speedup for datasets with repeated image+bbox combinations

12. DIAMETER CALCULATION SPECIFICATION
Diameter represents the diagonal length of the bounding box:
- Formula: diameter = sqrt(dx² + dy²) where dx = bbox_width, dy = bbox_height
- Applied after scaling: dx and dy are calculated from scaled bbox coordinates
- Implementation: dx = x2_scaled - x1_scaled, dy = y2_scaled - y1_scaled
- Result: diameter = torch.sqrt(torch.tensor(dx * dx + dy * dy)).item()
- Units: Same as scaled image coordinates (pixels in downscaled image)
- Purpose: Provides size information for the detected object alongside position

13. POINT SCALING SPECIFICATION
Point coordinates are scaled using scale_x and scale_y functions:

Input Calculation:
- Calculate absolute position in patch: abs_x = center_x - start_x, abs_y = center_y - start_y
- Apply horizontal flip (training only): abs_x = patch_width - abs_x (if flipped)
- Convert to center-relative coordinates: x_rel = abs_x - patch_width/2, y_rel = abs_y - patch_height/2

Scaling Functions:
- Function calls: scale_x(x_rel), scale_y(y_rel)
- Input: Position relative to patch center (can be negative)
- Output range: [-1, 1] for both x and y coordinates
- Functions imported from uc_ball_hyp_generator.scale module

Processing Order:
1. Calculate absolute position in patch
2. Apply augmentations (horizontal flip affects absolute position)  
3. Convert to center-relative coordinates
4. Apply scale_x/scale_y functions
5. Create final point tensor [x_scaled, y_scaled, diameter]
"""
