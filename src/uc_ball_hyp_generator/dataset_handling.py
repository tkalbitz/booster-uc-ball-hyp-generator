import multiprocessing
from pathlib import Path

import blake3
import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as transforms  # type: ignore[import-untyped]
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from uc_ball_hyp_generator.color_conversion import rgb2yuv_chw_normalized
from uc_ball_hyp_generator.config import img_scaled_height, img_scaled_width, patch_height, patch_width, scale_factor_f
from uc_ball_hyp_generator.scale import scale_x, scale_y

# Cache for pre-calculated crop regions to optimize repeated random cropping
type CropRegion = tuple[int, int, int, int]  # x_min, x_max, y_min, y_max
_crop_region_cache: dict[tuple[float, float, float], CropRegion] = {}

# Cache directory for preprocessed tensors
_cache_dir = Path.home() / ".cache" / "uc_ball_hyp_generator" / "tensors"


def _get_cache_key(image_path: str, img_height: int, img_width: int, label: tuple[int, int, int, int]) -> str:
    """Generate cache key from image path, processing parameters, and label."""
    cache_input = f"{image_path}_{img_height}_{img_width}_{label[0]}_{label[1]}_{label[2]}_{label[3]}"
    return blake3.blake3(cache_input.encode()).hexdigest()


def _get_cache_path(cache_key: str) -> Path:
    """Get cache file path for given cache key."""
    _cache_dir.mkdir(parents=True, exist_ok=True)
    return _cache_dir / f"{cache_key}.pt"


def _is_cache_valid(cache_path: Path, source_path: str) -> bool:
    """Check if cached tensor is newer than source image."""
    if not cache_path.exists():
        return False

    source_mtime = Path(source_path).stat().st_mtime
    cache_mtime = cache_path.stat().st_mtime
    return cache_mtime > source_mtime


def downsample_by_averaging(img: npt.NDArray[np.float32], scale: tuple[int, int]) -> npt.NDArray[np.float32]:
    """Downsample image by averaging blocks."""
    h, w, c = img.shape
    new_h, new_w = h // scale[0], w // scale[1]
    return img.reshape(new_h, scale[0], new_w, scale[1], c).mean(axis=(1, 3))


def load_image(path: str, label: tuple[int, int, int, int]) -> tuple[Tensor, Tensor]:
    """Load and process an image with its label using torchvision transforms, keeping CHW format."""
    cache_key = _get_cache_key(path, img_scaled_height, img_scaled_width, label)
    cache_path = _get_cache_path(cache_key)

    if _is_cache_valid(cache_path, path):
        try:
            cached_data = torch.load(cache_path, weights_only=True)
            return cached_data["image"], cached_data["processed_label"]
        except (OSError, KeyError):
            pass

    # Create transforms pipeline for efficient preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((img_scaled_height, img_scaled_width), antialias=True),
            transforms.ToTensor(),  # Converts PIL to tensor, normalizes to [0,1], and converts to CHW
        ]
    )

    pil_image = Image.open(path).convert("RGB")
    processed_image: Tensor = transform(pil_image)  # Keep in CHW format - no permutation needed!

    label_tensor = torch.tensor(label, dtype=torch.float32)
    center_x = ((label_tensor[2] + label_tensor[0]) / 2.0) / scale_factor_f
    center_y = ((label_tensor[3] + label_tensor[1]) / 2.0) / scale_factor_f
    dx = label_tensor[2] - label_tensor[0]
    dy = label_tensor[3] - label_tensor[1]
    d = torch.sqrt(dx * dx + dy * dy) / scale_factor_f
    final_label = torch.tensor([center_x, center_y, d])

    try:
        torch.save({"image": processed_image, "processed_label": final_label}, cache_path)
    except OSError:
        pass

    return processed_image, final_label


def crop_image_by_image(image: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
    """Crop image to patch containing the ball center, working with CHW format."""
    cx = label[0]
    cy = label[1]
    d = label[2]

    patches_x = img_scaled_width // patch_width
    patches_y = img_scaled_height // patch_height

    patch_x = torch.clamp(cx // patch_width, 0, patches_x - 1)
    patch_y = torch.clamp(cy // patch_height, 0, patches_y - 1)

    start_x = int(patch_x * patch_width)
    start_y = int(patch_y * patch_height)
    end_x = int(start_x + patch_width)
    end_y = int(start_y + patch_height)

    # Work with CHW format: image shape is (C, H, W)
    cropped_image = image[:, start_y:end_y, start_x:end_x]
    adjusted_label = torch.tensor([cx - float(start_x), cy - float(start_y), d])

    return cropped_image, adjusted_label


def _calculate_random_crop_bounds(cx: Tensor, cy: Tensor, d: Tensor) -> tuple[int, int]:
    """Calculate random crop position bounds to keep ball visible with optimized computation."""
    # Pre-calculate offset and convert tensors to float once
    offset = (d * 0.05).item()
    cx_val = cx.item()
    cy_val = cy.item()

    # Calculate bounds with single operations and proper ordering
    x_min = max(int(cx_val + offset - patch_width), 0)
    x_max = max(int(cx_val - offset), 0)
    y_min = max(int(cy_val + offset - patch_height), 0)
    y_max = max(int(cy_val - offset), 0)

    # Ensure proper ordering (min <= max)
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    # Generate random coordinates with single torch operations when needed
    if x_min == x_max:
        x = x_min
    else:
        x = int(torch.randint(x_min, x_max + 1, (1,), dtype=torch.int32).item())

    if y_min == y_max:
        y = y_min
    else:
        y = int(torch.randint(y_min, y_max + 1, (1,), dtype=torch.int32).item())

    return x, y


def clear_crop_region_cache() -> None:
    """Clear the crop region cache to free memory."""
    global _crop_region_cache
    _crop_region_cache.clear()


def _get_or_calculate_crop_region(cx: float, cy: float, d: float) -> CropRegion:
    """Get pre-calculated crop region or calculate and cache it."""
    cache_key = (cx, cy, d)

    if cache_key in _crop_region_cache:
        return _crop_region_cache[cache_key]

    # Pre-calculate valid crop region bounds
    offset = d * 0.05

    x_min = max(int(cx + offset - patch_width), 0)
    x_max = max(int(cx - offset), 0)
    y_min = max(int(cy + offset - patch_height), 0)
    y_max = max(int(cy - offset), 0)

    # Ensure proper ordering
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    crop_region: CropRegion = (x_min, x_max, y_min, y_max)
    _crop_region_cache[cache_key] = crop_region

    return crop_region


def _calculate_random_crop_bounds_optimized(cx: Tensor, cy: Tensor, d: Tensor) -> tuple[int, int]:
    """Optimized version using pre-calculated crop regions."""
    cx_val = cx.item()
    cy_val = cy.item()
    d_val = d.item()

    x_min, x_max, y_min, y_max = _get_or_calculate_crop_region(cx_val, cy_val, d_val)

    # Generate random coordinates with minimal operations
    x = x_min if x_min == x_max else int(torch.randint(x_min, x_max + 1, (1,), dtype=torch.int32).item())
    y = y_min if y_min == y_max else int(torch.randint(y_min, y_max + 1, (1,), dtype=torch.int32).item())

    return x, y


def _adjust_crop_bounds(start_x: int, start_y: int) -> tuple[int, int, int, int]:
    """Adjust crop bounds to fit within image dimensions."""
    end_x = start_x + patch_width
    end_y = start_y + patch_height

    if end_x >= img_scaled_width:
        end_x = img_scaled_width
        start_x = end_x - patch_width

    if end_y >= img_scaled_height:
        end_y = img_scaled_height
        start_y = end_y - patch_height

    return start_x, start_y, end_x, end_y


def crop_image_random_with_ball(image: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
    """Randomly crop image patch ensuring ball remains visible, working with CHW format."""
    cx, cy, d = label[0], label[1], label[2]

    start_x, start_y = _calculate_random_crop_bounds_optimized(cx, cy, d)
    start_x, start_y, end_x, end_y = _adjust_crop_bounds(start_x, start_y)

    # Work with CHW format: image shape is (C, H, W)
    cropped_image = image[:, start_y:end_y, start_x:end_x]
    adjusted_label = torch.tensor([cx - float(start_x), cy - float(start_y), d])

    return cropped_image, adjusted_label


def patchify_image(image: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
    """Split image into patches with adjusted labels using vectorized operations.

    LEGACY function for HWC processing - kept for backward compatibility.
    New code should use patchify_image_chw() for better performance.
    """
    cx, cy, d = label[0], label[1], label[2]

    patches_y = image.shape[0] // patch_height
    patches_x = image.shape[1] // patch_width

    # Use unfold to extract all patches in a vectorized way
    # unfold(dimension, size, step) extracts sliding windows
    patches = image.unfold(0, patch_height, patch_height).unfold(1, patch_width, patch_width)
    # Reshape from (patches_y, patches_x, channels, patch_height, patch_width)
    # to (num_patches, patch_height, patch_width, channels)
    patches = patches.permute(0, 1, 3, 4, 2).contiguous().view(-1, patch_height, patch_width, image.shape[2])

    # Create coordinate grids for vectorized label computation
    y_coords = torch.arange(patches_y, dtype=torch.float32) * patch_height
    x_coords = torch.arange(patches_x, dtype=torch.float32) * patch_width

    # Create meshgrid and flatten to match patch order
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # Vectorized label adjustment
    adjusted_cx = cx - xx_flat
    adjusted_cy = cy - yy_flat
    d_repeated = d.expand(len(xx_flat))

    labels = torch.stack([adjusted_cx, adjusted_cy, d_repeated], dim=1)

    return patches, labels


def patchify_image_chw(image: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
    """OPTIMIZED: Split CHW image into NCHW patches, eliminating permutations.

    Args:
        image: CHW format tensor (C, H, W)
        label: Label tensor (cx, cy, d)

    Returns:
        patches: NCHW format tensor (num_patches, C, patch_H, patch_W)
        labels: Adjusted labels tensor (num_patches, 3)
    """
    cx, cy, d = label[0], label[1], label[2]

    # Image is in CHW format: (C, H, W)
    patches_y = image.shape[1] // patch_height  # H dimension
    patches_x = image.shape[2] // patch_width  # W dimension

    # Use unfold to extract patches in CHW format
    # unfold on H dimension (dim=1), then W dimension (dim=2)
    patches = image.unfold(1, patch_height, patch_height).unfold(2, patch_width, patch_width)
    # patches shape: (C, patches_y, patches_x, patch_height, patch_width)

    # Reshape to (num_patches, C, patch_height, patch_width) - NCHW format
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, image.shape[0], patch_height, patch_width)

    # Create coordinate grids for vectorized label computation
    y_coords = torch.arange(patches_y, dtype=torch.float32) * patch_height
    x_coords = torch.arange(patches_x, dtype=torch.float32) * patch_width

    # Create meshgrid and flatten to match patch order
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # Vectorized label adjustment
    adjusted_cx = cx - xx_flat
    adjusted_cy = cy - yy_flat
    d_repeated = d.expand(len(xx_flat))

    labels = torch.stack([adjusted_cx, adjusted_cy, d_repeated], dim=1)

    return patches, labels


def train_augment_image(image: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
    """Apply training augmentations to image and label, working with CHW format."""
    r = torch.rand(1).item()

    if r < 0.5:
        # Horizontal flip - flip along width dimension (dim=2 for CHW)
        image = torch.flip(image, dims=[2])
        label = torch.tensor([patch_width - label[0], label[1], label[2]])

    # Random brightness adjustment
    brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.3  # ±0.15 range
    image = torch.clamp(image * brightness_factor, 0, 1)

    # OPTIMIZED: Direct CHW color conversion - no permutations needed!
    yuv_chw = rgb2yuv_chw_normalized(image)  # Direct CHW→CHW conversion

    return yuv_chw, label


def test_augment_image(image: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
    """Apply test augmentations (RGB to YUV conversion only), working with CHW format."""
    # OPTIMIZED: Direct CHW color conversion - no permutations needed!
    yuv_chw = rgb2yuv_chw_normalized(image)  # Direct CHW→CHW conversion

    return yuv_chw, label


def final_adjustments(image: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
    """Apply final scaling adjustments, keeping CHW format."""
    x = scale_x(label[0] - patch_width / 2)
    y = scale_y(label[1] - patch_height / 2)

    return image, torch.tensor([x, y, label[2]])  # No scaling needed - already in [0,1] range


def final_adjustments_patches(image: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Apply final scaling adjustments for patch-based processing."""
    x = scale_x(labels[:, 0] - patch_width / 2)
    y = scale_y(labels[:, 1] - patch_height / 2)

    x_tensor = torch.as_tensor(x)
    y_tensor = torch.as_tensor(y)
    return image, torch.stack([x_tensor, y_tensor, labels[:, 2]], dim=1)  # No scaling needed - already in [0,1] range


class BallDataset(Dataset):
    """PyTorch dataset for ball detection."""

    def __init__(self, images: list[str], labels: list[tuple[int, int, int, int]], trainset: bool = True) -> None:
        self.images = images
        self.labels = labels
        self.trainset = trainset

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image_path = self.images[idx]
        label_tuple = self.labels[idx]

        image, label_tensor = load_image(image_path, label_tuple)

        if self.trainset:
            image, label_tensor = crop_image_random_with_ball(image, label_tensor)
            image, label_tensor = train_augment_image(image, label_tensor)
        else:
            image, label_tensor = crop_image_by_image(image, label_tensor)
            image, label_tensor = test_augment_image(image, label_tensor)

        image, label_tensor = final_adjustments(image, label_tensor)

        # Image is already in CHW format - no permutation needed!
        return image, label_tensor


def create_dataset(
    images: list[str], labels: list[tuple[int, int, int, int]], batch_size: int, trainset: bool = True
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Create PyTorch DataLoader for the dataset."""
    dataset: Dataset[tuple[Tensor, Tensor]] = BallDataset(images, labels, trainset)
    return DataLoader[tuple[Tensor, Tensor]](
        dataset,
        batch_size=batch_size,
        shuffle=trainset,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )


class BallPatchDataset(Dataset[tuple[Tensor, Tensor]]):
    """PyTorch dataset for patch-based ball detection."""

    def __init__(self, images: list[str], labels: list[tuple[int, int, int, int]]) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image_path = self.images[idx]
        label_tuple = self.labels[idx]

        image, label_tensor = load_image(image_path, label_tuple)

        # OPTIMIZED: Direct CHW patchification - no permutations!
        patches, patch_labels = patchify_image_chw(image, label_tensor)  # CHW → NCHW

        # OPTIMIZED: Direct CHW color conversion - no permutations!
        augmented_patches = rgb2yuv_chw_normalized(patches)  # NCHW → NCHW conversion

        # Apply final adjustments using batch processing
        augmented_patches_normalized, final_labels = final_adjustments_patches(augmented_patches, patch_labels)

        # Already in NCHW format - no permutation needed!
        return augmented_patches_normalized, final_labels


def patch_collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Custom collate function for patch-based dataset to handle variable patch counts."""
    all_patches: list[Tensor] = []
    all_labels: list[Tensor] = []

    for patches, labels in batch:
        all_patches.append(patches)
        all_labels.append(labels)

    # Concatenate all patches and labels across batch dimension
    batched_patches = torch.cat(all_patches, dim=0)
    batched_labels = torch.cat(all_labels, dim=0)

    return batched_patches, batched_labels


def create_dataset_image_based(
    images: list[str], labels: list[tuple[int, int, int, int]], batch_size: int
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Create PyTorch DataLoader for patch-based dataset."""
    dataset = BallPatchDataset(images, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        collate_fn=patch_collate_fn,
        persistent_workers=True if len(images) > 100 else False,
    )


# Color conversion functions moved to uc_ball_hyp_generator.color_conversion module
# Import rgb2yuv_normalized, rgb2yuv_255, etc. from there for color space conversions


def show_dataset(ds: DataLoader) -> None:
    """Visualize dataset samples."""
    image_batch, label_batch = next(iter(ds))

    plt.figure(figsize=(30, 30))
    for i in range(min(9, len(image_batch))):
        plt.subplot(3, 3, i + 1)

        # Convert CHW to HWC for display
        image = image_batch[i].permute(1, 2, 0).numpy()
        plt.imshow(image)

        label = label_batch[i]
        plt.gca().add_patch(
            Ellipse((label[0], label[1]), label[2], label[2], linewidth=1, edgecolor="r", facecolor="none")
        )
        plt.axis("off")

    plt.waitforbuttonpress()
    plt.close()
