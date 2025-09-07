import multiprocessing

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


def downsample_by_averaging(img: npt.NDArray[np.float32], scale: tuple[int, int]) -> npt.NDArray[np.float32]:
    """Downsample image by averaging blocks."""
    h, w, c = img.shape
    new_h, new_w = h // scale[0], w // scale[1]
    return img.reshape(new_h, scale[0], new_w, scale[1], c).mean(axis=(1, 3))


def load_image(path: str, label: tuple[int, int, int, int]) -> tuple[Tensor, Tensor]:
    """Load and process an image with its label using torchvision transforms, keeping CHW format."""
    # Create transforms pipeline for efficient preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((img_scaled_height, img_scaled_width), antialias=True),
            transforms.ToTensor(),  # Converts PIL to tensor, normalizes to [0,1], and converts to CHW
        ]
    )

    pil_image = Image.open(path).convert("RGB")
    image_tensor: Tensor = transform(pil_image)  # Keep in CHW format - no permutation needed!

    label_tensor = torch.tensor(label, dtype=torch.float32)

    center_x = ((label_tensor[2] + label_tensor[0]) / 2.0) / scale_factor_f
    center_y = ((label_tensor[3] + label_tensor[1]) / 2.0) / scale_factor_f

    dx = label_tensor[2] - label_tensor[0]
    dy = label_tensor[3] - label_tensor[1]
    d = torch.sqrt(dx * dx + dy * dy) / scale_factor_f

    final_label = torch.tensor([center_x, center_y, d])

    return image_tensor, final_label


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
    """Calculate random crop position bounds to keep ball visible."""
    offset = d * 0.05

    left = max(int((cx + offset - patch_width).item()), 0)
    right = max(int((cx - offset).item()), 0)
    top = max(int((cy + offset - patch_height).item()), 0)
    bottom = max(int((cy - offset).item()), 0)

    if left > right:
        left, right = right, left
    if top > bottom:
        top, bottom = bottom, top

    x = left if left == right else torch.randint(left, right + 1, (1,)).item()
    y = top if top == bottom else torch.randint(top, bottom + 1, (1,)).item()

    return int(x), int(y)


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

    start_x, start_y = _calculate_random_crop_bounds(cx, cy, d)
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


def augment_image_test_mode(image: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
    """Apply test augmentations (RGB to YUV conversion only), working with CHW format."""
    # OPTIMIZED: Direct CHW color conversion - no permutations needed!
    yuv_chw = rgb2yuv_chw_normalized(image)  # Direct CHW→CHW conversion

    return yuv_chw, label


def train_augment_image_gpu(image: Tensor, label: Tensor, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """GPU-optimized training augmentations using pure tensor operations.

    Eliminates CPU-GPU transfer overhead by keeping all operations on GPU.

    Args:
        image: RGB tensor in CHW format on any device
        label: Label tensor on any device
        device: Target device (if None, uses image.device)

    Returns:
        Augmented YUV tensor and adjusted label, both on target device
    """
    if device is None:
        device = image.device

    # Ensure tensors are on target device
    image = image.to(device)
    label = label.to(device)

    # GPU-based random flip decision (no .item() call)
    flip_prob = torch.rand(1, device=device)

    # Vectorized horizontal flip - applies flip based on condition
    if flip_prob < 0.5:
        image = torch.flip(image, dims=[2])
        # Vectorized label adjustment on GPU
        label = torch.tensor([patch_width - label[0], label[1], label[2]], device=device, dtype=label.dtype)

    # GPU-based random brightness adjustment
    brightness_factor = 1.0 + (torch.rand(1, device=device) - 0.5) * 0.3  # ±0.15 range
    image = torch.clamp(image * brightness_factor, 0, 1)

    # GPU-accelerated color conversion
    yuv_chw = rgb2yuv_chw_normalized(image)

    return yuv_chw, label


def augment_image_test_mode_gpu(
    image: Tensor, label: Tensor, device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    """GPU-optimized test augmentations (RGB to YUV conversion only).

    Args:
        image: RGB tensor in CHW format on any device
        label: Label tensor on any device
        device: Target device (if None, uses image.device)

    Returns:
        YUV tensor and label, both on target device
    """
    if device is None:
        device = image.device

    # Ensure tensors are on target device
    image = image.to(device)
    label = label.to(device)

    # GPU-accelerated color conversion
    yuv_chw = rgb2yuv_chw_normalized(image)

    return yuv_chw, label


def train_augment_batch_gpu(
    images: Tensor, labels: Tensor, device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    """GPU-optimized batch training augmentations for maximum throughput.

    Processes entire batches on GPU for optimal memory bandwidth utilization.

    Args:
        images: RGB batch tensor (N, 3, H, W) on any device
        labels: Labels batch tensor (N, 3) on any device
        device: Target device (if None, uses images.device)

    Returns:
        Augmented YUV batch and adjusted labels, both on target device
    """
    if device is None:
        device = images.device

    # Ensure tensors are on target device
    images = images.to(device)
    labels = labels.to(device)

    batch_size = images.shape[0]

    # GPU-based vectorized random flip decisions for entire batch
    flip_probs = torch.rand(batch_size, device=device)
    flip_mask = flip_probs < 0.5

    # Vectorized horizontal flip for selected samples
    images[flip_mask] = torch.flip(images[flip_mask], dims=[3])  # Flip width dimension

    # Vectorized label adjustment for flipped samples
    flipped_labels = labels.clone()
    flipped_labels[flip_mask, 0] = patch_width - labels[flip_mask, 0]
    labels = flipped_labels

    # GPU-based vectorized brightness adjustment for entire batch
    brightness_factors = 1.0 + (torch.rand(batch_size, 1, 1, 1, device=device) - 0.5) * 0.3
    images = torch.clamp(images * brightness_factors, 0, 1)

    # GPU-accelerated batch color conversion
    yuv_batch = rgb2yuv_chw_normalized(images)

    return yuv_batch, labels


def augment_batch_test_mode_gpu(
    images: Tensor, labels: Tensor, device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    """GPU-optimized batch test augmentations (RGB to YUV conversion only).

    Args:
        images: RGB batch tensor (N, 3, H, W) on any device
        labels: Labels batch tensor (N, 3) on any device
        device: Target device (if None, uses images.device)

    Returns:
        YUV batch and labels, both on target device
    """
    if device is None:
        device = images.device

    # Ensure tensors are on target device
    images = images.to(device)
    labels = labels.to(device)

    # GPU-accelerated batch color conversion
    yuv_batch = rgb2yuv_chw_normalized(images)

    return yuv_batch, labels


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
            image, label_tensor = augment_image_test_mode(image, label_tensor)

        image, label_tensor = final_adjustments(image, label_tensor)

        # Image is already in CHW format - no permutation needed!
        return image, label_tensor


class BallDatasetGPU(Dataset):
    """GPU-optimized PyTorch dataset for ball detection.

    Performs augmentations on GPU to eliminate CPU-GPU transfer overhead.
    """

    def __init__(
        self,
        images: list[str],
        labels: list[tuple[int, int, int, int]],
        trainset: bool = True,
        device: torch.device | str | None = None,
        pin_memory: bool = True,
    ) -> None:
        self.images = images
        self.labels = labels
        self.trainset = trainset
        self.pin_memory = pin_memory

        # Device management
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image_path = self.images[idx]
        label_tuple = self.labels[idx]

        # Load on CPU (image loading is I/O bound)
        image, label_tensor = load_image(image_path, label_tuple)

        # Basic preprocessing on CPU (spatial operations)
        if self.trainset:
            image, label_tensor = crop_image_random_with_ball(image, label_tensor)
        else:
            image, label_tensor = crop_image_by_image(image, label_tensor)

        # Pin memory for faster CPU-GPU transfer if requested
        if self.pin_memory:
            image = image.pin_memory()
            label_tensor = label_tensor.pin_memory()

        # GPU-based augmentations - move to GPU and process
        if self.trainset:
            image, label_tensor = train_augment_image_gpu(image, label_tensor, self.device)
        else:
            image, label_tensor = augment_image_test_mode_gpu(image, label_tensor, self.device)

        # Final adjustments can stay on GPU if scale functions support it
        image, label_tensor = final_adjustments(image, label_tensor)

        return image, label_tensor


def create_dataset(
    images: list[str], labels: list[tuple[int, int, int, int]], batch_size: int, trainset: bool = True
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Create PyTorch DataLoader for the dataset."""
    dataset: Dataset[tuple[Tensor, Tensor]] = BallDatasetGPU(images, labels, trainset)
    return DataLoader[tuple[Tensor, Tensor]](
        dataset,
        batch_size=batch_size,
        shuffle=trainset,
        num_workers=multiprocessing.cpu_count() // 2,
        pin_memory=True,
        persistent_workers=True if len(images) > 100 else False,
    )


def create_dataset_gpu(
    images: list[str],
    labels: list[tuple[int, int, int, int]],
    batch_size: int,
    trainset: bool = True,
    device: torch.device | str | None = None,
    use_batch_augmentation: bool = True,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Create GPU-optimized PyTorch DataLoader for maximum performance.

    Args:
        images: List of image paths
        labels: List of label tuples
        batch_size: Batch size for training
        trainset: Whether this is a training set (affects augmentation)
        device: Target device for GPU processing
        use_batch_augmentation: Whether to use batch-level GPU augmentation

    Returns:
        GPU-optimized DataLoader with reduced CPU-GPU transfer overhead
    """
    if use_batch_augmentation:
        # Create dataset that stops before augmentation for batch processing
        class BallDatasetPreAugment(Dataset[tuple[Tensor, Tensor]]):
            """Dataset that stops before augmentation to allow batch-level GPU processing."""

            def __init__(
                self, images: list[str], labels: list[tuple[int, int, int, int]], trainset: bool = True
            ) -> None:
                self.images = images
                self.labels = labels
                self.trainset = trainset

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
                image_path = self.images[idx]
                label_tuple = self.labels[idx]

                image, label_tensor = load_image(image_path, label_tuple)

                # Do spatial preprocessing on CPU
                if self.trainset:
                    image, label_tensor = crop_image_random_with_ball(image, label_tensor)
                else:
                    image, label_tensor = crop_image_by_image(image, label_tensor)

                # Apply final adjustments on CPU
                image, label_tensor = final_adjustments(image, label_tensor)

                # Return RGB image (not yet converted to YUV) for batch GPU processing
                return image, label_tensor

        dataset = BallDatasetPreAugment(images, labels, trainset)

        # Create custom collate function for GPU batch processing
        def gpu_batch_collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
            """Custom collate function that performs batch augmentation on GPU."""
            images_list, labels_list = zip(*batch)

            # Stack into batches on CPU first
            images_batch = torch.stack(images_list, dim=0)  # (N, 3, H, W)
            labels_batch = torch.stack(labels_list, dim=0)  # (N, 3)

            # Determine target device
            target_device = device
            if target_device is None:
                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif isinstance(target_device, str):
                target_device = torch.device(target_device)

            # Move to GPU and perform batch augmentation
            # Input: RGB images, Output: YUV images
            if trainset:
                images_batch, labels_batch = train_augment_batch_gpu(images_batch, labels_batch, target_device)
            else:
                images_batch, labels_batch = augment_batch_test_mode_gpu(images_batch, labels_batch, target_device)

            return images_batch, labels_batch

        return DataLoader[tuple[Tensor, Tensor]](
            dataset,
            batch_size=batch_size,
            shuffle=trainset,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=True,
            persistent_workers=True if len(images) > 100 else False,
            collate_fn=gpu_batch_collate_fn,
        )
    else:
        # Use individual GPU augmentation
        gpu_dataset: Dataset[tuple[Tensor, Tensor]] = BallDatasetGPU(images, labels, trainset, device, pin_memory=True)
        return DataLoader[tuple[Tensor, Tensor]](
            gpu_dataset,
            batch_size=batch_size,
            shuffle=trainset,
            num_workers=multiprocessing.cpu_count() // 4,  # Fewer workers since GPU does more work
            pin_memory=False,  # Dataset handles pinning
            persistent_workers=True if len(images) > 100 else False,
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
        num_workers=multiprocessing.cpu_count() // 2,
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
