import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as transforms  # type: ignore[import-untyped]
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import img_scaled_height, img_scaled_width, patch_height, patch_width, scale_factor_f
from scale import scale_x, scale_y


def downsample_by_averaging(img: npt.NDArray[np.float32], scale: tuple[int, int]) -> npt.NDArray[np.float32]:
    """Downsample image by averaging blocks."""
    h, w, c = img.shape
    new_h, new_w = h // scale[0], w // scale[1]
    return img.reshape(new_h, scale[0], new_w, scale[1], c).mean(axis=(1, 3))


def load_image(path: str, label: tuple[int, int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and process an image with its label using torchvision transforms."""
    # Create transforms pipeline for efficient preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((img_scaled_height, img_scaled_width), antialias=True),
            transforms.ToTensor(),  # Converts PIL to tensor, normalizes to [0,1], and converts to CHW
        ]
    )

    pil_image = Image.open(path).convert("RGB")
    image_tensor = transform(pil_image)  # Already in CHW format
    image = image_tensor.permute(1, 2, 0)  # CHW to HWC for processing

    label_tensor = torch.tensor(label, dtype=torch.float32)

    center_x = ((label_tensor[2] + label_tensor[0]) / 2.0) / scale_factor_f
    center_y = ((label_tensor[3] + label_tensor[1]) / 2.0) / scale_factor_f

    dx = label_tensor[2] - label_tensor[0]
    dy = label_tensor[3] - label_tensor[1]
    d = torch.sqrt(dx * dx + dy * dy) / scale_factor_f

    final_label = torch.tensor([center_x, center_y, d])

    return image, final_label


# Set random seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)


def crop_image_by_image(image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop image to patch containing the ball center."""
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

    image2 = image[start_y:end_y, start_x:end_x, :]
    label = torch.tensor([cx - float(start_x), cy - float(start_y), d])

    return image2, label


def _calculate_random_crop_bounds(cx: torch.Tensor, cy: torch.Tensor, d: torch.Tensor) -> tuple[int, int]:
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


def crop_image_random_with_ball(image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly crop image patch ensuring ball remains visible."""
    cx, cy, d = label[0], label[1], label[2]

    start_x, start_y = _calculate_random_crop_bounds(cx, cy, d)
    start_x, start_y, end_x, end_y = _adjust_crop_bounds(start_x, start_y)

    cropped_image = image[start_y:end_y, start_x:end_x, :]
    adjusted_label = torch.tensor([cx - float(start_x), cy - float(start_y), d])

    return cropped_image, adjusted_label


def patchify_image(image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split image into patches with adjusted labels using vectorized operations."""
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


def train_augment_image(image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply training augmentations to image and label."""
    r = torch.rand(1).item()

    if r < 0.5:
        # Horizontal flip
        image = torch.flip(image, dims=[1])
        label = torch.tensor([patch_width - label[0], label[1], label[2]])

    # Random brightness adjustment
    brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.3  # ±0.15 range
    image = torch.clamp(image * brightness_factor, 0, 1)

    a = rgb2yuv(image * 255.0)

    return a, label


def test_augment_image(image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply test augmentations (RGB to YUV conversion only)."""
    return rgb2yuv(image * 255.0), label


def final_adjustments(image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply final scaling adjustments."""
    x = scale_x(label[0] - patch_width / 2)
    y = scale_y(label[1] - patch_height / 2)

    return image / 255.0, torch.tensor([x, y, label[2]])


def final_adjustments_patches(image: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply final scaling adjustments for patch-based processing."""
    x = scale_x(labels[:, 0] - patch_width / 2)
    y = scale_y(labels[:, 1] - patch_height / 2)

    x_tensor = torch.as_tensor(x)
    y_tensor = torch.as_tensor(y)
    return image / 255.0, torch.stack([x_tensor, y_tensor, labels[:, 2]], dim=1)


class BallDataset(Dataset):
    """PyTorch dataset for ball detection."""

    def __init__(self, images: list[str], labels: list[tuple[int, int, int, int]], trainset: bool = True) -> None:
        self.images = images
        self.labels = labels
        self.trainset = trainset

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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

        # Convert to CHW format for PyTorch
        image = image.permute(2, 0, 1)

        return image, label_tensor


def create_dataset(
    images: list[str], labels: list[tuple[int, int, int, int]], batch_size: int, trainset: bool = True
) -> DataLoader:
    """Create PyTorch DataLoader for the dataset."""
    dataset = BallDataset(images, labels, trainset)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=trainset, num_workers=4 if trainset else 2, pin_memory=True
    )


class BallPatchDataset(Dataset):
    """PyTorch dataset for patch-based ball detection."""

    def __init__(self, images: list[str], labels: list[tuple[int, int, int, int]]) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[idx]
        label_tuple = self.labels[idx]

        image, label_tensor = load_image(image_path, label_tuple)
        patches, patch_labels = patchify_image(image, label_tensor)

        # Apply augmentations to all patches using vectorized operations
        # patches shape: (num_patches, patch_height, patch_width, channels)

        # Convert all patches to YUV color space in batch
        patches_scaled = patches * 255.0
        augmented_patches = rgb2yuv(patches_scaled)

        # Apply final adjustments using batch processing
        augmented_patches_normalized, final_labels = final_adjustments_patches(augmented_patches, patch_labels)

        # Convert patches to CHW format
        augmented_patches_final = augmented_patches_normalized.permute(0, 3, 1, 2)  # NHWC to NCHW

        return augmented_patches_final, final_labels


def patch_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for patch-based dataset to handle variable patch counts."""
    all_patches = []
    all_labels = []

    for patches, labels in batch:
        all_patches.append(patches)
        all_labels.append(labels)

    # Concatenate all patches and labels across batch dimension
    batched_patches = torch.cat(all_patches, dim=0)
    batched_labels = torch.cat(all_labels, dim=0)

    return batched_patches, batched_labels


def create_dataset_image_based(
    images: list[str], labels: list[tuple[int, int, int, int]], batch_size: int
) -> DataLoader:
    """Create PyTorch DataLoader for patch-based dataset."""
    dataset = BallPatchDataset(images, labels)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=patch_collate_fn
    )


class ColorSpaceConverter:
    """Efficient color space conversion with cached transformation matrices."""

    _rgb_to_yuv_cache: dict[str, torch.Tensor] = {}
    _yuv_to_rgb_cache: dict[str, torch.Tensor] = {}
    _yuv_bias_cache: dict[str, torch.Tensor] = {}
    _rgb_bias_cache: dict[str, torch.Tensor] = {}

    @classmethod
    def _get_rgb_to_yuv_matrix(cls, device: torch.device) -> torch.Tensor:
        """Get cached RGB to YUV transformation matrix."""
        device_key = str(device)
        if device_key not in cls._rgb_to_yuv_cache:
            cls._rgb_to_yuv_cache[device_key] = torch.tensor(
                [[0.299, -0.169, 0.498], [0.587, -0.331, -0.419], [0.114, 0.499, -0.0813]],
                dtype=torch.float32,
                device=device,
            )
        return cls._rgb_to_yuv_cache[device_key]

    @classmethod
    def _get_yuv_to_rgb_matrix(cls, device: torch.device) -> torch.Tensor:
        """Get cached YUV to RGB transformation matrix."""
        device_key = str(device)
        if device_key not in cls._yuv_to_rgb_cache:
            cls._yuv_to_rgb_cache[device_key] = torch.tensor(
                [
                    [1.0, 1.0, 1.0],
                    [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                    [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235],
                ],
                dtype=torch.float32,
                device=device,
            )
        return cls._yuv_to_rgb_cache[device_key]

    @classmethod
    def _get_yuv_bias(cls, device: torch.device) -> torch.Tensor:
        """Get cached YUV bias tensor."""
        device_key = str(device)
        if device_key not in cls._yuv_bias_cache:
            cls._yuv_bias_cache[device_key] = torch.tensor([0, 128, 128], device=device)
        return cls._yuv_bias_cache[device_key]

    @classmethod
    def _get_rgb_bias(cls, device: torch.device) -> torch.Tensor:
        """Get cached RGB bias tensor."""
        device_key = str(device)
        if device_key not in cls._rgb_bias_cache:
            cls._rgb_bias_cache[device_key] = torch.tensor(
                [-179.45477266423404, 135.45870971679688, -226.8183044444304], device=device
            )
        return cls._rgb_bias_cache[device_key]


def rgb2yuv(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to YUV color space, supporting batch processing."""
    device = rgb.device
    m = ColorSpaceConverter._get_rgb_to_yuv_matrix(device)

    # Handle both single images and batches of images
    original_shape = rgb.shape

    # Flatten spatial dimensions while preserving batch dimension if present
    if len(original_shape) == 4:  # Batch of images: (batch, H, W, 3)
        rgb_flat = rgb.view(-1, 3)
    else:  # Single image: (H, W, 3)
        rgb_flat = rgb.view(-1, 3)

    yuv_flat = torch.mm(rgb_flat, m.t())
    yuv = yuv_flat.view(original_shape)
    yuv += ColorSpaceConverter._get_yuv_bias(device)

    return yuv


def yuv2rgb(yuv: torch.Tensor) -> torch.Tensor:
    """Convert YUV to RGB color space."""
    device = yuv.device
    m = ColorSpaceConverter._get_yuv_to_rgb_matrix(device)

    # Reshape for matrix multiplication
    original_shape = yuv.shape
    yuv_flat = yuv.view(-1, 3)
    rgb_flat = torch.mm(yuv_flat, m.t())
    rgb = rgb_flat.view(original_shape)
    rgb += ColorSpaceConverter._get_rgb_bias(device)

    return rgb


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
    plt.close()
    plt.close()
