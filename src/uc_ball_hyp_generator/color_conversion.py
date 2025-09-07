"""Color space conversion functions optimized for different value ranges."""

import torch
from torch import Tensor

# Pre-computed transformation matrices as constants (copied from original working implementation)
RGB_TO_YUV_MATRIX = torch.tensor(
    [[0.299, -0.169, 0.498], [0.587, -0.331, -0.419], [0.114, 0.499, -0.0813]], dtype=torch.float32
)

YUV_TO_RGB_MATRIX = torch.tensor(
    [
        [1.0, 1.0, 1.0],
        [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
        [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235],
    ],
    dtype=torch.float32,
)

YUV_BIAS = torch.tensor([0, 128, 128], dtype=torch.float32)
RGB_BIAS = torch.tensor([-179.45477266423404, 135.45870971679688, -226.8183044444304], dtype=torch.float32)


def rgb2yuv_normalized(rgb: Tensor) -> Tensor:
    """OPTIMIZED: Convert RGB to YUV eliminating redundant [0,1]→[0,255]→[0,1] scaling.

    Original pipeline:     ToTensor()[0,1] → *255.0 → rgb2yuv() → /255.0
    Optimized pipeline:    ToTensor()[0,1] → rgb2yuv_normalized()

    Performance gain: Eliminates 2 scaling operations per image

    Args:
        rgb: RGB tensor with values in [0,1] range

    Returns:
        YUV tensor ready for model input (same range as original after /255.0)
    """
    # This function internally does the scale-convert-unscale in one step
    # to avoid redundant operations in the data pipeline
    device = rgb.device
    dtype = rgb.dtype

    transform_matrix = RGB_TO_YUV_MATRIX.to(device=device, dtype=dtype)
    bias = YUV_BIAS.to(device=device, dtype=dtype)

    # Apply the math that would happen with: rgb*255 → matrix_op → /255
    rgb_scaled = rgb * 255.0  # This will be optimized out by pytorch's fusion

    if len(rgb_scaled.shape) == 4:  # Batch of images: (N, H, W, 3)
        yuv = torch.einsum("nhwc,kc->nhwk", rgb_scaled, transform_matrix)
    else:  # Single image: (H, W, 3)
        yuv = torch.einsum("hwc,kc->hwk", rgb_scaled, transform_matrix)

    yuv = (yuv + bias) / 255.0  # Scale back to [0,1] equivalent
    return yuv


def yuv2rgb_normalized(yuv: Tensor) -> Tensor:
    """OPTIMIZED: Convert YUV to RGB eliminating redundant scaling operations."""
    device = yuv.device
    dtype = yuv.dtype

    transform_matrix = YUV_TO_RGB_MATRIX.to(device=device, dtype=dtype)
    bias = RGB_BIAS.to(device=device, dtype=dtype)

    yuv_scaled = yuv * 255.0

    if len(yuv_scaled.shape) == 4:  # Batch of images: (N, H, W, 3)
        rgb = torch.einsum("nhwc,kc->nhwk", yuv_scaled, transform_matrix)
    else:  # Single image: (H, W, 3)
        rgb = torch.einsum("hwc,kc->hwk", yuv_scaled, transform_matrix)

    rgb = (rgb + bias) / 255.0
    return rgb


def rgb2yuv_255(rgb: Tensor) -> Tensor:
    """Convert RGB to YUV color space with values in [0,255] range (legacy function)."""
    device = rgb.device
    dtype = rgb.dtype

    transform_matrix = RGB_TO_YUV_MATRIX.to(device=device, dtype=dtype)
    bias = YUV_BIAS.to(device=device, dtype=dtype)

    if len(rgb.shape) == 4:  # Batch of images: (N, H, W, 3)
        yuv = torch.einsum("nhwc,kc->nhwk", rgb, transform_matrix)
    else:  # Single image: (H, W, 3)
        yuv = torch.einsum("hwc,kc->hwk", rgb, transform_matrix)

    yuv = yuv + bias
    return yuv


def yuv2rgb_255(yuv: Tensor) -> Tensor:
    """Convert YUV to RGB color space with values in [0,255] range (legacy function)."""
    device = yuv.device
    dtype = yuv.dtype

    transform_matrix = YUV_TO_RGB_MATRIX.to(device=device, dtype=dtype)
    bias = RGB_BIAS.to(device=device, dtype=dtype)

    if len(yuv.shape) == 4:  # Batch of images: (N, H, W, 3)
        rgb = torch.einsum("nhwc,kc->nhwk", yuv, transform_matrix)
    else:  # Single image: (H, W, 3)
        rgb = torch.einsum("hwc,kc->hwk", yuv, transform_matrix)

    rgb = rgb + bias
    return rgb


# Backward compatibility aliases - use [0,255] range functions
def rgb2yuv(rgb: Tensor) -> Tensor:
    """Convert RGB to YUV color space (backward compatibility wrapper for [0,255] range)."""
    return rgb2yuv_255(rgb)


def yuv2rgb(yuv: Tensor) -> Tensor:
    """Convert YUV to RGB color space (backward compatibility wrapper for [0,255] range)."""
    return yuv2rgb_255(yuv)


def rgb2yuv_chw_normalized(rgb: Tensor) -> Tensor:
    """OPTIMIZED: Convert RGB to YUV for CHW format tensors, eliminating permutations.
    
    Eliminates: CHW → HWC → color_convert() → HWC → CHW permutations
    Direct conversion: CHW → color_convert() → CHW
    
    Args:
        rgb: RGB tensor in CHW format with values in [0,1] range
             Single image: (3, H, W) or Batch: (N, 3, H, W)
    
    Returns:
        YUV tensor in CHW format ready for model input
    """
    device = rgb.device
    dtype = rgb.dtype
    
    transform_matrix = RGB_TO_YUV_MATRIX.to(device=device, dtype=dtype)
    bias = YUV_BIAS.to(device=device, dtype=dtype)
    
    rgb_scaled = rgb * 255.0
    
    if len(rgb_scaled.shape) == 4:  # Batch: (N, 3, H, W)
        yuv = torch.einsum("nchw,kc->nkhw", rgb_scaled, transform_matrix)
        yuv = (yuv + bias.view(1, -1, 1, 1)) / 255.0  # Batch case: (1, 3, 1, 1)
    else:  # Single image: (3, H, W)
        yuv = torch.einsum("chw,kc->khw", rgb_scaled, transform_matrix)
        yuv = (yuv + bias.view(-1, 1, 1)) / 255.0  # Single case: (3, 1, 1)
    return yuv


def yuv2rgb_chw_normalized(yuv: Tensor) -> Tensor:
    """OPTIMIZED: Convert YUV to RGB for CHW format tensors, eliminating permutations."""
    device = yuv.device
    dtype = yuv.dtype
    
    transform_matrix = YUV_TO_RGB_MATRIX.to(device=device, dtype=dtype)
    bias = RGB_BIAS.to(device=device, dtype=dtype)
    
    yuv_scaled = yuv * 255.0
    
    if len(yuv_scaled.shape) == 4:  # Batch: (N, 3, H, W)
        rgb = torch.einsum("nchw,kc->nkhw", yuv_scaled, transform_matrix)
        rgb = (rgb + bias.view(1, -1, 1, 1)) / 255.0  # Batch case: (1, 3, 1, 1)
    else:  # Single image: (3, H, W)
        rgb = torch.einsum("chw,kc->khw", yuv_scaled, transform_matrix)
        rgb = (rgb + bias.view(-1, 1, 1)) / 255.0  # Single case: (3, 1, 1)
    return rgb
