"""Unit tests for scale.py functions."""

import torch

from uc_ball_hyp_generator.utils.scale import scale
from uc_ball_hyp_generator.utils.scale_patch import (
    _patch_output_height,
    _patch_output_width,
    scale_patch_x,
    scale_patch_y,
    unscale_patch_x,
    unscale_patch_y,
)


def test_scale_basic_functionality() -> None:
    """Test basic scale function with float inputs."""
    # Test scaling from [0, 10] to [0, 1]
    result = scale(5.0, 0.0, 10.0, 0.0, 1.0)
    assert isinstance(result, float) and abs(result - 0.5) < 1e-6

    # Test scaling from [-1, 1] to [0, 100]
    result = scale(0.0, -1.0, 1.0, 0.0, 100.0)
    assert isinstance(result, float) and abs(result - 50.0) < 1e-6


def test_scale_with_tensors() -> None:
    """Test scale function with tensor inputs."""
    input_tensor = torch.tensor([0.0, 0.5, 1.0])
    result = scale(input_tensor, 0.0, 1.0, 0.0, 10.0)

    expected = torch.tensor([0.0, 5.0, 10.0])
    assert isinstance(result, torch.Tensor) and torch.allclose(result, expected)


def test_scale_edge_cases() -> None:
    """Test scale function with edge cases."""
    # Test with min value
    result = scale(0.0, 0.0, 10.0, -1.0, 1.0)
    assert isinstance(result, float) and abs(result - (-1.0)) < 1e-6

    # Test with max value
    result = scale(10.0, 0.0, 10.0, -1.0, 1.0)
    assert isinstance(result, float) and abs(result - 1.0) < 1e-6


def test_scale_negative_ranges() -> None:
    """Test scale function with negative ranges."""
    result = scale(-5.0, -10.0, 0.0, 0.0, 100.0)
    assert isinstance(result, float) and abs(result - 50.0) < 1e-6


def test_scale_x_basic() -> None:
    """Test scale_x function with basic inputs."""
    # Test center point (0.0)
    result = scale_patch_x(0.0)
    assert isinstance(result, float) and abs(result - 0.0) < 1e-6

    # Test positive boundary
    result = scale_patch_x(_patch_output_width)
    assert isinstance(result, float) and abs(result - 1.0) < 1e-6

    # Test negative boundary
    result = scale_patch_x(-_patch_output_width)
    assert isinstance(result, float) and abs(result - (-1.0)) < 1e-6


def test_scale_x_with_tensor() -> None:
    """Test scale_x function with tensor input."""
    input_tensor = torch.tensor([-_patch_output_width, 0.0, _patch_output_width])
    result = scale_patch_x(input_tensor)

    expected = torch.tensor([-1.0, 0.0, 1.0])
    assert isinstance(result, torch.Tensor) and torch.allclose(result, expected, rtol=1e-6)


def test_unscale_x_basic() -> None:
    """Test unscale_x function with basic inputs."""
    # Test center point
    result = unscale_patch_x(0.0)
    assert isinstance(result, float) and abs(result - 0.0) < 1e-6

    # Test positive boundary
    result = unscale_patch_x(1.0)
    assert isinstance(result, float) and abs(result - _patch_output_width) < 1e-6

    # Test negative boundary
    result = unscale_patch_x(-1.0)
    assert isinstance(result, float) and abs(result - (-_patch_output_width)) < 1e-6


def test_unscale_x_with_tensor() -> None:
    """Test unscale_x function with tensor input."""
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])
    result = unscale_patch_x(input_tensor)

    expected = torch.tensor([-_patch_output_width, 0.0, _patch_output_width])
    assert isinstance(result, torch.Tensor) and torch.allclose(result, expected, rtol=1e-6)


def test_scale_y_basic() -> None:
    """Test scale_y function with basic inputs."""
    # Test center point (0.0)
    result = scale_patch_y(0.0)
    assert isinstance(result, float) and abs(result - 0.0) < 1e-6

    # Test positive boundary
    result = scale_patch_y(_patch_output_height)
    assert isinstance(result, float) and abs(result - 1.0) < 1e-6

    # Test negative boundary
    result = scale_patch_y(-_patch_output_height)
    assert isinstance(result, float) and abs(result - (-1.0)) < 1e-6


def test_scale_y_with_tensor() -> None:
    """Test scale_y function with tensor input."""
    input_tensor = torch.tensor([-_patch_output_height, 0.0, _patch_output_height])
    result = scale_patch_y(input_tensor)

    expected = torch.tensor([-1.0, 0.0, 1.0])
    assert isinstance(result, torch.Tensor) and torch.allclose(result, expected, rtol=1e-6)


def test_unscale_y_basic() -> None:
    """Test unscale_y function with basic inputs."""
    # Test center point
    result = unscale_patch_y(0.0)
    assert isinstance(result, float) and abs(result - 0.0) < 1e-6

    # Test positive boundary
    result = unscale_patch_y(1.0)
    assert isinstance(result, float) and abs(result - _patch_output_height) < 1e-6

    # Test negative boundary
    result = unscale_patch_y(-1.0)
    assert isinstance(result, float) and abs(result - (-_patch_output_height)) < 1e-6


def test_unscale_y_with_tensor() -> None:
    """Test unscale_y function with tensor input."""
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])
    result = unscale_patch_y(input_tensor)

    expected = torch.tensor([-_patch_output_height, 0.0, _patch_output_height])
    assert isinstance(result, torch.Tensor) and torch.allclose(result, expected, rtol=1e-6)


def test_scale_unscale_symmetry_x() -> None:
    """Test that scale_x and unscale_x are symmetric operations."""
    original_values = [-_patch_output_width * 0.5, 0.0, _patch_output_width * 0.5]

    for val in original_values:
        scaled = scale_patch_x(val)
        unscaled = unscale_patch_x(scaled)
        assert isinstance(unscaled, float) and abs(unscaled - val) < 1e-6


def test_scale_unscale_symmetry_y() -> None:
    """Test that scale_y and unscale_y are symmetric operations."""
    original_values = [-_patch_output_height * 0.5, 0.0, _patch_output_height * 0.5]

    for val in original_values:
        scaled = scale_patch_y(val)
        unscaled = unscale_patch_y(scaled)
        assert isinstance(unscaled, float) and abs(unscaled - val) < 1e-6


def test_scale_unscale_symmetry_with_tensors() -> None:
    """Test scale/unscale symmetry with tensor inputs."""
    # Test x scaling
    x_values = torch.tensor(
        [
            -_patch_output_width * 0.8,
            -_patch_output_width * 0.3,
            0.0,
            _patch_output_width * 0.3,
            _patch_output_width * 0.8,
        ]
    )
    x_scaled = scale_patch_x(x_values)
    x_unscaled = unscale_patch_x(x_scaled)
    assert isinstance(x_unscaled, torch.Tensor) and torch.allclose(x_unscaled, x_values, rtol=1e-6)

    # Test y scaling
    y_values = torch.tensor(
        [
            -_patch_output_height * 0.8,
            -_patch_output_height * 0.3,
            0.0,
            _patch_output_height * 0.3,
            _patch_output_height * 0.8,
        ]
    )
    y_scaled = scale_patch_y(y_values)
    y_unscaled = unscale_patch_y(y_scaled)
    assert isinstance(y_unscaled, torch.Tensor) and torch.allclose(y_unscaled, y_values, rtol=1e-6)


def test_output_dimensions_consistency() -> None:
    """Test that output_width and output_height are calculated correctly."""
    from uc_ball_hyp_generator.config import patch_height, patch_width

    expected_width = (patch_width / 2) * 1.2
    expected_height = (patch_height / 2) * 1.2

    assert abs(_patch_output_width - expected_width) < 1e-6
    assert abs(_patch_output_height - expected_height) < 1e-6


def test_scale_preserves_dtype() -> None:
    """Test that scaling functions preserve tensor dtype."""
    float32_tensor = torch.tensor([0.0, 1.0], dtype=torch.float32)
    float64_tensor = torch.tensor([0.0, 1.0], dtype=torch.float64)

    result32 = scale_patch_x(float32_tensor)
    result64 = scale_patch_x(float64_tensor)

    assert isinstance(result32, torch.Tensor) and result32.dtype == torch.float32
    assert isinstance(result64, torch.Tensor) and result64.dtype == torch.float64


def test_scale_with_out_of_bounds_values() -> None:
    """Test scaling with values outside the expected range."""
    # Values larger than output_width
    large_val = _patch_output_width * 2.0
    result = scale_patch_x(large_val)
    assert isinstance(result, float) and result > 1.0  # Should be outside [-1, 1] range

    # Values smaller than -output_width
    small_val = -_patch_output_width * 2.0
    result = scale_patch_x(small_val)
    assert isinstance(result, float) and result < -1.0  # Should be outside [-1, 1] range
