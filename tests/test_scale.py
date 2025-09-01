"""Unit tests for scale.py functions."""

import torch

from src.scale import scale, scale_x, unscale_x, scale_y, unscale_y, output_width, output_height


def test_scale_basic_functionality() -> None:
    """Test basic scale function with float inputs."""
    # Test scaling from [0, 10] to [0, 1]
    result = scale(5.0, 0.0, 10.0, 0.0, 1.0)
    assert abs(result - 0.5) < 1e-6
    
    # Test scaling from [-1, 1] to [0, 100]
    result = scale(0.0, -1.0, 1.0, 0.0, 100.0)
    assert abs(result - 50.0) < 1e-6


def test_scale_with_tensors() -> None:
    """Test scale function with tensor inputs."""
    input_tensor = torch.tensor([0.0, 0.5, 1.0])
    result = scale(input_tensor, 0.0, 1.0, 0.0, 10.0)
    
    expected = torch.tensor([0.0, 5.0, 10.0])
    assert torch.allclose(result, expected)


def test_scale_edge_cases() -> None:
    """Test scale function with edge cases."""
    # Test with min value
    result = scale(0.0, 0.0, 10.0, -1.0, 1.0)
    assert abs(result - (-1.0)) < 1e-6
    
    # Test with max value  
    result = scale(10.0, 0.0, 10.0, -1.0, 1.0)
    assert abs(result - 1.0) < 1e-6


def test_scale_negative_ranges() -> None:
    """Test scale function with negative ranges."""
    result = scale(-5.0, -10.0, 0.0, 0.0, 100.0)
    assert abs(result - 50.0) < 1e-6


def test_scale_x_basic() -> None:
    """Test scale_x function with basic inputs."""
    # Test center point (0.0)
    result = scale_x(0.0)
    assert abs(result - 0.0) < 1e-6
    
    # Test positive boundary
    result = scale_x(output_width)
    assert abs(result - 1.0) < 1e-6
    
    # Test negative boundary
    result = scale_x(-output_width)
    assert abs(result - (-1.0)) < 1e-6


def test_scale_x_with_tensor() -> None:
    """Test scale_x function with tensor input."""
    input_tensor = torch.tensor([-output_width, 0.0, output_width])
    result = scale_x(input_tensor)
    
    expected = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(result, expected, rtol=1e-6)


def test_unscale_x_basic() -> None:
    """Test unscale_x function with basic inputs."""
    # Test center point
    result = unscale_x(0.0)
    assert abs(result - 0.0) < 1e-6
    
    # Test positive boundary
    result = unscale_x(1.0)
    assert abs(result - output_width) < 1e-6
    
    # Test negative boundary  
    result = unscale_x(-1.0)
    assert abs(result - (-output_width)) < 1e-6


def test_unscale_x_with_tensor() -> None:
    """Test unscale_x function with tensor input."""
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])
    result = unscale_x(input_tensor)
    
    expected = torch.tensor([-output_width, 0.0, output_width])
    assert torch.allclose(result, expected, rtol=1e-6)


def test_scale_y_basic() -> None:
    """Test scale_y function with basic inputs."""
    # Test center point (0.0)
    result = scale_y(0.0)
    assert abs(result - 0.0) < 1e-6
    
    # Test positive boundary
    result = scale_y(output_height)
    assert abs(result - 1.0) < 1e-6
    
    # Test negative boundary
    result = scale_y(-output_height)
    assert abs(result - (-1.0)) < 1e-6


def test_scale_y_with_tensor() -> None:
    """Test scale_y function with tensor input."""
    input_tensor = torch.tensor([-output_height, 0.0, output_height])
    result = scale_y(input_tensor)
    
    expected = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(result, expected, rtol=1e-6)


def test_unscale_y_basic() -> None:
    """Test unscale_y function with basic inputs."""
    # Test center point
    result = unscale_y(0.0)
    assert abs(result - 0.0) < 1e-6
    
    # Test positive boundary
    result = unscale_y(1.0)
    assert abs(result - output_height) < 1e-6
    
    # Test negative boundary
    result = unscale_y(-1.0)
    assert abs(result - (-output_height)) < 1e-6


def test_unscale_y_with_tensor() -> None:
    """Test unscale_y function with tensor input."""
    input_tensor = torch.tensor([-1.0, 0.0, 1.0])
    result = unscale_y(input_tensor)
    
    expected = torch.tensor([-output_height, 0.0, output_height])
    assert torch.allclose(result, expected, rtol=1e-6)


def test_scale_unscale_symmetry_x() -> None:
    """Test that scale_x and unscale_x are symmetric operations."""
    original_values = [-output_width * 0.5, 0.0, output_width * 0.5]
    
    for val in original_values:
        scaled = scale_x(val)
        unscaled = unscale_x(scaled)
        assert abs(unscaled - val) < 1e-6


def test_scale_unscale_symmetry_y() -> None:
    """Test that scale_y and unscale_y are symmetric operations."""
    original_values = [-output_height * 0.5, 0.0, output_height * 0.5]
    
    for val in original_values:
        scaled = scale_y(val)
        unscaled = unscale_y(scaled)
        assert abs(unscaled - val) < 1e-6


def test_scale_unscale_symmetry_with_tensors() -> None:
    """Test scale/unscale symmetry with tensor inputs."""
    # Test x scaling
    x_values = torch.tensor([-output_width * 0.8, -output_width * 0.3, 0.0, output_width * 0.3, output_width * 0.8])
    x_scaled = scale_x(x_values)
    x_unscaled = unscale_x(x_scaled)
    assert torch.allclose(x_unscaled, x_values, rtol=1e-6)
    
    # Test y scaling  
    y_values = torch.tensor([-output_height * 0.8, -output_height * 0.3, 0.0, output_height * 0.3, output_height * 0.8])
    y_scaled = scale_y(y_values)
    y_unscaled = unscale_y(y_scaled)
    assert torch.allclose(y_unscaled, y_values, rtol=1e-6)


def test_output_dimensions_consistency() -> None:
    """Test that output_width and output_height are calculated correctly."""
    from src.config import patch_width, patch_height
    
    expected_width = (patch_width / 2) * 1.2
    expected_height = (patch_height / 2) * 1.2
    
    assert abs(output_width - expected_width) < 1e-6
    assert abs(output_height - expected_height) < 1e-6


def test_scale_preserves_dtype() -> None:
    """Test that scaling functions preserve tensor dtype."""
    float32_tensor = torch.tensor([0.0, 1.0], dtype=torch.float32)
    float64_tensor = torch.tensor([0.0, 1.0], dtype=torch.float64)
    
    result32 = scale_x(float32_tensor)
    result64 = scale_x(float64_tensor)
    
    assert result32.dtype == torch.float32
    assert result64.dtype == torch.float64


def test_scale_with_out_of_bounds_values() -> None:
    """Test scaling with values outside the expected range."""
    # Values larger than output_width
    large_val = output_width * 2.0
    result = scale_x(large_val)
    assert result > 1.0  # Should be outside [-1, 1] range
    
    # Values smaller than -output_width
    small_val = -output_width * 2.0  
    result = scale_x(small_val)
    assert result < -1.0  # Should be outside [-1, 1] range