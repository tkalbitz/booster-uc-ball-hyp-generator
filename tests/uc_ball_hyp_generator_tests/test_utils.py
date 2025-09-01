"""Unit tests for utils.py functions."""

from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from uc_ball_hyp_generator.utils import get_flops


class SimpleModel(nn.Module):
    """Simple model for testing FLOP calculation."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def test_get_flops_basic() -> None:
    """Test basic FLOP calculation functionality."""
    model = SimpleModel()
    input_shape = (3, 32, 32)

    flops = get_flops(model, input_shape)

    assert isinstance(flops, int)
    assert flops > 0


def test_get_flops_model_evaluation_mode() -> None:
    """Test that model is put in evaluation mode during FLOP calculation."""
    model = SimpleModel()
    model.train()
    input_shape = (3, 32, 32)

    original_mode = model.training
    get_flops(model, input_shape)

    assert original_mode is True
    assert model.training is False


def test_get_flops_different_input_shapes() -> None:
    """Test FLOP calculation with different input shapes."""
    model = SimpleModel()

    # Test smaller input
    flops_small = get_flops(model, (3, 16, 16))

    # Test larger input
    flops_large = get_flops(model, (3, 64, 64))

    assert flops_large > flops_small


def test_get_flops_macs_conversion() -> None:
    """Test that MACs are correctly converted to FLOPs (multiplication by 2)."""
    model = SimpleModel()
    input_shape = (3, 32, 32)

    mock_macs = 1000
    with patch("uc_ball_hyp_generator.utils.profile_macs", return_value=mock_macs):
        flops = get_flops(model, input_shape)

    assert flops == mock_macs * 2


@patch("uc_ball_hyp_generator.utils.profile_macs")
def test_get_flops_no_grad_context(mock_profile_macs: Mock) -> None:
    """Test that FLOP calculation is done in no_grad context."""
    model = SimpleModel()
    input_shape = (3, 32, 32)
    mock_profile_macs.return_value = 1000

    with torch.enable_grad():
        assert torch.is_grad_enabled()
        get_flops(model, input_shape)

    mock_profile_macs.assert_called_once()
    call_args = mock_profile_macs.call_args

    # Verify the input tensor was created correctly
    dummy_input = call_args[0][1]  # Second argument should be the dummy input
    assert dummy_input.shape == (1, *input_shape)
    assert dummy_input.requires_grad is False


def test_get_flops_single_channel_input() -> None:
    """Test FLOP calculation with single channel input."""
    model = nn.Conv2d(1, 8, 3)
    input_shape = (1, 28, 28)

    flops = get_flops(model, input_shape)

    assert isinstance(flops, int)
    assert flops > 0


def test_get_flops_batch_size_one() -> None:
    """Test that dummy input always has batch size of 1."""
    model = SimpleModel()
    input_shape = (3, 32, 32)

    with patch("uc_ball_hyp_generator.utils.profile_macs") as mock_profile_macs:
        mock_profile_macs.return_value = 1000
        get_flops(model, input_shape)

        call_args = mock_profile_macs.call_args
        dummy_input = call_args[0][1]

        assert dummy_input.shape[0] == 1  # Batch size should be 1
        assert dummy_input.shape[1:] == input_shape


def test_get_flops_with_complex_model() -> None:
    """Test FLOP calculation with a more complex model."""
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )
    input_shape = (3, 224, 224)

    flops = get_flops(model, input_shape)

    assert isinstance(flops, int)
    assert flops > 100000  # Should be a reasonably large number for this model
    assert flops > 100000  # Should be a reasonably large number for this model
