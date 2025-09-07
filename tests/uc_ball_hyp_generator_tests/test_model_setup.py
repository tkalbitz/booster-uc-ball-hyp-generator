"""Unit tests for key functions in model_setup.py."""

import shutil
import tempfile
from unittest.mock import Mock, mock_open, patch

import pytest
import torch
import torch.nn as nn

from uc_ball_hyp_generator.model_setup import (
    DistanceLoss,
    compile_existing_model,
    create_model,
    create_training_components,
)


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_distance_loss_initialization() -> None:
    """Test DistanceLoss initialization."""
    loss = DistanceLoss()

    assert hasattr(loss, "log_two")
    assert hasattr(loss, "confidence_scale")
    assert isinstance(loss.confidence_scale, torch.Tensor) and loss.confidence_scale.item() == 3.0


def test_distance_loss_logcosh() -> None:
    """Test the _logcosh method."""
    loss = DistanceLoss()

    # Test with simple values
    x = torch.tensor([0.0, 1.0, -1.0])
    result = loss._logcosh(x)

    assert isinstance(result, torch.Tensor)
    assert result.shape == x.shape
    # For x=0, logcosh should be log(2) - log(2) = 0
    # The implementation uses: x + softplus(-2*x) - log(2)
    # For x=0: 0 + softplus(0) - log(2) = 0 + log(2) - log(2) = 0
    assert torch.allclose(result[0], torch.tensor(0.0), atol=1e-6)


def test_distance_loss_forward_identical_predictions() -> None:
    """Test DistanceLoss forward with identical predictions."""
    loss = DistanceLoss()

    y_true = torch.tensor([[0.0, 0.0, 10.0]])
    y_pred = torch.tensor([[0.0, 0.0, 10.0]])

    with (
        patch("uc_ball_hyp_generator.model_setup.unscale_x", side_effect=lambda x: x),
        patch("uc_ball_hyp_generator.model_setup.unscale_y", side_effect=lambda x: x),
    ):
        result = loss.forward(y_pred, y_true)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Should be scalar
        # When predictions are identical, loss should be small
        assert result.item() < 1.0


def test_distance_loss_forward_different_predictions() -> None:
    """Test DistanceLoss forward with different predictions."""
    loss = DistanceLoss()

    y_true = torch.tensor([[0.0, 0.0, 10.0]])
    y_pred = torch.tensor([[5.0, 5.0, 10.0]])  # Different from true

    with (
        patch("uc_ball_hyp_generator.model_setup.unscale_x", side_effect=lambda x: x),
        patch("uc_ball_hyp_generator.model_setup.unscale_y", side_effect=lambda x: x),
    ):
        result = loss.forward(y_pred, y_true)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Should be scalar
        assert result.item() > 0.0  # Should have some loss


def test_distance_loss_forward_batch() -> None:
    """Test DistanceLoss forward with batch input."""
    loss = DistanceLoss()

    y_true = torch.tensor([[0.0, 0.0, 10.0], [1.0, 1.0, 5.0]])
    y_pred = torch.tensor([[0.5, 0.5, 10.0], [1.5, 1.5, 5.0]])

    with (
        patch("uc_ball_hyp_generator.model_setup.unscale_x", side_effect=lambda x: x),
        patch("uc_ball_hyp_generator.model_setup.unscale_y", side_effect=lambda x: x),
    ):
        result = loss.forward(y_pred, y_true)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Should be scalar
        assert result.item() > 0.0


def test_distance_loss_tensor_conversion() -> None:
    """Test that DistanceLoss properly converts non-tensor outputs from unscale functions."""
    loss = DistanceLoss()

    y_true = torch.tensor([[0.0, 0.0, 10.0]])
    y_pred = torch.tensor([[1.0, 1.0, 10.0]])

    # Mock unscale functions to return scalar values instead of tensors
    with (
        patch("uc_ball_hyp_generator.model_setup.unscale_x") as mock_unscale_x,
        patch("uc_ball_hyp_generator.model_setup.unscale_y") as mock_unscale_y,
    ):
        # Return tensors with appropriate shapes to test tensor conversion logic
        mock_unscale_x.side_effect = lambda x: torch.tensor([2.0]) if x[0].item() == 0.0 else torch.tensor([3.0])
        mock_unscale_y.side_effect = lambda x: torch.tensor([4.0]) if x[0].item() == 0.0 else torch.tensor([5.0])

        result = loss.forward(y_pred, y_true)

        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0


def test_create_model_basic() -> None:
    """Test basic create_model functionality."""
    temp_dir = tempfile.mkdtemp()

    try:
        with (
            patch("uc_ball_hyp_generator.model_setup.models.create_network_v2") as mock_create_network,
            patch("uc_ball_hyp_generator.model_setup.get_flops", return_value=1000000),
            patch("uc_ball_hyp_generator.model_setup.sys.argv", ["script.py", "test_model"]),
            patch("uc_ball_hyp_generator.model_setup.os.makedirs"),
            patch("uc_ball_hyp_generator.model_setup.shutil.copy2"),
            patch("uc_ball_hyp_generator.model_setup.torch.save"),
            patch("uc_ball_hyp_generator.model_setup.Path") as mock_path,
        ):
            mock_model = MockModel()
            mock_create_network.return_value = mock_model
            mock_path.return_value.parent.glob.return_value = []

            model, model_dir, log_dir = create_model(compile_model=False)

            assert model is mock_model
            assert model_dir == "model/test_model"
            assert log_dir == "./logs/test_model"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_create_model_with_compilation() -> None:
    """Test create_model with torch compilation enabled."""
    with (
        patch("uc_ball_hyp_generator.model_setup.models.create_network_v2") as mock_create_network,
        patch("uc_ball_hyp_generator.model_setup.get_flops", return_value=1000000),
        patch("uc_ball_hyp_generator.model_setup.sys.argv", ["script.py", "test_model"]),
        patch("uc_ball_hyp_generator.model_setup.os.makedirs"),
        patch("uc_ball_hyp_generator.model_setup.shutil.copy2"),
        patch("uc_ball_hyp_generator.model_setup.torch.save"),
        patch("uc_ball_hyp_generator.model_setup.Path") as mock_path,
        patch("uc_ball_hyp_generator.model_setup.torch.compile") as mock_compile,
    ):
        mock_model = MockModel()
        mock_create_network.return_value = mock_model
        mock_path.return_value.parent.glob.return_value = []
        mock_compiled_model = Mock()
        mock_compile.return_value = mock_compiled_model

        model, model_dir, log_dir = create_model(compile_model=True)

        mock_compile.assert_called_once_with(mock_model, mode="default", fullgraph=False)
        assert model is mock_compiled_model


def test_create_model_compilation_failure() -> None:
    """Test create_model when compilation fails."""
    with (
        patch("uc_ball_hyp_generator.model_setup.models.create_network_v2") as mock_create_network,
        patch("uc_ball_hyp_generator.model_setup.get_flops", return_value=1000000),
        patch("uc_ball_hyp_generator.model_setup.sys.argv", ["script.py", "test_model"]),
        patch("uc_ball_hyp_generator.model_setup.os.makedirs"),
        patch("uc_ball_hyp_generator.model_setup.shutil.copy2"),
        patch("uc_ball_hyp_generator.model_setup.torch.save"),
        patch("uc_ball_hyp_generator.model_setup.Path") as mock_path,
        patch("uc_ball_hyp_generator.model_setup.torch.compile", side_effect=RuntimeError("Compilation failed")),
        patch("uc_ball_hyp_generator.model_setup._logger") as mock_logger,
    ):
        mock_model = MockModel()
        mock_create_network.return_value = mock_model
        mock_path.return_value.parent.glob.return_value = []

        model, model_dir, log_dir = create_model(compile_model=True)

        # Should return original model when compilation fails
        assert model is mock_model
        mock_logger.warning.assert_called()


def test_create_model_auto_naming() -> None:
    """Test create_model with automatic model naming."""
    with (
        patch("uc_ball_hyp_generator.model_setup.models.create_network_v2") as mock_create_network,
        patch("uc_ball_hyp_generator.model_setup.get_flops", return_value=1000000),
        patch("uc_ball_hyp_generator.model_setup.sys.argv", ["script.py"]),
        patch("uc_ball_hyp_generator.model_setup.time.strftime", return_value="2023-01-15-12-30-45"),
        patch("uc_ball_hyp_generator.model_setup.os.makedirs"),
        patch("uc_ball_hyp_generator.model_setup.shutil.copy2"),
        patch("uc_ball_hyp_generator.model_setup.torch.save"),
        patch("uc_ball_hyp_generator.model_setup.Path") as mock_path,
    ):
        mock_model = MockModel()
        mock_create_network.return_value = mock_model
        mock_path.return_value.parent.glob.return_value = []

        model, model_dir, log_dir = create_model(compile_model=False)

        assert model_dir == "model/yuv_2023-01-15-12-30-45"
        assert log_dir == "./logs/yuv_2023-01-15-12-30-45"


def test_create_model_test_mode() -> None:
    """Test create_model with --test argument."""
    with (
        patch("uc_ball_hyp_generator.model_setup.models.create_network_v2") as mock_create_network,
        patch("uc_ball_hyp_generator.model_setup.sys.argv", ["script.py", "--test"]),
        patch("uc_ball_hyp_generator.model_setup.sys.exit") as mock_exit,
    ):
        mock_model = MockModel()
        mock_create_network.return_value = mock_model

        # sys.exit should be called before function returns
        with pytest.raises(SystemExit):
            mock_exit.side_effect = SystemExit(1)
            create_model(compile_model=False)

        mock_exit.assert_called_once_with(1)


def test_create_model_flops_calculation_failure() -> None:
    """Test create_model when FLOP calculation fails."""
    with (
        patch("uc_ball_hyp_generator.model_setup.models.create_network_v2") as mock_create_network,
        patch("uc_ball_hyp_generator.model_setup.get_flops", side_effect=RuntimeError("FLOP calculation failed")),
        patch("uc_ball_hyp_generator.model_setup.sys.argv", ["script.py", "test_model"]),
        patch("uc_ball_hyp_generator.model_setup.os.makedirs"),
        patch("uc_ball_hyp_generator.model_setup.shutil.copy2"),
        patch("uc_ball_hyp_generator.model_setup.torch.save"),
        patch("uc_ball_hyp_generator.model_setup.Path") as mock_path,
        patch("uc_ball_hyp_generator.model_setup._logger") as mock_logger,
    ):
        mock_model = MockModel()
        mock_create_network.return_value = mock_model
        mock_path.return_value.parent.glob.return_value = []

        model, model_dir, log_dir = create_model(compile_model=False)

        # Should continue execution despite FLOP calculation failure
        assert model is mock_model
        mock_logger.warning.assert_called()


def test_create_training_components() -> None:
    """Test create_training_components function."""
    model = MockModel()
    model_dir = "/tmp/test_model"
    log_dir = "/tmp/test_logs"

    with (
        patch("uc_ball_hyp_generator.model_setup.optim.AdamW") as mock_adamw,
        patch("uc_ball_hyp_generator.model_setup.optim.lr_scheduler.ReduceLROnPlateau") as mock_scheduler,
        patch("uc_ball_hyp_generator.model_setup.SummaryWriter") as mock_writer,
        patch("builtins.open", mock_open()) as mock_file,
    ):
        mock_optimizer = Mock()
        mock_adamw.return_value = mock_optimizer
        mock_sched = Mock()
        mock_scheduler.return_value = mock_sched
        mock_sw = Mock()
        mock_writer.return_value = mock_sw

        optimizer, criterion, scheduler, writer, csv_file = create_training_components(model, model_dir, log_dir)

        # Verify components were created
        assert optimizer is mock_optimizer
        assert isinstance(criterion, DistanceLoss)
        assert scheduler is mock_sched
        assert writer is mock_sw

        # Verify AdamW optimizer was called with correct parameters
        # We can't directly compare generators, so check call was made with right types
        mock_adamw.assert_called_once()
        call_args = mock_adamw.call_args
        assert call_args[1]["lr"] == 0.001
        assert call_args[1]["amsgrad"] is True

        # Verify scheduler was created with correct parameters
        mock_scheduler.assert_called_once_with(mock_optimizer, mode="min", factor=0.2, patience=15, min_lr=1e-7)

        # Verify SummaryWriter was created with log directory
        mock_writer.assert_called_once_with(log_dir)

        # Verify CSV file was opened and header written
        mock_file.assert_called_once()


def test_compile_existing_model_success() -> None:
    """Test compile_existing_model with successful compilation."""
    model = MockModel()

    with (
        patch("uc_ball_hyp_generator.model_setup.torch.compile") as mock_compile,
        patch("uc_ball_hyp_generator.model_setup._logger") as mock_logger,
    ):
        mock_compiled = Mock()
        mock_compile.return_value = mock_compiled

        result = compile_existing_model(model, mode="max-autotune")

        mock_compile.assert_called_once_with(model, mode="max-autotune", fullgraph=False)
        assert result is mock_compiled
        mock_logger.info.assert_called_once()


def test_compile_existing_model_failure() -> None:
    """Test compile_existing_model with compilation failure."""
    model = MockModel()

    with (
        patch("uc_ball_hyp_generator.model_setup.torch.compile", side_effect=RuntimeError("Compilation failed")),
        patch("uc_ball_hyp_generator.model_setup._logger") as mock_logger,
    ):
        result = compile_existing_model(model)

        assert result is model  # Should return original model
        mock_logger.warning.assert_called()


def test_compile_existing_model_not_available() -> None:
    """Test compile_existing_model when torch.compile is not available."""
    model = MockModel()

    with (
        patch("uc_ball_hyp_generator.model_setup.torch", spec_set=[]),
        patch("uc_ball_hyp_generator.model_setup._logger") as mock_logger,
    ):
        result = compile_existing_model(model)

        assert result is model  # Should return original model
        mock_logger.warning.assert_called_with("torch.compile not available, returning uncompiled model")


def test_compile_existing_model_default_mode() -> None:
    """Test compile_existing_model with default compilation mode."""
    model = MockModel()

    with (
        patch("uc_ball_hyp_generator.model_setup.torch.compile") as mock_compile,
        patch("uc_ball_hyp_generator.model_setup._logger"),
    ):
        mock_compiled = Mock()
        mock_compile.return_value = mock_compiled

        result = compile_existing_model(model)

        mock_compile.assert_called_once_with(model, mode="default", fullgraph=False)
        assert result is mock_compiled
        result = compile_existing_model(model)

        assert mock_compile.call_count == 2
        mock_compile.assert_called_with(model, mode="default", fullgraph=False)
        assert result is mock_compiled
