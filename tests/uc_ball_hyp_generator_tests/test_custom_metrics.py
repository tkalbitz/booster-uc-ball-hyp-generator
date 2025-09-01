"""Unit tests for custom_metrics.py FoundBallMetric class."""

from unittest.mock import patch

import torch

from uc_ball_hyp_generator.config import patch_height, patch_width
from uc_ball_hyp_generator.custom_metrics import FoundBallMetric


def test_found_ball_metric_initialization() -> None:
    """Test FoundBallMetric initialization."""
    metric = FoundBallMetric()

    assert metric.name == "found_balls"
    assert metric.found_balls == 0.0
    assert metric.totals_balls == 0.0


def test_found_ball_metric_custom_name() -> None:
    """Test FoundBallMetric initialization with custom name."""
    custom_name = "test_balls"
    metric = FoundBallMetric(custom_name)

    assert metric.name == custom_name
    assert metric.found_balls == 0.0
    assert metric.totals_balls == 0.0


def test_reset_states() -> None:
    """Test reset_states method."""
    metric = FoundBallMetric()

    # Manually set some values
    metric.found_balls = 10.0
    metric.totals_balls = 15.0

    metric.reset_states()

    assert metric.found_balls == 0.0
    assert metric.totals_balls == 0.0


def test_result_empty_state() -> None:
    """Test result method when no balls have been processed."""
    metric = FoundBallMetric()

    result = metric.result()

    assert result == 0.0


def test_result_with_data() -> None:
    """Test result method with some found balls."""
    metric = FoundBallMetric()

    metric.found_balls = 8.0
    metric.totals_balls = 10.0

    result = metric.result()

    assert result == 0.8


def test_update_state_perfect_prediction() -> None:
    """Test update_state with perfect predictions (distance = 0)."""
    metric = FoundBallMetric()

    # Create identical predictions and ground truth
    y_true = torch.tensor([[0.0, 0.0, 10.0]])  # x, y, radius
    y_pred = torch.tensor([[0.0, 0.0, 10.0]])  # identical prediction

    with (
        patch("src.custom_metrics.unscale_x", side_effect=lambda x: x),
        patch("src.custom_metrics.unscale_y", side_effect=lambda x: x),
    ):
        metric.update_state(y_true, y_pred)

    assert metric.found_balls == 1.0
    assert metric.totals_balls == 1.0
    assert metric.result() == 1.0


def test_update_state_prediction_within_radius() -> None:
    """Test update_state with predictions within the ball radius."""
    metric = FoundBallMetric()

    # Ground truth: ball at (0, 0) with radius 10
    y_true = torch.tensor([[0.0, 0.0, 10.0]])
    # Prediction: ball at (5, 5) - distance = sqrt(50) ≈ 7.07, which is < 10
    y_pred = torch.tensor([[5.0, 5.0, 10.0]])

    with (
        patch("src.custom_metrics.unscale_x", side_effect=lambda x: x),
        patch("src.custom_metrics.unscale_y", side_effect=lambda x: x),
    ):
        metric.update_state(y_true, y_pred)

    assert metric.found_balls == 1.0
    assert metric.totals_balls == 1.0
    assert metric.result() == 1.0


def test_update_state_prediction_outside_radius() -> None:
    """Test update_state with predictions outside the ball radius."""
    metric = FoundBallMetric()

    # Ground truth: ball at (0, 0) with radius 10
    y_true = torch.tensor([[0.0, 0.0, 10.0]])
    # Prediction: ball at (15, 0) - distance = 15, which is > 10
    y_pred = torch.tensor([[15.0, 0.0, 10.0]])

    with (
        patch("src.custom_metrics.unscale_x", side_effect=lambda x: x),
        patch("src.custom_metrics.unscale_y", side_effect=lambda x: x),
    ):
        metric.update_state(y_true, y_pred)

    assert metric.found_balls == 0.0
    assert metric.totals_balls == 1.0
    assert metric.result() == 0.0


def test_update_state_batch_mixed_results() -> None:
    """Test update_state with batch of mixed results."""
    metric = FoundBallMetric()

    # Batch of 3 balls
    y_true = torch.tensor(
        [
            [0.0, 0.0, 10.0],  # Ball 1: center (0,0), radius 10
            [20.0, 20.0, 5.0],  # Ball 2: center (20,20), radius 5
            [50.0, 50.0, 15.0],  # Ball 3: center (50,50), radius 15
        ]
    )

    y_pred = torch.tensor(
        [
            [3.0, 4.0, 10.0],  # Ball 1: distance = 5 < 10 (found)
            [25.0, 25.0, 5.0],  # Ball 2: distance ≈ 7.07 > 5 (not found)
            [45.0, 45.0, 15.0],  # Ball 3: distance ≈ 7.07 < 15 (found)
        ]
    )

    with (
        patch("src.custom_metrics.unscale_x", side_effect=lambda x: x),
        patch("src.custom_metrics.unscale_y", side_effect=lambda x: x),
    ):
        metric.update_state(y_true, y_pred)

    assert metric.found_balls == 2.0  # 2 out of 3 found
    assert metric.totals_balls == 3.0
    assert metric.result() == 2.0 / 3.0


def test_update_state_multiple_calls() -> None:
    """Test update_state with multiple calls accumulating results."""
    metric = FoundBallMetric()

    # First batch: 2 balls, 1 found
    y_true1 = torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, 5.0]])
    y_pred1 = torch.tensor([[3.0, 4.0, 10.0], [10.0, 0.0, 5.0]])  # First found, second not

    # Second batch: 1 ball, 1 found
    y_true2 = torch.tensor([[0.0, 0.0, 20.0]])
    y_pred2 = torch.tensor([[5.0, 5.0, 20.0]])  # Found

    with (
        patch("src.custom_metrics.unscale_x", side_effect=lambda x: x),
        patch("src.custom_metrics.unscale_y", side_effect=lambda x: x),
    ):
        metric.update_state(y_true1, y_pred1)
        metric.update_state(y_true2, y_pred2)

    assert metric.found_balls == 2.0  # 1 + 1 found
    assert metric.totals_balls == 3.0  # 2 + 1 total
    assert metric.result() == 2.0 / 3.0


def test_update_state_calls_unscale_functions() -> None:
    """Test that update_state correctly calls unscale functions."""
    metric = FoundBallMetric()

    y_true = torch.tensor([[0.5, -0.5, 10.0]])
    y_pred = torch.tensor([[-0.5, 0.5, 10.0]])

    with (
        patch("src.custom_metrics.unscale_x") as mock_unscale_x,
        patch("src.custom_metrics.unscale_y") as mock_unscale_y,
    ):
        # Set up mock returns to make calculation simple
        mock_unscale_x.side_effect = lambda x: x * 100  # Scale up for testing
        mock_unscale_y.side_effect = lambda x: x * 100

        metric.update_state(y_true, y_pred)

        # Verify unscale_x was called for both true and predicted x values
        assert mock_unscale_x.call_count == 2
        # Verify unscale_y was called for both true and predicted y values
        assert mock_unscale_y.call_count == 2


def test_update_state_adds_patch_offset() -> None:
    """Test that update_state correctly adds patch width/height offset."""
    metric = FoundBallMetric()

    y_true = torch.tensor([[0.0, 0.0, 10.0]])
    y_pred = torch.tensor([[0.0, 0.0, 10.0]])

    with (
        patch("src.custom_metrics.unscale_x") as mock_unscale_x,
        patch("src.custom_metrics.unscale_y") as mock_unscale_y,
    ):
        mock_unscale_x.return_value = torch.tensor([100.0])  # Mock return value
        mock_unscale_y.return_value = torch.tensor([200.0])  # Mock return value

        metric.update_state(y_true, y_pred)

        # Check that patch offset was added to unscaled values
        expected_x_true = mock_unscale_x.return_value + patch_width / 2
        expected_y_true = mock_unscale_y.return_value + patch_height / 2

        assert torch.allclose(expected_x_true, torch.tensor([100.0 + patch_width / 2]))
        assert torch.allclose(expected_y_true, torch.tensor([200.0 + patch_height / 2]))


def test_update_state_tensor_conversion() -> None:
    """Test that update_state properly handles tensor conversions."""
    metric = FoundBallMetric()

    y_true = torch.tensor([[0.0, 0.0, 10.0]])
    y_pred = torch.tensor([[0.0, 0.0, 10.0]])

    with (
        patch("src.custom_metrics.unscale_x", return_value=0.0),
        patch("src.custom_metrics.unscale_y", return_value=0.0),
    ):
        # Should not raise any tensor conversion errors
        metric.update_state(y_true, y_pred)

        assert metric.totals_balls == 1.0


def test_update_state_with_sample_weight_none() -> None:
    """Test update_state with sample_weight=None (default behavior)."""
    metric = FoundBallMetric()

    y_true = torch.tensor([[0.0, 0.0, 10.0]])
    y_pred = torch.tensor([[3.0, 4.0, 10.0]])

    with (
        patch("src.custom_metrics.unscale_x", side_effect=lambda x: x),
        patch("src.custom_metrics.unscale_y", side_effect=lambda x: x),
    ):
        # sample_weight parameter exists but is not used in current implementation
        metric.update_state(y_true, y_pred, sample_weight=None)

    assert metric.found_balls == 1.0
    assert metric.totals_balls == 1.0
    assert metric.totals_balls == 1.0
