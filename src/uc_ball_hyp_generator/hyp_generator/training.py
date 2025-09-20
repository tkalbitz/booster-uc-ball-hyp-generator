"""Training utilities and functions for the ball detection model."""

import os
import time
from dataclasses import dataclass

import torch

from uc_ball_hyp_generator.hyp_generator.patch_found_ball_metric import PatchBallRadiusMetric, PatchFoundBallMetric
from uc_ball_hyp_generator.utils.early_stopping_on_lr import EarlyStoppingOnLR
from uc_ball_hyp_generator.utils.logger import get_logger

_logger = get_logger(__name__)

# Global variables that should be passed as parameters in the future
train_steps_per_epoch: int = 0  # Will be set by the main script
test_steps_per_epoch: int = 0  # Will be set by the main script


@dataclass
class Result:
    loss: float
    acc: float
    found_balls_coord: float
    found_balls_radius: float


def calculate_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Calculate accuracy for position prediction."""
    diff = torch.abs(y_pred - y_true[:, :3])
    threshold = 0.1
    accuracy = torch.mean((diff < threshold).float())
    return accuracy


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_metric_coord: PatchFoundBallMetric,
    train_metric_radius: PatchBallRadiusMetric,
    device: torch.device,
) -> Result:
    """Run a single training epoch and return metrics."""
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_metric_coord.reset_states()
    train_metric_radius.reset_states()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += calculate_accuracy(output, target).item()
        train_metric_coord.update_state(target, output)
        train_metric_radius.update_state(target, output)

        if batch_idx >= train_steps_per_epoch:
            break

    train_loss /= train_steps_per_epoch
    train_acc /= train_steps_per_epoch
    train_found_balls_coord = train_metric_coord.result()
    train_found_balls_radius = train_metric_radius.result()

    return Result(train_loss, train_acc, train_found_balls_coord, train_found_balls_radius)


def validate_epoch(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    val_metric_ball_coord: PatchFoundBallMetric,
    val_metric_ball_radius: PatchBallRadiusMetric,
    device: torch.device,
) -> Result:
    """Run a single validation epoch and return metrics."""
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_metric_ball_coord.reset_states()
    val_metric_ball_radius.reset_states()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            val_acc += calculate_accuracy(output, target).item()
            val_metric_ball_coord.update_state(target, output)
            val_metric_ball_radius.update_state(target, output)

            if batch_idx >= test_steps_per_epoch:
                break

    val_loss /= test_steps_per_epoch
    val_acc /= test_steps_per_epoch
    val_found_balls_coord = val_metric_ball_coord.result()
    val_found_balls_radius = val_metric_ball_radius.result()

    return Result(val_loss, val_acc, val_found_balls_coord, val_found_balls_radius)


def log_epoch_metrics(
    epoch: int,
    epochs: int,
    train_result: Result,
    val_result: Result,
    best_result: Result,
    train_time: float,
    val_time: float,
) -> None:
    """Log epoch metrics to console, TensorBoard, and CSV."""
    _logger.info(
        "Epoch %d/%d: Min Loss: %.6f, Max Balls: %.6f, %.6f Dur: %.2fs",
        epoch + 1,
        epochs,
        best_result.loss,
        best_result.found_balls_coord,
        best_result.found_balls_radius,
        train_time + val_time,
    )
    _logger.info(
        "Train - Loss: %.6f, Acc: %.6f, Found Balls: %.6f, %.6f",
        train_result.loss,
        train_result.acc,
        train_result.found_balls_coord,
        train_result.found_balls_radius,
    )
    _logger.info(
        "Val   - Loss: %.6f, Acc: %.6f, Found Balls: %.6f, %.6f",
        val_result.loss,
        val_result.acc,
        val_result.found_balls_coord,
        val_result.found_balls_radius,
    )


def save_model_checkpoints(
    model: torch.nn.Module,
    model_dir: str,
    epoch: int,
    patience_counter: int,
    val_result: Result,
    best_val_result: Result,
) -> tuple[Result, int]:
    """Save model checkpoints and return updated best metrics and patience counter."""

    new_best_loss = False
    new_best_balls_coord = False
    new_best_balls_radius = False

    if val_result.loss < best_val_result.loss:
        loss_filename = (
            f"weights.loss.{epoch + 1:03d}-{val_result.loss:.6f}-"
            f"{val_result.found_balls_coord:.6f}-{val_result.found_balls_radius:.6f}.pth"
        )
        torch.save(model.state_dict(), os.path.join(model_dir, loss_filename), pickle_protocol=5)
        delta = val_result.loss - best_val_result.loss
        _logger.info(f"Save new best model (loss): {epoch + 1:03d} {val_result.loss:.6f} {delta:.6f}")
        new_best_loss = True
        best_val_result.loss = val_result.loss

    if val_result.found_balls_coord > best_val_result.found_balls_coord:
        most_balls_filename = (
            f"weights.balls.coord.{epoch + 1:03d}-"
            f"{val_result.found_balls_coord:.6f}-{val_result.found_balls_radius:.6f}-"
            f"{val_result.loss:.6f}.pth"
        )
        torch.save(model.state_dict(), os.path.join(model_dir, most_balls_filename), pickle_protocol=5)

        delta = val_result.found_balls_coord - best_val_result.found_balls_coord
        _logger.info(
            f"Save new best model (ball coord): {epoch + 1:03d} {val_result.found_balls_coord:.6f} {delta:.6f}"
        )
        best_val_result.found_balls_coord = val_result.found_balls_coord
        new_best_balls_coord = True

    if val_result.found_balls_radius > best_val_result.found_balls_radius:
        most_balls_filename = (
            f"weights.balls.radius.{epoch + 1:03d}-"
            f"{val_result.found_balls_radius:.6f}-{val_result.found_balls_coord:.6f}-"
            f"{val_result.loss:.6f}.pth"
        )
        torch.save(model.state_dict(), os.path.join(model_dir, most_balls_filename), pickle_protocol=5)

        delta = val_result.found_balls_radius - best_val_result.found_balls_radius
        _logger.info(
            f"Save new best model (ball radius): {epoch + 1:03d} {val_result.found_balls_radius:.6f} {delta:.6f}"
        )
        best_val_result.found_balls_radius = val_result.found_balls_radius
        new_best_balls_radius = True

    if new_best_loss or new_best_balls_coord or new_best_balls_radius:
        patience_counter = 0
    else:
        patience_counter += 1

    return best_val_result, patience_counter


def run_training_loop(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    model_dir: str,
    device: torch.device,
    epochs: int = 10000,
) -> None:
    """Main training loop for the model."""
    best_val_result = Result(float("inf"), 0.0, 0.0, 0.0)
    patience_counter = 0
    early_stopping_patience = 150

    min_loss = float("inf")
    max_accuracy = 0.0
    max_found_balls = 0.0

    train_metric = PatchFoundBallMetric()
    train_metric_radius = PatchBallRadiusMetric()
    val_metric = PatchFoundBallMetric()
    val_metric_radius = PatchBallRadiusMetric()
    early_stopping_on_lr = EarlyStoppingOnLR()

    for epoch in range(epochs):
        train_start_time = time.time()
        train_result = train_epoch(model, train_loader, optimizer, criterion, train_metric, train_metric_radius, device)
        train_time = time.time() - train_start_time

        val_start_time = time.time()
        val_result = validate_epoch(model, test_loader, criterion, val_metric, val_metric_radius, device)
        val_time = time.time() - val_start_time

        min_loss = min(min_loss, val_result.loss)
        max_accuracy = max(max_accuracy, val_result.acc)
        max_found_balls = max(max_found_balls, val_result.found_balls_coord)

        log_epoch_metrics(
            epoch,
            epochs,
            train_result,
            val_result,
            best_val_result,
            train_time,
            val_time,
        )

        best_val_result, patience_counter = save_model_checkpoints(
            model, model_dir, epoch, patience_counter, val_result, best_val_result
        )

        scheduler.step()

        if early_stopping_on_lr.check_lr(optimizer):
            _logger.info("Early stopping triggered after %d epochs due to lr is 0", epoch + 1)
            break

        if patience_counter >= early_stopping_patience:
            _logger.info("Early stopping triggered after %d epochs due to patience", epoch + 1)
            break
