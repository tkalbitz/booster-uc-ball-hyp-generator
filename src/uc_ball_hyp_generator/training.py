"""Training utilities and functions for the ball detection model."""

import os
from typing import TextIO

import torch
from torch.utils.tensorboard import SummaryWriter

from uc_ball_hyp_generator.custom_metrics import FoundBallMetric
from uc_ball_hyp_generator.logger import get_logger

_logger = get_logger(__name__)

# Global variables that should be passed as parameters in the future
train_steps_per_epoch: int = 0  # Will be set by the main script
test_steps_per_epoch: int = 0  # Will be set by the main script


def calculate_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Calculate accuracy for position prediction."""
    diff = torch.abs(y_pred - y_true[:, :2])
    threshold = 0.1
    accuracy = torch.mean((diff < threshold).float())
    return accuracy


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_metric: FoundBallMetric,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run a single training epoch and return metrics."""
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_metric.reset_states()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += calculate_accuracy(output, target).item()
        train_metric.update_state(target, output)

        if batch_idx >= train_steps_per_epoch:
            break

    train_loss /= train_steps_per_epoch
    train_acc /= train_steps_per_epoch
    train_found_balls = train_metric.result()

    return train_loss, train_acc, train_found_balls


def validate_epoch(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    val_metric: FoundBallMetric,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run a single validation epoch and return metrics."""
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_metric.reset_states()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            val_acc += calculate_accuracy(output, target).item()
            val_metric.update_state(target, output)

            if batch_idx >= test_steps_per_epoch:
                break

    val_loss /= test_steps_per_epoch
    val_acc /= test_steps_per_epoch
    val_found_balls = val_metric.result()

    return val_loss, val_acc, val_found_balls


def log_epoch_metrics(
    epoch: int,
    epochs: int,
    train_loss: float,
    train_acc: float,
    train_found_balls: float,
    val_loss: float,
    val_acc: float,
    val_found_balls: float,
    writer: SummaryWriter,
    csv_file: TextIO,
) -> None:
    """Log epoch metrics to console, TensorBoard, and CSV."""
    _logger.info(
        "Epoch %d/%d: Train - Loss: %.6f, Acc: %.6f, Found Balls: %.6f",
        epoch + 1,
        epochs,
        train_loss,
        train_acc,
        train_found_balls,
    )
    _logger.info(
        "Epoch %d/%d: Val   - Loss: %.6f, Acc: %.6f, Found Balls: %.6f",
        epoch + 1,
        epochs,
        val_loss,
        val_acc,
        val_found_balls,
    )

    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_acc, epoch)
    writer.add_scalar("Accuracy/Validation", val_acc, epoch)
    writer.add_scalar("FoundBalls/Train", train_found_balls, epoch)
    writer.add_scalar("FoundBalls/Validation", val_found_balls, epoch)

    csv_file.write(
        f"{epoch + 1},{train_loss},{val_loss},{train_acc},{val_acc},{train_found_balls},{val_found_balls}\\n"
    )
    csv_file.flush()


def save_model_checkpoints(
    model: torch.nn.Module,
    model_dir: str,
    epoch: int,
    val_loss: float,
    val_found_balls: float,
    best_val_loss: float,
    best_val_found_balls: float,
) -> tuple[float, float, int]:
    """Save model checkpoints and return updated best metrics and patience counter."""
    patience_counter = 0

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        loss_filename = f"weights.loss.{epoch + 1:03d}-{val_loss:.6f}-{val_found_balls:.6f}.pth"
        torch.save(
            model.state_dict(),
            os.path.join(model_dir, loss_filename),
        )
        patience_counter = 0
        _logger.info("Save new best model (loss): %s", loss_filename)
    else:
        patience_counter += 1

    if val_found_balls > best_val_found_balls:
        best_val_found_balls = val_found_balls
        most_balls_filename = f"weights.balls.{epoch + 1:03d}-{val_found_balls:.6f}-{val_loss:.6f}.pth"
        torch.save(
            model.state_dict(),
            os.path.join(model_dir, most_balls_filename),
        )
        _logger.info("Save new best model (ball): %s", most_balls_filename)

    return best_val_loss, best_val_found_balls, patience_counter


def run_training_loop(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    writer: SummaryWriter,
    csv_file: TextIO,
    model_dir: str,
    device: torch.device,
    epochs: int = 10000,
) -> None:
    """Main training loop for the model."""
    best_val_loss = float("inf")
    best_val_found_balls = 0.0
    patience_counter = 0
    early_stopping_patience = 60

    train_metric = FoundBallMetric()
    val_metric = FoundBallMetric()

    for epoch in range(epochs):
        train_loss, train_acc, train_found_balls = train_epoch(
            model, train_loader, optimizer, criterion, train_metric, device
        )
        val_loss, val_acc, val_found_balls = validate_epoch(model, test_loader, criterion, val_metric, device)

        log_epoch_metrics(
            epoch,
            epochs,
            train_loss,
            train_acc,
            train_found_balls,
            val_loss,
            val_acc,
            val_found_balls,
            writer,
            csv_file,
        )

        best_val_loss, best_val_found_balls, patience_counter = save_model_checkpoints(
            model, model_dir, epoch, val_loss, val_found_balls, best_val_loss, best_val_found_balls
        )

        scheduler.step(val_loss)

        if patience_counter >= early_stopping_patience:
            _logger.info("Early stopping triggered after %d epochs", epoch + 1)
            break

    writer.close()
    csv_file.close()
