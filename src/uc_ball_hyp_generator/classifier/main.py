"""Training script for the ball classifier."""

import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import BCELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from uc_ball_hyp_generator.classifier.config import CPATCH_SIZE, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE
from uc_ball_hyp_generator.classifier.dataset import BallClassifierDataset
from uc_ball_hyp_generator.classifier.model import BallClassifierHypercolumn, get_ball_classifier_model
from uc_ball_hyp_generator.hyp_generator.config import (
    image_dir,
    testset_csv_collection,
    trainingset_csv_collection,
)
from uc_ball_hyp_generator.hyp_generator.utils import load_ball_hyp_model
from uc_ball_hyp_generator.utils.csv_label_reader import load_csv_collection
from uc_ball_hyp_generator.utils.flops import get_flops
from uc_ball_hyp_generator.utils.logger import get_logger

_logger = get_logger(__name__)


def create_datasets(hyp_model: torch.nn.Module) -> tuple[BallClassifierDataset, BallClassifierDataset]:
    """Create training and validation datasets."""
    # Get image files mapping
    png_files = {f.name: str(f) for f in image_dir.glob("**/*.png")}

    # Load training data

    train_positive_imgs, train_positive_labels, train_negative_imgs, _ = load_csv_collection(
        trainingset_csv_collection, png_files
    )

    # Load test data

    test_positive_imgs, test_positive_labels, test_negative_imgs, _ = load_csv_collection(
        testset_csv_collection, png_files
    )

    _logger.info("Training set: %d positive, %d negative images", len(train_positive_imgs), len(train_negative_imgs))
    _logger.info("Test set: %d positive, %d negative images", len(test_positive_imgs), len(test_negative_imgs))

    # Create datasets
    train_dataset = BallClassifierDataset(
        train_positive_imgs, train_positive_labels, train_negative_imgs, hyp_model, is_training=True
    )

    val_dataset = BallClassifierDataset(
        test_positive_imgs, test_positive_labels, test_negative_imgs, hyp_model, is_training=False
    )

    return train_dataset, val_dataset


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: AdamW,
    criterion: BCELoss,
    device: torch.device,
) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Create progress bar for training batches
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")

    for _batch_idx, (patches, labels) in progress_bar:
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(patches)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar with current loss
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: BCELoss,
    device: torch.device,
) -> tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Create progress bar for validation batches
    progress_bar = tqdm(dataloader, total=len(dataloader), desc="Validation")

    with torch.no_grad():
        for patches, labels in progress_bar:
            patches = patches.to(device)
            labels = labels.to(device)

            outputs = model(patches)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            correct += batch_correct
            total += batch_total

            # Update progress bar with current accuracy
            batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0.0
            progress_bar.set_postfix(accuracy=batch_accuracy)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def main() -> None:
    """Main training function."""
    # Create default output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    default_output_dir = f"model/classifier-{timestamp}/"

    parser = argparse.ArgumentParser(description="Train ball classifier")
    parser.add_argument("--weights", type=str, required=True, help="Path to hyp_model weights file")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default=default_output_dir, help="Output directory for model files")

    args = parser.parse_args()

    # Setup device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    _logger.info("Using device: %s", device)

    # Load hypothesis model
    weights_path = Path(args.weights)
    hyp_model = load_ball_hyp_model(weights_path, torch.device("cpu"))

    # Create datasets
    train_dataset, val_dataset = create_datasets(hyp_model)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    classifier = BallClassifierHypercolumn().to("cpu")
    _logger.info(summary(classifier, input_size=(1, 3, CPATCH_SIZE, CPATCH_SIZE)))
    classifier = BallClassifierHypercolumn().to(torch.device("cpu"))
    try:
        flops = get_flops(classifier, (3, CPATCH_SIZE, CPATCH_SIZE))
        _logger.info("Model has %.2f MFlops", flops / 1e6)
    except Exception as e:
        _logger.warning("Could not calculate FLOPs: %s", e)

    # Create classifier model
    classifier: torch.nn.Module = get_ball_classifier_model().to(device)

    # Setup training components
    optimizer = AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = BCELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_accuracy = 0.0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        _logger.info("Epoch %d/%d", epoch + 1, args.epochs)

        # Training
        train_loss = train_epoch(classifier, train_loader, optimizer, criterion, device)

        # Validation
        val_loss, val_accuracy = validate(classifier, val_loader, criterion, device)

        scheduler.step()

        _logger.info(
            "%04d: Train Loss: %.4f, Val Loss: %.4f, Val Accuracy: %.4f", epoch, train_loss, val_loss, val_accuracy
        )

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            # Save model with accuracy-first filename format
            acc_first_filename = f"model.acc-{epoch + 1:03d}-{val_accuracy:.6f}-{val_loss:.6f}.pth"
            acc_first_path = output_dir / acc_first_filename
            torch.save(classifier.state_dict(), acc_first_path, pickle_protocol=5)

            _logger.info("acc: New best model saved: %.6f", best_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save model with loss-first filename format
            loss_first_filename = f"model.loss-{epoch + 1:03d}-{val_loss:.6f}-{val_accuracy:.6f}.pth"
            loss_first_path = output_dir / loss_first_filename
            torch.save(classifier.state_dict(), loss_first_path, pickle_protocol=5)

            _logger.info("loss: New best model saved: %.6f", best_val_loss)

    _logger.info("Training completed. Best validation accuracy: %.4f", best_accuracy)


if __name__ == "__main__":
    main()
