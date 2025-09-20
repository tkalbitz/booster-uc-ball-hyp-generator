#!/usr/bin/env python3

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from uc_ball_hyp_generator.utils.logger import get_logger

_logger = get_logger(__name__)


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""

    train_ratio: float = 0.7
    val_ratio: float = 0.3


@dataclass
class LabeledImage:
    """Represents a labeled image with its CSV line."""

    image_path: Path
    csv_line: str
    is_ball: bool


def parse_csv_for_splitting(csv_file: Path, source_dir: Path) -> list[LabeledImage]:
    """Parse CSV file and return list of labeled images for splitting."""
    labeled_images: list[LabeledImage] = []

    for line in csv_file.open():
        line_stripped = line.strip()
        if not line_stripped:
            continue

        parts = line_stripped.split(";")
        if len(parts) < 4:
            continue

        image_filename = parts[0]
        label_type = parts[3] if len(parts) > 3 else ""

        # Skip ignored files
        if label_type == "Ignore":
            continue

        image_path = source_dir / image_filename
        if not image_path.exists():
            _logger.warning("Image file not found: %s", image_path)
            continue

        is_ball = label_type == "Ball"
        labeled_images.append(LabeledImage(image_path=image_path, csv_line=line_stripped, is_ball=is_ball))

    return labeled_images


def split_labeled_images(
    labeled_images: list[LabeledImage], config: SplitConfig
) -> tuple[list[LabeledImage], list[LabeledImage]]:
    """Split labeled images into train and validation sets maintaining Ball/NoBall ratios."""
    ball_images = [img for img in labeled_images if img.is_ball]
    noball_images = [img for img in labeled_images if not img.is_ball]

    # Shuffle both groups
    random.shuffle(ball_images)
    random.shuffle(noball_images)

    # Calculate split points
    ball_train_count = int(len(ball_images) * config.train_ratio)
    noball_train_count = int(len(noball_images) * config.train_ratio)

    # Split Ball images
    train_balls = ball_images[:ball_train_count]
    val_balls = ball_images[ball_train_count:]

    # Split NoBall images
    train_noballs = noball_images[:noball_train_count]
    val_noballs = noball_images[noball_train_count:]

    # Combine and shuffle final sets
    train_images = train_balls + train_noballs
    val_images = val_balls + val_noballs

    random.shuffle(train_images)
    random.shuffle(val_images)

    return train_images, val_images


def create_output_directory(output_dir: Path, labeled_images: list[LabeledImage], suffix: str) -> None:
    """Create output directory with images and labels.csv."""
    target_dir = output_dir.parent / f"{output_dir.name}_{suffix}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    for labeled_image in labeled_images:
        target_image_path = target_dir / labeled_image.image_path.name
        shutil.copy2(labeled_image.image_path, target_image_path)

    # Create labels.csv
    labels_csv_path = target_dir / "labels.csv"
    with labels_csv_path.open("w") as f:
        for labeled_image in labeled_images:
            f.write(f"{labeled_image.csv_line}\n")

    _logger.info("Created %s with %d images", target_dir, len(labeled_images))


def split_dataset(source_dir: Path, config: SplitConfig) -> None:
    """Split dataset into train and validation directories."""
    # Validate source directory
    if not source_dir.exists():
        msg = f"Source directory does not exist: {source_dir}"
        raise FileNotFoundError(msg)

    labels_csv = source_dir / "labels.csv"
    if not labels_csv.exists():
        msg = f"labels.csv not found in source directory: {source_dir}"
        raise FileNotFoundError(msg)

    # Parse CSV and get labeled images
    _logger.info("Parsing labels from %s", labels_csv)
    labeled_images = parse_csv_for_splitting(labels_csv, source_dir)

    if not labeled_images:
        msg = "No valid labeled images found"
        raise ValueError(msg)

    # Count Ball vs NoBall
    ball_count = sum(1 for img in labeled_images if img.is_ball)
    noball_count = len(labeled_images) - ball_count

    _logger.info("Found %d Ball images and %d NoBall images", ball_count, noball_count)

    # Split the dataset
    train_images, val_images = split_labeled_images(labeled_images, config)

    # Count split results
    train_balls = sum(1 for img in train_images if img.is_ball)
    train_noballs = len(train_images) - train_balls
    val_balls = sum(1 for img in val_images if img.is_ball)
    val_noballs = len(val_images) - val_balls

    _logger.info("Train set: %d Ball, %d NoBall (%d total)", train_balls, train_noballs, len(train_images))
    _logger.info("Validation set: %d Ball, %d NoBall (%d total)", val_balls, val_noballs, len(val_images))

    # Create output directories
    create_output_directory(source_dir, train_images, "train")
    create_output_directory(source_dir, val_images, "val")


def main() -> None:
    """Main entry point for the dataset splitting script."""
    parser = argparse.ArgumentParser(description="Split labeled dataset into train and validation sets")
    parser.add_argument("source_dir", type=Path, help="Directory containing images and labels.csv")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio for training set (default: 0.7)")

    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        msg = "Train ratio must be between 0 and 1"
        raise ValueError(msg)

    config = SplitConfig(train_ratio=args.train_ratio, val_ratio=1.0 - args.train_ratio)

    try:
        split_dataset(args.source_dir, config)
        _logger.info("Dataset splitting completed successfully")
    except Exception as e:
        _logger.exception("Dataset splitting failed: %s", e)
        raise


if __name__ == "__main__":
    main()
