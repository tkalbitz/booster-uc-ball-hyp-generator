"""Main training script for ball detection model."""

import numpy as np
import torch

import uc_ball_hyp_generator.training as training
from uc_ball_hyp_generator.config import image_dir, testset_csv_collection, trainingset_csv_collection
from uc_ball_hyp_generator.csv_label_reader import load_csv_collection
from uc_ball_hyp_generator.dataset_handling import create_dataset
from uc_ball_hyp_generator.logger import setup_logger
from uc_ball_hyp_generator.model_setup import create_model, create_training_components
from uc_ball_hyp_generator.training import run_training_loop


def main() -> None:
    """Main training function."""
    _logger = setup_logger(__name__)

    # Set random seed for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("Using device: %s", device)

    batch_size_train: int = 64
    batch_size_test: int = 128

    png_files: dict[str, str] = {f.name: str(f) for f in image_dir.glob("**/*.png")}

    train_img, train_labels, skipped_trainingset = load_csv_collection(trainingset_csv_collection, png_files)
    test_img, test_labels, skipped_testset = load_csv_collection(testset_csv_collection, png_files)

    _logger.info("Trainingset contains %d images, removed %d balls", len(train_img), skipped_trainingset)
    _logger.info("Testset contains %d images, removed %d balls", len(test_img), skipped_testset)

    train_ds = create_dataset(train_img, train_labels, batch_size_train)
    test_ds = create_dataset(test_img, test_labels, batch_size_test, trainset=False)

    train_steps_per_epoch = int(len(train_img) / batch_size_train) + 1
    test_steps_per_epoch = int(len(test_img) / batch_size_test)

    _logger.info("Train steps: %d, test steps: %d", train_steps_per_epoch, test_steps_per_epoch)

    # Set global variables in training module
    training.train_steps_per_epoch = train_steps_per_epoch
    training.test_steps_per_epoch = test_steps_per_epoch

    model, model_dir, log_dir = create_model()
    optimizer, criterion, scheduler, writer, csv_file = create_training_components(model, model_dir, log_dir)

    run_training_loop(model, train_ds, test_ds, optimizer, criterion, scheduler, writer, csv_file, model_dir, device)


if __name__ == "__main__":
    main()
