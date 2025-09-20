"""Main training script for ball detection model."""

import numpy as np
import torch
from torch import optim

import uc_ball_hyp_generator.hyp_generator.training as training
from uc_ball_hyp_generator.hyp_generator.config import image_dir, testset_csv_collection, trainingset_csv_collection
from uc_ball_hyp_generator.hyp_generator.dataset_handling import create_dataset
from uc_ball_hyp_generator.hyp_generator.model_setup import DistanceLoss, create_model
from uc_ball_hyp_generator.hyp_generator.training import run_training_loop
from uc_ball_hyp_generator.utils.csv_label_reader import load_csv_collection
from uc_ball_hyp_generator.utils.logger import setup_logger


def main() -> None:
    """Main training function."""
    _logger = setup_logger(__name__)
    # mp.set_start_method("spawn")

    # Set random seed for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)

    device: torch.device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda")

    _logger.info("Using device: %s", device)

    batch_size_train: int = 64
    batch_size_test: int = 128

    # Support both PNG and JPG files
    image_files: dict[str, str] = {}
    for pattern in ["**/*.png", "**/*.jpg", "**/*.jpeg"]:
        image_files.update({f.name: str(f) for f in image_dir.glob(pattern)})

    train_img, train_labels, neg_img, skipped_trainingset = load_csv_collection(trainingset_csv_collection, image_files)
    test_img, test_labels, neg_img, skipped_testset = load_csv_collection(testset_csv_collection, image_files)

    _logger.info("Trainingset contains %d images, removed %d balls", len(train_img), skipped_trainingset)
    _logger.info("Testset contains %d images, removed %d balls", len(test_img), skipped_testset)

    train_ds = create_dataset(train_img, train_labels, batch_size_train, device=device)
    test_ds = create_dataset(test_img, test_labels, batch_size_test, device=device, trainset=False)

    train_steps_per_epoch = int(len(train_img) / batch_size_train) + 1
    test_steps_per_epoch = int(len(test_img) / batch_size_test)

    _logger.info("Train steps: %d, test steps: %d", train_steps_per_epoch, test_steps_per_epoch)

    # Set global variables in training module
    training.train_steps_per_epoch = train_steps_per_epoch
    training.test_steps_per_epoch = test_steps_per_epoch

    model, model_dir, log_dir = create_model(compile_model=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, amsgrad=False)
    criterion = DistanceLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=700)

    run_training_loop(model, train_ds, test_ds, optimizer, criterion, scheduler, model_dir, device)


if __name__ == "__main__":
    main()
