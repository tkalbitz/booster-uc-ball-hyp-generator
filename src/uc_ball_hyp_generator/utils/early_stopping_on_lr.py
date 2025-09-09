import torch


class EarlyStoppingOnLR:
    """Stop training when learning rate reaches minimum threshold."""

    def __init__(self, min_lr: float = 1e-7, patience: int = 10) -> None:
        """
        Initialize early stopping based on learning rate.

        Args:
            min_lr: Minimum learning rate threshold to stop training
            patience: Number of epochs to wait after reaching min_lr before stopping
        """
        self.min_lr = min_lr
        self.patience = patience
        self.patience_counter = 0
        self.should_stop = False

    def check_lr(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        Check if current learning rate is below threshold.

        Args:
            optimizer: PyTorch optimizer to check learning rate

        Returns:
            True if training should stop, False otherwise
        """
        current_lr = optimizer.param_groups[0]["lr"]

        if current_lr <= self.min_lr:
            self.patience_counter += 1
            if self.patience_counter > self.patience:
                self.should_stop = True
                return True
        else:
            self.patience_counter = 0

        return False
