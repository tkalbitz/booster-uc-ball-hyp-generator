"""Configuration for the ball classifier."""

CPATCH_SIZE: int = 32
"""Size of the square patch for the classifier (e.g., 32x32 pixels)."""

TRAIN_BATCH_SIZE: int = 512
"""Batch size for training."""

VAL_BATCH_SIZE: int = 1024
"""Batch size for validation."""

CLASSIFIER_DILATION_FACTOR: float = 1.2
"""Factor to dilate the ball diameter for classifier patch extraction."""
