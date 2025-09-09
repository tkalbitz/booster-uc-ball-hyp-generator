"""Logging configuration for the ball detection project."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = __name__, level: int = logging.INFO, log_file: Optional[str | Path] = None
) -> logging.Logger:
    """Set up a logger with console and optional file output.

    Args:
        name: Logger name (defaults to module name)
        level: Logging level (default: INFO)
        log_file: Optional file path for logging output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """Get an existing logger or create a new one with default settings."""
    return logging.getLogger(name) if logging.getLogger(name).handlers else setup_logger(name)
