from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(name: str = __name__, level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """
    Configure a logger with a simple console handler.

    Log format: "%(levelname)s: %(message)s"
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter("%(levelname)s: %(message)s")

        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file is not None:
            fh = logging.FileHandler(str(log_file))
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Return a logger with default configuration if not already set up.

    This keeps things simple for library usage; applications can reconfigure as needed.
    """
    return setup_logger(name)
