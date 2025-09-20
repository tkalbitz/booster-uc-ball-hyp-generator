from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


class CustomRichHandler(RichHandler):
    """Custom RichHandler that formats log records with file:line format."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.console = Console(stderr=True)

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with file:line - Message format."""
        pathname = Path(record.pathname)
        try:
            relative_path = pathname.relative_to(Path.cwd())
        except ValueError:
            relative_path = pathname

        location = f"{relative_path}:{record.lineno}"

        color_map = {
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
            "DEBUG": "blue",
        }

        color = color_map.get(record.levelname, "white")
        return f"[{color}]{location} - {record.getMessage()}[/{color}]"


def setup_logger(name: str = __name__, level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """
    Configure a logger with custom formatting.

    Log format: "file:line - Message" with colored output
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        ch = CustomRichHandler(show_time=False, show_level=False, show_path=False, markup=True)
        ch.setLevel(level)
        logger.addHandler(ch)

        if log_file is not None:
            file_fmt = logging.Formatter("%(pathname)s:%(lineno)d - %(message)s")
            fh = logging.FileHandler(str(log_file))
            fh.setLevel(level)
            fh.setFormatter(file_fmt)
            logger.addHandler(fh)

    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Return a logger with default configuration if not already set up.

    This keeps things simple for library usage; applications can reconfigure as needed.
    """
    return setup_logger(name)
