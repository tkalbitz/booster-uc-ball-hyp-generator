"""
uc_ball_hyp_generator.labelingtool

A Qt-based labeling tool for soccer ball annotations.
This package exposes a minimal public API now (Milestones 1 & 2),
with full functionality arriving in later milestones.

Public API:
- BoundingBox: Immutable data model for labeled boxes.
- load_config: Load YAML config merged with sensible defaults.
- get_shape_for_class: Resolve the configured shape for a class name.
- label_image: High-level API entry (stub for now).
- save_labels: CSV persistence function (stub for now).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from uc_ball_hyp_generator.labelingtool.model import BoundingBox, Shape
from uc_ball_hyp_generator.labelingtool.config import load_config, get_shape_for_class
from uc_ball_hyp_generator.labelingtool.utils.logger import get_logger

__all__ = [
    "BoundingBox",
    "Shape",
    "load_config",
    "get_shape_for_class",
    "label_image",
    "save_labels",
    "_logger",
]

# Package-level logger following AGENTS.md convention
_logger = get_logger("uc_ball_hyp_generator.labelingtool")


def label_image(image_path: str | Path, *, config: dict[str, object] | None = None) -> BoundingBox:
    """
    Run SAM + optional manual correction for a single image and return a BoundingBox.

    Note:
        This is a placeholder stub for Milestones 1 & 2. The full implementation
        (UI, SAM integration, etc.) will be delivered in later milestones.

    Args:
        image_path: Path to the image file.
        config: Optional pre-loaded configuration dictionary, typically from load_config().

    Returns:
        BoundingBox: The labeled bounding box.

    Raises:
        NotImplementedError: Always, until later milestones implement the UI workflow.
    """
    raise NotImplementedError("label_image() is not implemented yet (planned in later milestones).")


def save_labels(csv_path: str | Path, entries: Sequence[BoundingBox]) -> None:
    """
    Save labeling results to a semicolon-separated CSV file in the specified format.

    Note:
        This is a placeholder stub for Milestones 1 & 2. CSV persistence, sorting,
        and stdout support will be implemented in a later milestone.

    Args:
        csv_path: Output CSV path.
        entries: Iterable of BoundingBox entries to persist.

    Raises:
        NotImplementedError: Always, until later milestones implement persistence.
    """
    raise NotImplementedError("save_labels() is not implemented yet (planned in later milestones).")
