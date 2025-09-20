from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal

from uc_ball_hyp_generator.labelingtool.utils.logger import get_logger

_logger = get_logger("uc_ball_hyp_generator.labelingtool.sam")


class SamManager:
    """Provide SAM inference or a lightweight placeholder if unavailable."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._loaded: bool = False
        self._available: bool = False

    def available(self) -> bool:
        """Return whether a real SAM backend is available."""
        return self._available

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import segment_anything  # type: ignore[import-untyped]  # noqa: F401

            self._available = True
        except Exception:  # noqa: BLE001
            self._available = False
        self._loaded = True

    def infer(self, width: int, height: int, progress_cb: Callable[[int], None] | None = None) -> list[np.ndarray]:
        """Return placeholder masks sized to the image."""
        self._ensure_loaded()
        if progress_cb is not None:
            progress_cb(10)
        cy = height // 2
        cx = width // 2
        ry = max(8, height // 6)
        rx = max(8, width // 6)
        y, x = np.ogrid[:height, :width]
        mask1 = ((x - cx) * (x - cx)) / (rx * rx) + ((y - cy) * (y - cy)) / (ry * ry) <= 1.0
        if progress_cb is not None:
            progress_cb(60)
        mask2 = np.zeros((height, width), dtype=bool)
        r2 = max(6, min(width, height) // 10)
        y0 = r2 + 10
        x0 = r2 + 10
        mask2[((x - x0) * (x - x0) + (y - y0) * (y - y0)) <= r2 * r2] = True
        if progress_cb is not None:
            progress_cb(100)
        return [mask1.astype(bool), mask2.astype(bool)]


class SamWorker(QObject):
    """Run SAM inference in a background thread."""

    progress = Signal(int)
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, path: Path, width: int, height: int, manager: SamManager) -> None:
        super().__init__()
        self._path = path
        self._width = int(width)
        self._height = int(height)
        self._manager = manager

    def run(self) -> None:
        try:
            self.progress.emit(0)
            masks = self._manager.infer(self._width, self._height, self.progress.emit)
            self.finished.emit(self._path, masks)
        except Exception as exc:  # noqa: BLE001
            msg: str = str(exc)
            self.error.emit(msg)
