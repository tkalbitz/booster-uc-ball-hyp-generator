from __future__ import annotations

from pathlib import Path
from typing import Sequence, cast

import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QMainWindow,
    QMenu,
    QMenuBar,
    QProgressBar,
    QStatusBar,
)

from uc_ball_hyp_generator.labelingtool.config import get_shape_for_class, load_config
from uc_ball_hyp_generator.labelingtool.image_canvas import ImageCanvas
from uc_ball_hyp_generator.labelingtool.model import BoundingBox
from uc_ball_hyp_generator.labelingtool.persistence import (
    load_existing_labels,
    load_noball_images,
)
from uc_ball_hyp_generator.labelingtool.persistence import (
    save_labels as persist_save_labels,
)
from uc_ball_hyp_generator.labelingtool.sam import SamManager, SamWorker
from uc_ball_hyp_generator.labelingtool.shortcuts_overlay import ShortcutsOverlay
from uc_ball_hyp_generator.labelingtool.utils.logger import get_logger

_logger = get_logger("uc_ball_hyp_generator.labelingtool.gui")


class LabelingToolWindow(QMainWindow):
    """Main window for the labeling tool with image display and basic navigation."""

    def __init__(self, cfg: dict[str, object] | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Labeling Tool")
        self.setMinimumSize(1024, 768)
        self._image_paths: list[Path] = []
        self._index: int = -1
        self._single_action_mode: bool = False
        self._current_class_name: str = "Ball"

        self._view = ImageCanvas(self)
        self.setCentralWidget(self._view)
        self._view.zoom_changed.connect(self._on_zoom_changed)  # type: ignore[arg-type]
        self._view.set_shape_provider(lambda: get_shape_for_class(self._current_class_name))
        self._view.set_class_provider(lambda: self._current_class_name)

        self._menubar = QMenuBar(self)
        self.setMenuBar(self._menubar)
        self._options_menu = QMenu("Options", self)
        self._help_menu = QMenu("Help", self)
        self._menubar.addMenu(self._options_menu)
        self._menubar.addMenu(self._help_menu)

        self.action_single_action_mode = QAction("Single-Action Mode", self)
        self.action_single_action_mode.setObjectName("action_single_action_mode")
        self.action_single_action_mode.setCheckable(True)
        self.action_single_action_mode.triggered.connect(self._toggle_single_action_mode)  # type: ignore[arg-type]
        self._options_menu.addAction(self.action_single_action_mode)

        self.action_keyboard_shortcuts = QAction("Keyboard Shortcuts", self)
        self.action_keyboard_shortcuts.setObjectName("action_keyboard_shortcuts")
        self._help_menu.addAction(self.action_keyboard_shortcuts)
        self.action_keyboard_shortcuts.triggered.connect(self._show_shortcuts_overlay)  # type: ignore[arg-type]

        self._overlay = ShortcutsOverlay(self)
        self._overlay.setObjectName("shortcuts_overlay")
        self._overlay.hide()

        cfg = cfg or load_config()
        self._cfg = cfg
        sam_section = cast(dict[str, object], cfg.get("sam", {}))
        model_name = str(sam_section.get("model_name", "sam_vit_b_01"))
        cache_dir = str(sam_section.get("cache_dir", "~/.cache/uc_ball_hyp_generator/models/"))
        self._sam_manager = SamManager(model_name=model_name, cache_dir=cache_dir)

        shape_map = cast(dict[str, str], cfg.get("shape", {}))
        self._class_names: list[str] = list(shape_map.keys())

        self._sam_results: dict[Path, list[np.ndarray]] = {}
        self._sam_thread: QThread | None = None
        self._sam_worker: SamWorker | None = None
        self._prev_class_before_n: str | None = None

        self._image_undo: dict[Path, list[list[tuple[float, float, float, float, str, str]]]] = {}
        self._image_state: dict[Path, list[tuple[float, float, float, float, str, str]]] = {}
        self._image_noball_set: set[Path] = set()
        self._current_path: Path | None = None
        self._csv_path: Path | None = None

        self._status = QStatusBar(self)
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

        self._progress = QProgressBar(self)
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        self._status.addPermanentWidget(self._progress)

        self._class_combo = QComboBox(self)
        self._class_combo.setObjectName("class_selector")
        self._class_combo.addItems(self._class_names)
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)  # type: ignore[arg-type]
        self._status.addPermanentWidget(self._class_combo)
        try:
            initial_index = self._class_names.index(self._current_class_name)
        except ValueError:
            initial_index = 0
        self._class_combo.setCurrentIndex(initial_index)

    def set_images(self, paths: Sequence[str | Path]) -> None:
        """Set the list of images to navigate."""
        self._image_paths = [Path(p).expanduser() for p in paths]
        self._index = 0 if self._image_paths else -1
        self._csv_path = self._determine_csv_path()
        self._preload_labels_from_csv()
        self._load_current_image()

    def open_path(self, path: str | Path) -> None:
        """Open a single file or a directory of images."""
        p = Path(path).expanduser()
        if p.is_dir():
            candidates = [x for x in sorted(p.iterdir()) if x.suffix.lower() in (".png", ".jpg", ".jpeg")]
            self.set_images(candidates)
            return
        if p.is_file():
            self.set_images([p])
            return
        msg = "Path does not exist"
        raise RuntimeError(msg)

    @property
    def current_index(self) -> int:
        """Return the current image index, -1 when no images are loaded."""
        return self._index

    def get_bounding_boxes(self) -> list[BoundingBox]:
        """Return bounding boxes for the current image."""
        return self._view.get_bounding_boxes()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        if self._overlay.isVisible() and key != Qt.Key.Key_Question:
            self._overlay.hide_overlay()

        if (key == Qt.Key.Key_Z) and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            if self._view.undo():
                self._status.showMessage("Undo", 1500)
            else:
                self._status.showMessage("Nothing to undo", 1500)
            return

        if key in (Qt.Key.Key_Right, Qt.Key.Key_Space):
            self.on_next()
            return
        if key == Qt.Key.Key_Left:
            self.on_prev()
            return

        if key == Qt.Key.Key_Question:
            self._show_shortcuts_overlay()
            return

        if key == Qt.Key.Key_S:
            self._toggle_sam_overlay()
            return

        if key == Qt.Key.Key_N:
            self._toggle_noball_mode()
            return

        if key == Qt.Key.Key_Delete:
            if self._view.delete_selected_bboxes():
                self._status.showMessage("Deleted selected bounding box", 1500)
            return

        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            if self._view.is_noball_active():
                self._status.showMessage("NoBall active; class forced to NoBall", 1500)
                return
            idx = int(key) - int(Qt.Key.Key_0)
            if 1 <= idx <= len(self._class_names):
                self._class_combo.setCurrentIndex(idx - 1)
                self._status.showMessage(f"Class: {self._class_names[idx - 1]}", 1500)
                return

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._status.showMessage("Accepted", 800)
            if self._single_action_mode:
                self.on_next()
            return

        if key == Qt.Key.Key_Escape:
            if self._view.abort_current_edit():
                self._status.showMessage("Edit aborted", 800)
            return

        super().keyPressEvent(event)

    def _show_shortcuts_overlay(self) -> None:
        self._overlay.show_overlay()

    def _on_class_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._class_names):
            return
        name = self._class_names[idx]
        if self._view.is_noball_active() and name != "NoBall":
            self._status.showMessage("NoBall active; class forced to NoBall", 1500)
            try:
                nb_index = self._class_names.index("NoBall")
            except ValueError:
                nb_index = idx
            self._class_combo.blockSignals(True)
            self._class_combo.setCurrentIndex(nb_index)
            self._class_combo.blockSignals(False)
            self._current_class_name = "NoBall"
            return
        self._current_class_name = name
        self._status.showMessage(f"Class: {self._current_class_name}", 1500)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._overlay.isVisible():
            self._overlay.setGeometry(self.rect())

    def _save_current_image_state(self) -> None:
        if self._current_path is None:
            return
        self._image_state[self._current_path] = self._view.get_current_state()
        self._image_undo[self._current_path] = self._view.get_undo_stack()
        if self._view.is_noball_active():
            self._image_noball_set.add(self._current_path)
        else:
            self._image_noball_set.discard(self._current_path)
        self._persist_current_image_label()

    def _restore_image_state(self, path: Path) -> None:
        state = self._image_state.get(path)
        if state is not None:
            self._view.set_current_state(state)
        undo_stack = self._image_undo.get(path, [])
        self._view.set_undo_stack(undo_stack)

    def on_next(self) -> bool:
        """Advance to the next image if possible."""
        if not self._image_paths:
            return False
        if self._index + 1 >= len(self._image_paths):
            return False
        self._index += 1
        self._load_current_image()
        return True

    def on_prev(self) -> bool:
        """Return to the previous image if possible."""
        if not self._image_paths:
            return False
        if self._index - 1 < 0:
            return False
        self._index -= 1
        self._load_current_image()
        return True

    def _toggle_single_action_mode(self, checked: bool) -> None:
        self._single_action_mode = bool(checked)
        self._status.showMessage(
            "Single-Action Mode: ON" if self._single_action_mode else "Single-Action Mode: OFF", 2000
        )

    def _on_zoom_changed(self, ratio: float) -> None:
        """Handle zoom updates from the canvas and show percent in the status bar."""
        percent = int(round(ratio * 100))
        self._status.showMessage(f"Zoom: {percent}%")

    def _toggle_noball_mode(self) -> None:
        prev = self._view.is_noball_active()
        self._view.toggle_noball_x()
        now = self._view.is_noball_active()
        if now and not prev:
            if self._current_class_name != "NoBall":
                self._prev_class_before_n = self._current_class_name
            self._current_class_name = "NoBall"
            try:
                nb_index = self._class_names.index("NoBall")
                self._class_combo.blockSignals(True)
                self._class_combo.setCurrentIndex(nb_index)
                self._class_combo.blockSignals(False)
            except ValueError:
                pass
            self._class_combo.setEnabled(False)
            self._status.showMessage("NoBall ON — class forced to NoBall", 1500)
            return
        if prev and not now:
            self._class_combo.setEnabled(True)
            if self._prev_class_before_n and self._prev_class_before_n in self._class_names:
                self._current_class_name = self._prev_class_before_n
                try:
                    idx = self._class_names.index(self._prev_class_before_n)
                    self._class_combo.blockSignals(True)
                    self._class_combo.setCurrentIndex(idx)
                    self._class_combo.blockSignals(False)
                except ValueError:
                    pass
            self._prev_class_before_n = None
            self._status.showMessage("NoBall OFF", 1500)

    def _toggle_sam_overlay(self) -> None:
        if self._current_path is None or not self._view.has_image():
            self._status.showMessage("No image for SAM", 1500)
            return
        if self._view.has_sam_overlay():
            self._view.hide_sam_overlay()
            self._status.showMessage("SAM overlay hidden", 1500)
            return
        cached = self._sam_results.get(self._current_path, [])
        if cached:
            self._view.set_sam_masks(cached)
            self._view.show_sam_overlay()
            self._status.showMessage("SAM overlay shown", 1500)
            return
        self._start_sam_for_current_image()

    def _start_sam_for_current_image(self) -> None:
        if self._sam_thread is not None:
            self._status.showMessage("SAM already running", 1500)
            return
        if self._current_path is None:
            return
        width, height = self._view.get_image_size()
        if width <= 0 or height <= 0:
            self._status.showMessage("Invalid image size for SAM", 1500)
            return
        self._sam_thread = QThread(self)
        self._sam_worker = SamWorker(self._current_path, self._sam_manager)
        self._sam_worker.moveToThread(self._sam_thread)
        self._sam_thread.started.connect(self._sam_worker.run)  # type: ignore[arg-type]
        self._sam_worker.progress.connect(self._on_sam_progress)  # type: ignore[arg-type]
        self._sam_worker.finished.connect(self._on_sam_finished)  # type: ignore[arg-type]
        self._sam_worker.error.connect(self._on_sam_error)  # type: ignore[arg-type]
        self._sam_thread.start()
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._status.showMessage("SAM running... 0%")

    def _on_sam_progress(self, value: int) -> None:
        self._progress.setValue(value)
        self._status.showMessage(f"SAM running... {value}%")

    def _on_sam_finished(self, path_obj: object, masks_obj: object) -> None:
        self._progress.setVisible(False)
        self._cleanup_sam_thread()
        path: Path | None = None
        if isinstance(path_obj, Path):
            path = path_obj
        elif isinstance(path_obj, str):
            path = Path(path_obj)
        masks: list[np.ndarray] = masks_obj if isinstance(masks_obj, list) else []
        if path is None or self._current_path is None or path != self._current_path:
            self._status.showMessage("SAM finished for another image", 2000)
            return
        self._sam_results[path] = masks
        if masks:
            self._view.set_sam_masks(masks)
            self._view.show_sam_overlay()
            self._status.showMessage(f"SAM complete ({len(masks)} masks)", 2000)
        else:
            self._status.showMessage("SAM complete (no masks)", 2000)

    def _on_sam_error(self, message: str) -> None:
        self._progress.setVisible(False)
        self._cleanup_sam_thread()
        self._status.showMessage(message, 3000)

    def _cleanup_sam_thread(self) -> None:
        if self._sam_thread is not None:
            try:
                self._sam_thread.quit()
                self._sam_thread.wait(500)
            except Exception:  # noqa: BLE001
                pass
        self._sam_worker = None
        self._sam_thread = None

    def _load_current_image(self) -> None:
        self._save_current_image_state()
        if self._index < 0 or self._index >= len(self._image_paths):
            self._view.clear_pixmap()
            self._status.showMessage("No image")
            self._current_path = None
            self._update_title()
            return
        path = self._image_paths[self._index]
        pix = QPixmap(str(path))
        if pix.isNull():
            self._view.clear_pixmap()
            self._status.showMessage("Failed to load image")
            _logger.debug("Failed to load image %s", path)
            self._current_path = None
            self._update_title()
            return
        self._view.set_pixmap(pix)
        self._restore_image_state(path)
        # Sync NoBall overlay with cached state
        need_nb = path in self._image_noball_set
        if self._view.is_noball_active() != need_nb:
            self._view.toggle_noball_x()
        self._current_path = path
        self._status.showMessage(f"{self._index + 1}/{len(self._image_paths)}: {path.name}")
        _logger.debug("Loaded image %s at index %s", path, self._index)
        self._update_title()

    def _update_title(self) -> None:
        base = "Labeling Tool"
        if self._index >= 0 and self._image_paths:
            path = self._image_paths[self._index]
            self.setWindowTitle(f"{base} — {path.name} ({self._index + 1}/{len(self._image_paths)})")
        else:
            self.setWindowTitle(base)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_current_image_state()
        self._persist_all_labels()
        super().closeEvent(event)

    def _determine_csv_path(self) -> Path | None:
        if not self._image_paths:
            return None
        cli = cast(dict[str, object], self._cfg.get("cli", {}))
        out_val = cli.get("output")
        if isinstance(out_val, str) and out_val.strip():
            return Path(out_val).expanduser()
        return self._image_paths[0].parent / "labels.csv"

    def _preload_labels_from_csv(self) -> None:
        if not self._csv_path or not self._csv_path.is_file():
            return
        try:
            bbox_map = load_existing_labels(self._csv_path)
            noballs = load_noball_images(self._csv_path)
        except Exception:  # noqa: BLE001
            return
        name_to_path: dict[str, Path] = {p.name: p for p in self._image_paths}
        for name, bb in bbox_map.items():
            path = name_to_path.get(name)
            if path is None:
                continue
            shape = get_shape_for_class(bb.class_name)
            self._image_state[path] = [
                (float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2), shape.value, bb.class_name)
            ]
        for name in noballs:
            path = name_to_path.get(name)
            if path is None:
                continue
            self._image_noball_set.add(path)

    def _collect_entry_for_path(self, path: Path) -> tuple[Path, BoundingBox | None] | None:
        if path in self._image_noball_set:
            return (path, None)
        state = self._image_state.get(path, [])
        if not state:
            return None
        x1, y1, x2, y2, _shape, class_name = state[0]
        bb = BoundingBox(int(x1), int(y1), int(x2), int(y2), class_name=class_name, subclass=class_name)
        return (path, bb)

    def _persist_current_image_label(self) -> None:
        if not self._csv_path or self._current_path is None:
            return
        entry = self._collect_entry_for_path(self._current_path)
        if entry is None:
            return
        try:
            persist_save_labels(self._csv_path, [entry])
        except Exception:  # noqa: BLE001
            pass

    def _persist_all_labels(self) -> None:
        if not self._csv_path:
            return
        entries: list[tuple[Path, BoundingBox | None]] = []
        for p in self._image_paths:
            if entry := self._collect_entry_for_path(p):
                entries.append(entry)
        if not entries:
            return
        try:
            persist_save_labels(self._csv_path, entries)
        except Exception:  # noqa: BLE001
            pass
