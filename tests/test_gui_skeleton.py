from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure 'src' is importable without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_basics(tmp_path: Path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    from uc_ball_hyp_generator.labelingtool.gui import LabelingToolWindow

    img1 = QImage(64, 64, QImage.Format_RGB32)
    img1.fill(Qt.red)
    img_path1 = tmp_path / "img1.png"
    assert img1.save(str(img_path1))

    img2 = QImage(64, 64, QImage.Format_RGB32)
    img2.fill(Qt.green)
    img_path2 = tmp_path / "img2.png"
    assert img2.save(str(img_path2))

    window = LabelingToolWindow()
    window.set_images([img_path1, img_path2])
    window.show()
    app.processEvents()

    assert "Labeling Tool" in window.windowTitle()
    assert window.statusBar() is not None

    options_menu = next((a.menu() for a in window.menuBar().actions() if a.text() == "Options"), None)
    assert options_menu is not None
    help_menu = next((a.menu() for a in window.menuBar().actions() if a.text() == "Help"), None)
    assert help_menu is not None

    sam_action = next((a for a in options_menu.actions() if a.objectName() == "action_single_action_mode"), None)
    assert sam_action is not None
    kb_action = next((a for a in help_menu.actions() if a.objectName() == "action_keyboard_shortcuts"), None)
    assert kb_action is not None

    assert window.current_index == 0
    assert window.on_next() is True
    assert window.current_index == 1
    assert window.on_next() is False
    assert window.on_prev() is True
    assert window.current_index == 0

    window.close()
