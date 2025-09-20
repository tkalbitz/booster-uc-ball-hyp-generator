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


def _make_image(path: Path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage

    img = QImage(128, 96, QImage.Format_RGB32)
    img.fill(Qt.blue)
    assert img.save(str(path))


def _send_wheel(view, delta: int) -> None:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication

    center = view.viewport().rect().center()
    pos = QPointF(center)
    event = QWheelEvent(
        pos,
        pos,
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButtons(Qt.NoButton),
        Qt.KeyboardModifiers(Qt.NoModifier),
        Qt.ScrollPhase.ScrollUpdate,
        False,
        Qt.MouseEventSource.MouseEventNotSynthesized,
    )
    QApplication.sendEvent(view.viewport(), event)
    QApplication.processEvents()


def _send_middle_drag(view, start_dx: int, start_dy: int) -> None:
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    start = view.viewport().rect().center()
    p1 = QPointF(start.x(), start.y())
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        p1,
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifiers(Qt.NoModifier),
    )
    QApplication.sendEvent(view.viewport(), press)
    QApplication.processEvents()

    p2 = QPointF(start.x() + start_dx, start.y() + start_dy)
    move = QMouseEvent(
        QEvent.Type.MouseMove,
        p2,
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifiers(Qt.NoModifier),
    )
    QApplication.sendEvent(view.viewport(), move)
    QApplication.processEvents()

    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        p2,
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifiers(Qt.NoModifier),
    )
    QApplication.sendEvent(view.viewport(), release)
    QApplication.processEvents()


def test_wheel_zoom_updates_transform_and_status(tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    from uc_ball_hyp_generator.labelingtool.gui import LabelingToolWindow

    app = QApplication.instance() or QApplication([])

    img_path = tmp_path / "z.png"
    _make_image(img_path)

    w = LabelingToolWindow()
    w.set_images([img_path])
    w.show()
    app.processEvents()

    view = w.findChild(type(w._view), None)
    assert view is not None

    t0 = view.transform()
    s0 = float(t0.m11())

    _send_wheel(view, 120)
    t1 = view.transform()
    s1 = float(t1.m11())
    assert s1 > s0

    msg = w.statusBar().currentMessage()
    assert "Zoom:" in msg

    # Clamp test: zoom out many steps shouldn't go below 0.1
    for _ in range(30):
        _send_wheel(view, -120)
    s_min = float(view.transform().m11())
    assert s_min >= 0.099


def test_middle_button_pan_translates_view(tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    from uc_ball_hyp_generator.labelingtool.gui import LabelingToolWindow

    app = QApplication.instance() or QApplication([])

    img_path = tmp_path / "p.png"
    _make_image(img_path)

    w = LabelingToolWindow()
    w.set_images([img_path])
    w.show()
    app.processEvents()

    view = w.findChild(type(w._view), None)
    assert view is not None

    # Ensure some zoom in to make panning more visible
    _send_wheel(view, 120)
    app.processEvents()

    t_before = view.transform()
    tx0, ty0 = float(t_before.m31()), float(t_before.m32())

    _send_middle_drag(view, 40, 25)

    t_after = view.transform()
    tx1, ty1 = float(t_after.m31()), float(t_after.m32())

    assert (tx1, ty1) != (tx0, ty0)
