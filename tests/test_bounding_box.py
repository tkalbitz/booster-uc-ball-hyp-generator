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


def _make_image(path: Path, w: int = 128, h: int = 96) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage

    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(Qt.blue)
    assert img.save(str(path))


def _left_drag(view, p1, p2) -> None:
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(p1),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifiers(Qt.NoModifier),
    )
    QApplication.sendEvent(view.viewport(), press)
    QApplication.processEvents()

    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(p2),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifiers(Qt.NoModifier),
    )
    QApplication.sendEvent(view.viewport(), move)
    QApplication.processEvents()

    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(p2),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifiers(Qt.NoModifier),
    )
    QApplication.sendEvent(view.viewport(), release)
    QApplication.processEvents()


def _right_click(view, p) -> None:
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(p),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifiers(Qt.NoModifier),
    )
    QApplication.sendEvent(view.viewport(), press)
    QApplication.processEvents()

    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(p),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifiers(Qt.NoModifier),
    )
    QApplication.sendEvent(view.viewport(), release)
    QApplication.processEvents()


def _drag_handle(view, handle_item, dx: int, dy: int) -> None:
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    center_scene = handle_item.mapToScene(handle_item.rect().center())
    start_view = view.mapFromScene(center_scene)
    end_view = start_view + view.mapFromScene(center_scene + QPointF(dx, dy)) - view.mapFromScene(center_scene)

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(start_view),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifiers(Qt.NoModifier),
    )
    QApplication.sendEvent(view.viewport(), press)
    QApplication.processEvents()

    move = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(end_view),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifiers(Qt.NoModifier),
    )
    QApplication.sendEvent(view.viewport(), move)
    QApplication.processEvents()

    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(end_view),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifiers(Qt.NoModifier),
    )
    QApplication.sendEvent(view.viewport(), release)
    QApplication.processEvents()


def _find_bbox_item(scene):
    from PySide6.QtWidgets import QGraphicsRectItem

    for it in scene.items():
        if isinstance(it, QGraphicsRectItem) and it.data(0) == "bbox":
            return it
    return None


def _find_handle_item(rect_item, key: str):
    from PySide6.QtWidgets import QGraphicsRectItem

    for it in rect_item.childItems():
        if isinstance(it, QGraphicsRectItem) and it.data(0) == "handle" and it.data(1) == key:
            return it
    return None


def test_draw_move_delete_bbox(tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    from uc_ball_hyp_generator.labelingtool.gui import LabelingToolWindow

    app = QApplication.instance() or QApplication([])

    img_path = tmp_path / "bbox.png"
    _make_image(img_path)

    w = LabelingToolWindow()
    w.set_images([img_path])
    w.show()
    app.processEvents()

    view = w.findChild(type(w._view), None)
    assert view is not None

    vp = view.viewport().rect()
    start = vp.center() - vp.center() / 10
    end = vp.center() + vp.center() / 10

    _left_drag(view, start.toPoint(), end.toPoint())

    boxes = w.get_bounding_boxes()
    assert len(boxes) == 1
    b = boxes[0]
    assert b.x2 > b.x1
    assert b.y2 > b.y1

    scene = view.scene()
    assert scene is not None
    bbox_item = _find_bbox_item(scene)
    assert bbox_item is not None

    center_scene = bbox_item.mapToScene(bbox_item.rect().center())
    center_view = view.mapFromScene(center_scene)
    _left_drag(view, center_view, center_view + (center_view - vp.center()))

    boxes2 = w.get_bounding_boxes()
    assert len(boxes2) == 1
    b2 = boxes2[0]
    assert (b2.x1, b2.y1, b2.x2, b2.y2) != (b.x1, b.y1, b.x2, b.y2)

    # Delete with right click
    center_scene2 = bbox_item.mapToScene(bbox_item.rect().center())
    center_view2 = view.mapFromScene(center_scene2)
    _right_click(view, center_view2)

    boxes3 = w.get_bounding_boxes()
    assert len(boxes3) == 0


def test_resize_bbox_via_handle(tmp_path: Path) -> None:
    from PySide6.QtWidgets import QApplication

    from uc_ball_hyp_generator.labelingtool.gui import LabelingToolWindow

    app = QApplication.instance() or QApplication([])

    img_path = tmp_path / "bbox2.png"
    _make_image(img_path)

    w = LabelingToolWindow()
    w.set_images([img_path])
    w.show()
    app.processEvents()

    view = w.findChild(type(w._view), None)
    assert view is not None

    vp = view.viewport().rect()
    start = vp.center() - vp.center() / 8
    end = vp.center() + vp.center() / 8
    _left_drag(view, start.toPoint(), end.toPoint())

    boxes = w.get_bounding_boxes()
    assert len(boxes) == 1
    b = boxes[0]
    w0, h0 = (b.x2 - b.x1), (b.y2 - b.y1)

    scene = view.scene()
    assert scene is not None
    bbox_item = _find_bbox_item(scene)
    assert bbox_item is not None

    handle_se = _find_handle_item(bbox_item, "se")
    assert handle_se is not None

    _drag_handle(view, handle_se, 20, 15)

    b2 = w.get_bounding_boxes()[0]
    w1, h1 = (b2.x2 - b2.x1), (b2.y2 - b2.y1)
    assert w1 > w0
    assert h1 > h0
