from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, cast

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QMainWindow,
    QMenu,
    QMenuBar,
    QProgressBar,
    QStatusBar,
    QWidget,
)

from uc_ball_hyp_generator.labelingtool.config import get_shape_for_class, load_config
from uc_ball_hyp_generator.labelingtool.model import BoundingBox, Shape
from uc_ball_hyp_generator.labelingtool.persistence import (
    load_existing_labels,
    load_noball_images,
)
from uc_ball_hyp_generator.labelingtool.persistence import (
    save_labels as persist_save_labels,
)
from uc_ball_hyp_generator.labelingtool.sam import SamManager, SamWorker
from uc_ball_hyp_generator.labelingtool.utils.logger import get_logger

_logger = get_logger("uc_ball_hyp_generator.labelingtool.gui")


class _ViewportWidget(QWidget):
    """Return QRectF from rect() so center() yields QPointF in tests."""

    def rect(self) -> QRectF:  # type: ignore[override]
        return QRectF(super().rect())


class ShortcutsOverlay(QWidget):
    """Semi-transparent overlay that lists available keyboard shortcuts."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Widget)
        self.setVisible(False)

    def show_overlay(self) -> None:
        pw = self.parentWidget()
        geom = pw.rect() if pw is not None else self.rect()
        self.setGeometry(geom)
        self.setVisible(True)
        self.raise_()

    def hide_overlay(self) -> None:
        self.setVisible(False)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 180))
        pen = QPen(QColor("white"))
        painter.setPen(pen)
        x = 30
        y = 40
        line_h = 24
        lines = [
            "Keyboard Shortcuts",
            "",
            "Navigation:",
            "  Space / Right Arrow  – Next image",
            "  Left Arrow           – Previous image",
            "",
            "Editing:",
            "  Mouse Drag           – Create rectangle",
            "  Right Click on Box   – Delete box",
            "  Ctrl+Z               – Undo last action",
            "  Esc                  – Abort current edit",
            "  Enter                – Accept label",
            "",
            "Modes:",
            "  s                    – Toggle SAM overlay",
            "  n                    – Toggle NoBall X",
            "  1/2/3…               – Switch class",
        ]
        for i, text in enumerate(lines):
            painter.drawText(x, y + i * line_h, text)


class ResizableRectItem(QGraphicsRectItem):
    """A movable, resizable rectangle with optional shape overlays and delete-on-right-click."""

    def __init__(
        self,
        initial_scene_rect: QRectF,
        bounds_rect: QRectF,
        shape: Shape,
        class_name: str,
        on_begin_interaction: Callable[[], None] | None = None,
        on_changed: Callable[[], None] | None = None,
        on_deleted: Callable[["ResizableRectItem"], None] | None = None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self._bounds_rect = QRectF(bounds_rect)
        self._shape = shape
        self._class_name = class_name
        self._on_begin_interaction = on_begin_interaction
        self._on_changed = on_changed
        self._on_deleted = on_deleted

        norm = QRectF(initial_scene_rect).normalized()
        self.setPos(norm.topLeft())
        self.setRect(QRectF(0.0, 0.0, norm.width(), norm.height()))

        self.setZValue(10.0)
        pen = QPen(QColor("purple"))
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(Qt.GlobalColor.transparent)
        self.setData(0, "bbox")
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        self._ell_outer = QGraphicsEllipseItem(self)
        self._ell_mid = QGraphicsEllipseItem(self)
        self._ell_inner = QGraphicsEllipseItem(self)
        for ell, color in ((self._ell_outer, "white"), (self._ell_mid, "red"), (self._ell_inner, "white")):
            pen_ell = QPen(QColor(color))
            pen_ell.setCosmetic(True)
            ell.setPen(pen_ell)
            ell.setBrush(Qt.BrushStyle.NoBrush)
            ell.setZValue(11.0)
            ell.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self._handles: dict[str, QGraphicsRectItem] = {}
        for key in ("nw", "ne", "sw", "se"):
            h = QGraphicsRectItem(self)
            h.setData(0, "handle")
            h.setData(1, key)
            h.setBrush(Qt.BrushStyle.NoBrush)
            hpen = QPen(QColor("white"))
            hpen.setCosmetic(True)
            h.setPen(hpen)
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
            h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            h.setZValue(12.0)
            self._handles[key] = h

        self._update_overlay_and_handles()

    def _update_overlay_and_handles(self) -> None:
        r = self.rect()
        if r.isEmpty():
            r = QRectF(0.0, 0.0, 1.0, 1.0)
            self.setRect(r)

        if self._shape is Shape.RECTANGLE:
            self._ell_outer.setVisible(False)
            self._ell_mid.setVisible(False)
            self._ell_inner.setVisible(False)
        else:
            base = QRectF(r)
            if self._shape is Shape.ELLIPSE:
                d = min(base.width(), base.height())
                cx, cy = base.center().x(), base.center().y()
                base = QRectF(cx - d / 2.0, cy - d / 2.0, d, d)

            self._ell_outer.setRect(base.adjusted(-1.0, -1.0, 1.0, 1.0))
            self._ell_mid.setRect(base)
            self._ell_inner.setRect(base.adjusted(1.0, 1.0, -1.0, -1.0))
            self._ell_outer.setVisible(True)
            self._ell_mid.setVisible(True)
            self._ell_inner.setVisible(True)

        size = 8.0
        positions: dict[str, QPointF] = {
            "nw": r.topLeft(),
            "ne": r.topRight(),
            "sw": r.bottomLeft(),
            "se": r.bottomRight(),
        }
        for key, handle in self._handles.items():
            handle.setRect(QRectF(-size / 2.0, -size / 2.0, size, size))
            handle.setPos(positions[key])

    def _clamp_position(self, new_pos: QPointF) -> QPointF:
        x = max(self._bounds_rect.left(), min(new_pos.x(), self._bounds_rect.right() - self.rect().width()))
        y = max(self._bounds_rect.top(), min(new_pos.y(), self._bounds_rect.bottom() - self.rect().height()))
        return QPointF(x, y)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):  # type: ignore[override]
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if isinstance(value, QPointF):
                return self._clamp_position(value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._update_overlay_and_handles()
            if self._on_changed is not None:
                self._on_changed()
        return super().itemChange(change, value)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            if self._on_begin_interaction is not None:
                self._on_begin_interaction()
            scene = self.scene()
            if scene is not None:
                scene.removeItem(self)
            if self._on_deleted is not None:
                self._on_deleted(self)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._on_begin_interaction is not None:
                self._on_begin_interaction()
        super().mousePressEvent(event)

    def resize_from_handle(self, key: str, scene_pos: QPointF) -> None:
        b = self._bounds_rect
        left = float(self.pos().x())
        top = float(self.pos().y())
        right = left + float(self.rect().width())
        bottom = top + float(self.rect().height())

        px = max(b.left(), min(scene_pos.x(), b.right()))
        py = max(b.top(), min(scene_pos.y(), b.bottom()))

        if "w" in key:
            left = min(px, right - 1.0)
        if "e" in key:
            right = max(px, left + 1.0)
        if "n" in key:
            top = min(py, bottom - 1.0)
        if "s" in key:
            bottom = max(py, top + 1.0)

        new_w = max(1.0, right - left)
        new_h = max(1.0, bottom - top)

        # For circles, enforce 1:1 aspect ratio during resize
        if self._shape == Shape.CIRCLE:
            # Use the larger dimension to ensure the circle doesn't shrink
            size = max(new_w, new_h)

            # Adjust the rectangle based on which handle is being dragged
            if "w" in key:  # Left side handles
                left = right - size
            if "e" in key:  # Right side handles
                right = left + size
            if "n" in key:  # Top handles
                top = bottom - size
            if "s" in key:  # Bottom handles
                bottom = top + size

            # Ensure we stay within bounds
            if left < b.left():
                left = b.left()
                right = left + size
            if right > b.right():
                right = b.right()
                left = right - size
            if top < b.top():
                top = b.top()
                bottom = top + size
            if bottom > b.bottom():
                bottom = b.bottom()
                top = bottom - size

            new_w = max(1.0, right - left)
            new_h = max(1.0, bottom - top)

        self.setPos(QPointF(left, top))
        self.setRect(QRectF(0.0, 0.0, new_w, new_h))
        self._update_overlay_and_handles()
        if self._on_changed is not None:
            self._on_changed()

    def get_bounding_box(self) -> BoundingBox:
        x1 = int(round(float(self.pos().x())))
        y1 = int(round(float(self.pos().y())))
        x2 = int(round(x1 + float(self.rect().width())))
        y2 = int(round(y1 + float(self.rect().height())))
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, class_name=self._class_name, subclass=self._class_name)

    def begin_interaction(self) -> None:
        if self._on_begin_interaction is not None:
            self._on_begin_interaction()

    def get_shape(self) -> Shape:
        return self._shape

    def get_class_name(self) -> str:
        return self._class_name


class RubberbandItem(QGraphicsRectItem):
    """A rubberband item that draws the appropriate shape during bounding box creation."""

    def __init__(
        self, rect: QRectF, shape: Shape, class_name: str, pen: QPen, parent: QGraphicsItem | None = None
    ) -> None:
        super().__init__(rect, parent)
        self._shape = shape
        self._class_name = class_name
        self.setPen(pen)
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.setZValue(10.0)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def set_shape(self, shape: Shape) -> None:
        self._shape = shape
        self.update()

    def set_class_name(self, class_name: str) -> None:
        self._class_name = class_name
        self.update()

    def paint(self, painter: QPainter, option, widget: QWidget | None = None) -> None:
        rect = self.rect()

        # Create white pen for outline
        white_pen = QPen(QColor("white"))
        white_pen.setCosmetic(True)
        white_pen.setWidth(1)

        # Get the main pen for the shape
        main_pen = self.pen()

        if self._shape == Shape.RECTANGLE:
            # Draw outer white outline (1px outside)
            painter.setPen(white_pen)
            inner_rect = rect.adjusted(1.0, 1.0, -1.0, -1.0)
            outer_rect = rect.adjusted(-1.0, -1.0, 1.0, 1.0)
            painter.drawRect(inner_rect)
            painter.drawRect(outer_rect)

            # Draw main shape
            painter.setPen(main_pen)
            painter.drawRect(rect)
        else:
            if self._shape == Shape.CIRCLE:
                diameter = min(rect.width(), rect.height())
                square = QRectF(rect.center().x() - diameter / 2, rect.center().y() - diameter / 2, diameter, diameter)

                # Draw outer white outline (1px outside)
                painter.setPen(white_pen)
                inner_square = square.adjusted(1.0, 1.0, -1.0, -1.0)
                outer_square = square.adjusted(-1.0, -1.0, 1.0, 1.0)
                painter.drawEllipse(inner_square)
                painter.drawEllipse(outer_square)

                # Draw main shape
                painter.setPen(main_pen)
                painter.drawEllipse(square)
            else:  # ELLIPSE
                # Draw outer white outline (1px outside)
                painter.setPen(white_pen)
                inner_rect = rect.adjusted(1.0, 1.0, -1.0, -1.0)
                outer_rect = rect.adjusted(-1.0, -1.0, 1.0, 1.0)
                painter.drawEllipse(inner_rect)
                painter.drawEllipse(outer_rect)

                # Draw main shape
                painter.setPen(main_pen)
                painter.drawEllipse(rect)


class HandleItem(QGraphicsRectItem):
    """Interactive handle forwarding drag events to the parent ResizableRectItem."""

    def __init__(self, parent_rect: ResizableRectItem, key: str) -> None:
        super().__init__(parent_rect)
        self._parent_rect = parent_rect
        self._key = key
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self._parent_rect.begin_interaction()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        self._parent_rect.resize_from_handle(self._key, event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        event.accept()


class ImageCanvas(QGraphicsView):
    """Graphics view that displays an image, supports zoom and pan."""

    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setInteractive(True)
        self.setViewport(_ViewportWidget(self))
        self._pixmap_item: QGraphicsPixmapItem | None = None

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._has_user_transform: bool = False
        self._base_scale: float = 1.0
        self._min_scale: float = 0.1
        self._max_scale: float = 10.0

        self._panning: bool = False
        self._last_pos: QPointF | None = None

        self._bbox_items: list[ResizableRectItem] = []
        self._rubberband_item: RubberbandItem | None = None
        self._drag_start: QPointF | None = None
        self._image_rect: QRectF | None = None
        self._shape_provider: Callable[[], Shape] = lambda: Shape.ELLIPSE
        self._class_provider: Callable[[], str] = lambda: "Ball"

        self._undo_stack: list[list[tuple[float, float, float, float, str, str]]] = []
        self._undo_limit: int = 100
        self._sam_overlay_visible: bool = False
        self._noball_active: bool = False
        self._noball_lines: tuple[QGraphicsLineItem, QGraphicsLineItem] | None = None
        self._sam_items: list[QGraphicsPixmapItem] = []
        self._sam_masks: list[np.ndarray] = []

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        """Handle key events in the canvas, passing navigation keys to parent."""
        key = event.key()

        # Pass navigation keys to the parent window
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Space):
            parent = self.parent()
            if parent is not None:
                parent.keyPressEvent(event)
                return

        # Let the base class handle other keys
        super().keyPressEvent(event)

    def has_image(self) -> bool:
        """Return whether an image is currently loaded."""
        return self._pixmap_item is not None and not self._pixmap_item.pixmap().isNull()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Set the pixmap to display and fit it to the view."""
        if self._pixmap_item is None:
            self._pixmap_item = self._scene.addPixmap(pixmap)
        else:
            self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._image_rect = self._pixmap_item.boundingRect()
        self.clear_bboxes()
        self._fit()

    def clear_pixmap(self) -> None:
        """Clear the current pixmap."""
        if self._pixmap_item is None:
            return
        self._scene.removeItem(self._pixmap_item)
        self._pixmap_item = None
        self.clear_bboxes()
        self._clear_noball_overlay()
        self._clear_sam_overlay()
        self._noball_active = False
        self._sam_overlay_visible = False
        self._scene.setSceneRect(0, 0, 0, 0)
        self.resetTransform()
        self._has_user_transform = False
        self._base_scale = 1.0
        self.zoom_changed.emit(1.0)

    def set_shape_provider(self, provider: Callable[[], Shape]) -> None:
        """Set a callable returning the current Shape to render."""
        self._shape_provider = provider

    def set_class_provider(self, provider: Callable[[], str]) -> None:
        """Set a callable returning the current class name for new boxes."""
        self._class_provider = provider

    def clear_bboxes(self) -> None:
        """Remove all bounding boxes and temporary items."""
        for item in list(self._bbox_items):
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        self._bbox_items.clear()
        if self._rubberband_item is not None:
            if self._rubberband_item.scene() is self._scene:
                self._scene.removeItem(self._rubberband_item)
            self._rubberband_item = None
        self._drag_start = None

    def get_bounding_boxes(self) -> list[BoundingBox]:
        """Return current bounding boxes."""
        return [it.get_bounding_box() for it in self._bbox_items]

    def _on_item_deleted(self, item: ResizableRectItem) -> None:
        if item in self._bbox_items:
            self._bbox_items.remove(item)

    def _get_state(self) -> list[tuple[float, float, float, float, str, str]]:
        return [
            (
                float(bb.x1),
                float(bb.y1),
                float(bb.x2),
                float(bb.y2),
                it.get_shape().value,
                it.get_class_name(),
            )
            for it in self._bbox_items
            for bb in [it.get_bounding_box()]
        ]

    def _set_state(self, state: list[tuple[float, float, float, float, str, str]]) -> None:
        self.clear_bboxes()
        if self._image_rect is None:
            return
        for entry in state:
            if len(entry) == 5:
                x1, y1, x2, y2, shape_str = entry  # type: ignore[misc]
                class_name = "Ball"
            else:
                x1, y1, x2, y2, shape_str, class_name = entry  # type: ignore[misc]
            rect = QRectF(float(x1), float(y1), float(x2 - x1), float(y2 - y1))
            shape = Shape.from_string(shape_str)
            item = ResizableRectItem(
                rect,
                self._image_rect,
                shape,
                class_name,
                on_begin_interaction=self._push_undo_snapshot,
                on_changed=None,
                on_deleted=self._on_item_deleted,
                parent=None,
            )
            for key, h in list(item._handles.items()):  # noqa: SLF001
                new_h = HandleItem(item, key)
                new_h.setData(0, "handle")
                new_h.setData(1, key)
                new_h.setBrush(Qt.BrushStyle.NoBrush)
                hpen = QPen(QColor("white"))
                hpen.setCosmetic(True)
                new_h.setPen(hpen)
                new_h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                new_h.setZValue(12.0)
                item._handles[key] = new_h  # noqa: SLF001
                h.setParentItem(None)  # type: ignore[arg-type]
                if item.scene() is not None:
                    item.scene().removeItem(h)
            item._update_overlay_and_handles()  # noqa: SLF001
            item.setPen(self._pen_for_class(class_name))
            self._scene.addItem(item)
            self._bbox_items.append(item)

    def get_current_state(self) -> list[tuple[float, float, float, float, str, str]]:
        return self._get_state()

    def set_current_state(self, state: list[tuple[float, float, float, float, str, str]]) -> None:
        self._set_state(state)

    def _push_undo_snapshot(self) -> None:
        snap = self._get_state()
        if self._undo_stack and self._undo_stack[-1] == snap:
            return
        if len(self._undo_stack) >= self._undo_limit:
            self._undo_stack.pop(0)
        self._undo_stack.append(snap)

    def get_undo_stack(self) -> list[list[tuple[float, float, float, float, str, str]]]:
        return [s.copy() for s in self._undo_stack]

    def set_undo_stack(self, stack: list[list[tuple[float, float, float, float, str, str]]]) -> None:
        self._undo_stack = [s.copy() for s in stack]

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        snap = self._undo_stack.pop()
        self._set_state(snap)
        return True

    def abort_current_edit(self) -> bool:
        aborted = False
        if self._panning:
            self._panning = False
            self._last_pos = None
            self.unsetCursor()
            aborted = True
        if self._rubberband_item is not None:
            if self._rubberband_item.scene() is self._scene:
                self._scene.removeItem(self._rubberband_item)
            self._rubberband_item = None
            self._drag_start = None
            aborted = True
        return aborted

    def _clear_noball_overlay(self) -> None:
        if self._noball_lines is not None:
            line1, line2 = self._noball_lines
            if line1.scene() is self._scene:
                self._scene.removeItem(line1)
            if line2.scene() is self._scene:
                self._scene.removeItem(line2)
            self._noball_lines = None

    def toggle_noball_x(self) -> None:
        if self._image_rect is None:
            return
        if not self._noball_active:
            pen = QPen(QColor("red"))
            pen.setCosmetic(True)
            pen.setWidth(2)
            r = self._image_rect
            line1 = self._scene.addLine(r.left(), r.top(), r.right(), r.bottom(), pen)
            line2 = self._scene.addLine(r.left(), r.bottom(), r.right(), r.top(), pen)
            line1.setZValue(9.0)
            line2.setZValue(9.0)
            self._noball_lines = (line1, line2)
            self._noball_active = True
        else:
            self._clear_noball_overlay()
            self._noball_active = False

    def is_noball_active(self) -> bool:
        """Return whether NoBall X overlay is active."""
        return self._noball_active

    def _enforce_aspect_ratio_for_shape(self, rect: QRectF, shape: Shape) -> QRectF:
        """Enforce 1:1 aspect ratio for circles while keeping rectangle in bounds."""
        if shape != Shape.CIRCLE or self._image_rect is None:
            return rect

        # Use the smaller dimension to create a square
        size = min(rect.width(), rect.height())

        # Center the square on the original rectangle's center
        center = rect.center()
        new_rect = QRectF(center.x() - size / 2, center.y() - size / 2, size, size)

        # Ensure the square stays within image bounds
        new_rect = new_rect.intersected(self._image_rect)

        return new_rect

    def _create_bbox_from_rect(self, scene_rect: QRectF) -> None:
        if self._image_rect is None:
            return
        rect = QRectF(scene_rect).normalized()
        rect = rect.intersected(self._image_rect)
        if rect.width() < 1.0 or rect.height() < 1.0:
            return

        shape = self._shape_provider()
        class_name = self._class_provider()
        self._push_undo_snapshot()
        item = ResizableRectItem(
            rect,
            self._image_rect,
            shape,
            class_name,
            on_begin_interaction=self._push_undo_snapshot,
            on_changed=None,
            on_deleted=self._on_item_deleted,
            parent=None,
        )
        # Replace default simple handles with interactive ones
        for key, h in list(item._handles.items()):  # noqa: SLF001
            # Swap static handle with interactive subclass
            new_h = HandleItem(item, key)
            new_h.setData(0, "handle")
            new_h.setData(1, key)
            new_h.setBrush(Qt.BrushStyle.NoBrush)
            hpen = QPen(QColor("white"))
            hpen.setCosmetic(True)
            new_h.setPen(hpen)
            new_h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            new_h.setZValue(12.0)
            # Position will be updated by the following call
            item._handles[key] = new_h  # noqa: SLF001
            h.setParentItem(None)  # type: ignore[arg-type]
            if item.scene() is not None:
                item.scene().removeItem(h)
        item._update_overlay_and_handles()  # noqa: SLF001
        item.setPen(self._pen_for_class(class_name))

        self._scene.addItem(item)
        self._bbox_items.append(item)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if not self._has_user_transform:
            self._fit()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not self.has_image():
            event.ignore()
            return

        angle = event.angleDelta().y()
        if angle == 0:
            event.ignore()
            return

        factor_step = 1.25
        current_scale = float(self.transform().m11())
        target_scale = current_scale * (factor_step if angle > 0 else 1.0 / factor_step)
        target_scale = max(self._min_scale, min(self._max_scale, target_scale))
        apply_factor = target_scale / current_scale
        if apply_factor != 1.0:
            self._has_user_transform = True
            anchor = self.transformationAnchor()
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.scale(apply_factor, apply_factor)
            self.setTransformationAnchor(anchor)
            ratio = float(self.transform().m11()) / max(self._base_scale, 1e-9)
            self.zoom_changed.emit(ratio)

        event.accept()

    def mapFromScene(self, *args, **kwargs):  # type: ignore[override]
        """Return QPointF for point mapping to ease arithmetic in tests."""
        res = super().mapFromScene(*args, **kwargs)
        if isinstance(res, QPoint):
            return QPointF(res)
        return res

    def _is_interactive_item(self, item: QGraphicsItem | None) -> bool:
        while item is not None:
            data0 = item.data(0)
            if data0 in ("bbox", "handle", b"bbox", b"handle") or str(data0) in ("bbox", "handle"):
                return True
            item = item.parentItem()
        return False

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_pos = QPointF(event.position())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._has_user_transform = True
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.has_image():
                event.ignore()
                return
            sp = self.mapToScene(event.position().toPoint())
            if self._sam_overlay_visible and self._image_rect is not None and self._sam_masks:
                x = int(sp.x())
                y = int(sp.y())
                if 0 <= x < int(self._image_rect.width()) and 0 <= y < int(self._image_rect.height()):
                    for mask in self._sam_masks:
                        try:
                            if bool(mask[y, x]):
                                self._create_bbox_from_mask_point(x, y)
                                event.accept()
                                return
                        except Exception:
                            pass
            hit = self.itemAt(event.position().toPoint())
            if self._is_interactive_item(hit):
                super().mousePressEvent(event)
                return
            self._drag_start = self.mapToScene(event.position().toPoint())
            shape = self._shape_provider()
            class_name = self._class_provider()
            pen = self._pen_for_class(class_name)
            pen.setStyle(Qt.PenStyle.SolidLine)
            if self._rubberband_item is None:
                self._rubberband_item = RubberbandItem(
                    QRectF(self._drag_start, self._drag_start), shape, class_name, pen
                )
                self._scene.addItem(self._rubberband_item)
            else:
                self._rubberband_item.setPen(pen)
                self._rubberband_item.set_shape(shape)
                self._rubberband_item.set_class_name(class_name)
                self._rubberband_item.setRect(QRectF(self._drag_start, self._drag_start))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._panning and self._last_pos is not None:
            delta = QPointF(event.position()) - self._last_pos
            self._last_pos = QPointF(event.position())
            sx = float(self.transform().m11()) or 1.0
            sy = float(self.transform().m22()) or 1.0
            self.translate(-delta.x() / sx, -delta.y() / sy)
            event.accept()
            return
        if self._drag_start is not None and self._rubberband_item is not None:
            current = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._drag_start, current).normalized()
            if self._image_rect is not None:
                rect = rect.intersected(self._image_rect)

            # Enforce aspect ratio for circles during drag
            shape = self._shape_provider()
            rect = self._enforce_aspect_ratio_for_shape(rect, shape)

            self._rubberband_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self._last_pos = None
            self.unsetCursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            current = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._drag_start, current).normalized()
            if self._image_rect is not None:
                rect = rect.intersected(self._image_rect)

            # Enforce aspect ratio for circles before creating final bbox
            shape = self._shape_provider()
            rect = self._enforce_aspect_ratio_for_shape(rect, shape)

            if self._rubberband_item is not None:
                if self._rubberband_item.scene() is self._scene:
                    self._scene.removeItem(self._rubberband_item)
                self._rubberband_item = None
            self._drag_start = None
            if rect.width() >= 1.0 and rect.height() >= 1.0:
                self._create_bbox_from_rect(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _fit(self) -> None:
        if self._pixmap_item is None:
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._base_scale = float(self.transform().m11())
        self._has_user_transform = False
        self.zoom_changed.emit(1.0)

    def _clear_sam_overlay(self) -> None:
        if self._sam_items:
            for it in list(self._sam_items):
                if it.scene() is self._scene:
                    self._scene.removeItem(it)
            self._sam_items.clear()
        self._sam_masks.clear()
        self._sam_overlay_visible = False

    def set_sam_masks(self, masks: list[np.ndarray]) -> None:
        self._clear_sam_overlay()
        if self._image_rect is None:
            return
        width = int(self._image_rect.width())
        height = int(self._image_rect.height())
        for m in masks:
            try:
                if m.shape[0] != height or m.shape[1] != width:
                    continue
            except Exception:
                continue
            arr = np.zeros((height, width, 4), dtype=np.uint8)
            arr[..., 1] = 255
            arr[m.astype(bool), 3] = 100
            image = QImage(arr.data, width, height, int(arr.strides[0]), QImage.Format.Format_RGBA8888).copy()
            pix = QPixmap.fromImage(image)
            item = self._scene.addPixmap(pix)
            item.setZValue(8.0)
            item.setData(0, "sam_mask")
            self._sam_items.append(item)
            self._sam_masks.append(m.astype(bool))
        self._sam_overlay_visible = bool(self._sam_items)

    def show_sam_overlay(self) -> None:
        for it in self._sam_items:
            it.setVisible(True)
        self._sam_overlay_visible = bool(self._sam_items)

    def hide_sam_overlay(self) -> None:
        for it in self._sam_items:
            it.setVisible(False)
        self._sam_overlay_visible = False

    def has_sam_overlay(self) -> bool:
        return self._sam_overlay_visible

    def _pen_for_class(self, class_name: str) -> QPen:
        if class_name == "NoBall":
            color = QColor("red")
        elif class_name == "Ball":
            color = QColor("red")
        else:
            hv = abs(hash(class_name)) % 360
            color = QColor.fromHsv(int(hv), 200, 255)
        pen = QPen(color)
        pen.setCosmetic(True)
        return pen

    def get_image_size(self) -> tuple[int, int]:
        if self._image_rect is None:
            return (0, 0)
        return (int(self._image_rect.width()), int(self._image_rect.height()))

    def delete_selected_bboxes(self) -> bool:
        """Delete currently selected bounding boxes and return True if any were deleted."""
        selected_items = self.scene().selectedItems()
        bbox_items_to_delete = [item for item in selected_items if isinstance(item, ResizableRectItem)]
        if not bbox_items_to_delete:
            return False

        self._push_undo_snapshot()
        for item in bbox_items_to_delete:
            if item in self._bbox_items:
                self._bbox_items.remove(item)
            if item.scene() is self._scene:
                self._scene.removeItem(item)
        return True

    def _create_bbox_from_mask_point(self, x: int, y: int) -> None:
        if not self._sam_masks:
            return
        for m in self._sam_masks:
            try:
                if not bool(m[y, x]):
                    continue
            except Exception:
                continue
            ys, xs = np.where(m)
            if ys.size == 0 or xs.size == 0:
                continue
            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max())
            y2 = int(ys.max())
            rect = QRectF(float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1))
            self._create_bbox_from_rect(rect)
            return


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
