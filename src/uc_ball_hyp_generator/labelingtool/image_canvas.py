from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from uc_ball_hyp_generator.labelingtool.model import BoundingBox, Shape
from uc_ball_hyp_generator.labelingtool.resizable_rect_item import ResizableRectItem
from uc_ball_hyp_generator.labelingtool.rubberband_item import RubberbandItem
from uc_ball_hyp_generator.labelingtool.utils.logger import get_logger

_logger = get_logger("uc_ball_hyp_generator.labelingtool.gui")


class HandleItem(QGraphicsRectItem):
    """Interactive handle forwarding drag events to the parent ResizableRectItem."""

    def __init__(self, parent_rect: ResizableRectItem, key: str) -> None:
        super().__init__(parent_rect)
        self._parent_rect = parent_rect
        self._key = key
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self._parent_rect.begin_interaction()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        self._parent_rect.resize_from_handle(self._key, event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        event.accept()

    def hoverEnterEvent(self, event) -> None:  # type: ignore[override]
        if self._key in ("nw", "se"):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif self._key in ("ne", "sw"):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif self._key in ("e", "w"):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif self._key in ("n", "s"):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # type: ignore[override]
        self.unsetCursor()
        super().hoverLeaveEvent(event)


class _ViewportWidget(QWidget):
    """Return QRectF from rect() so center() yields QPointF in tests."""

    def rect(self) -> QRectF:  # type: ignore[override]
        return QRectF(super().rect())


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

    def _calculate_circle_from_circumference_point(
        self, circumference_point: QPointF, current_point: QPointF
    ) -> QRectF:
        """Calculate bounding box for circle where circumference_point is on the circle's edge."""
        # Calculate center as the point that makes circumference_point equidistant
        # For simplicity, we'll use current_point as center and circumference_point to determine radius
        center = current_point
        radius = ((circumference_point.x() - center.x()) ** 2 + (circumference_point.y() - center.y()) ** 2) ** 0.5

        # Create bounding box from center and radius
        rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)

        # Ensure the circle stays within image bounds
        if self._image_rect is not None:
            rect = rect.intersected(self._image_rect)

        return rect

    def _calculate_ellipse_from_circumference_point(
        self, circumference_point: QPointF, current_point: QPointF
    ) -> QRectF:
        """Calculate bounding box for ellipse where circumference_point is on the ellipse's edge."""
        # For ellipses, we'll use the drag vector to determine the ellipse orientation
        # circumference_point is on the edge, current_point helps define the ellipse
        center_x = (circumference_point.x() + current_point.x()) / 2
        center_y = (circumference_point.y() + current_point.y()) / 2

        # Calculate radii from the center to both points
        dx = abs(current_point.x() - center_x)
        dy = abs(current_point.y() - center_y)

        # Ensure minimum size
        radius_x = max(dx, 1.0)
        radius_y = max(dy, 1.0)

        rect = QRectF(center_x - radius_x, center_y - radius_y, radius_x * 2, radius_y * 2)

        # Ensure the ellipse stays within image bounds
        if self._image_rect is not None:
            rect = rect.intersected(self._image_rect)

        return rect

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
            if self._rubberband_item is None and self._drag_start:
                self._rubberband_item = RubberbandItem(
                    QRectF(self._drag_start, self._drag_start), shape, class_name, pen
                )
                self._scene.addItem(self._rubberband_item)
            elif self._rubberband_item and self._drag_start:
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
            shape = self._shape_provider()

            # Calculate rectangle based on shape type
            if shape == Shape.CIRCLE:
                rect = self._calculate_circle_from_circumference_point(self._drag_start, current)
            elif shape == Shape.ELLIPSE:
                rect = self._calculate_ellipse_from_circumference_point(self._drag_start, current)
            else:
                # Rectangle: use traditional bounding box approach
                rect = QRectF(self._drag_start, current).normalized()
                if self._image_rect is not None:
                    rect = rect.intersected(self._image_rect)

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
            shape = self._shape_provider()

            # Calculate final rectangle based on shape type
            if shape == Shape.CIRCLE:
                rect = self._calculate_circle_from_circumference_point(self._drag_start, current)
            elif shape == Shape.ELLIPSE:
                rect = self._calculate_ellipse_from_circumference_point(self._drag_start, current)
            else:
                # Rectangle: use traditional bounding box approach
                rect = QRectF(self._drag_start, current).normalized()
                if self._image_rect is not None:
                    rect = rect.intersected(self._image_rect)

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
