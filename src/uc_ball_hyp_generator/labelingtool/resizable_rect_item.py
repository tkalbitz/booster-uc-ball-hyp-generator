from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsRectItem

from uc_ball_hyp_generator.labelingtool.bounding_box import BoundingBox
from uc_ball_hyp_generator.labelingtool.shape import Shape


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

        if self._shape in (Shape.CIRCLE, Shape.ELLIPSE):
            # For circles and ellipses, maintain circumference-based behavior
            current_rect = QRectF(self.pos(), self.rect().size())
            current_center = current_rect.center()

            # Clamp the drag position to bounds
            px = max(b.left(), min(scene_pos.x(), b.right()))
            py = max(b.top(), min(scene_pos.y(), b.bottom()))
            clamped_pos = QPointF(px, py)

            if self._shape == Shape.CIRCLE:
                # For circles, calculate radius from center to drag point
                radius = (
                    (clamped_pos.x() - current_center.x()) ** 2 + (clamped_pos.y() - current_center.y()) ** 2
                ) ** 0.5
                radius = max(1.0, radius)  # Ensure minimum size

                # Create new rectangle centered on current center
                new_rect = QRectF(current_center.x() - radius, current_center.y() - radius, radius * 2, radius * 2)
            else:
                # For ellipses, adjust radii based on which handle is being dragged
                dx = abs(clamped_pos.x() - current_center.x())
                dy = abs(clamped_pos.y() - current_center.y())

                # Determine which radius to adjust based on handle direction
                if "w" in key or "e" in key:
                    # Horizontal handles - adjust width
                    radius_x = max(1.0, dx)
                    radius_y = current_rect.height() / 2
                else:
                    # Vertical handles - adjust height
                    radius_x = current_rect.width() / 2
                    radius_y = max(1.0, dy)

                new_rect = QRectF(
                    current_center.x() - radius_x, current_center.y() - radius_y, radius_x * 2, radius_y * 2
                )

            # Ensure the new rectangle stays within bounds
            new_rect = new_rect.intersected(b)

            self.setPos(new_rect.topLeft())
            self.setRect(QRectF(0.0, 0.0, new_rect.width(), new_rect.height()))
        else:
            # Rectangle: use traditional corner-based resize
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
