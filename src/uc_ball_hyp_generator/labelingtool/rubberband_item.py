from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QWidget

from uc_ball_hyp_generator.labelingtool.model import Shape


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
