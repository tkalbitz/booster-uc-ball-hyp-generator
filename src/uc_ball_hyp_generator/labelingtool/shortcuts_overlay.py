from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


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
