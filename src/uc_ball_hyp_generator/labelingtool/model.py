from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Shape(str, Enum):
    """
    Supported annotation shapes for visualization.

    - RECTANGLE: Draw a rectangle.
    - CIRCLE: Draw a circle that fits inside the rectangle bounds.
    - ELLIPSE: Draw an ellipse inside the rectangle bounds (default for Ball).
    """

    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"

    @classmethod
    def from_string(cls, value: str) -> "Shape":
        """
        Convert a user/config string to a Shape enum instance (case-insensitive).

        Args:
            value: String representation ("rectangle", "circle", or "ellipse").

        Returns:
            Shape: Matching enum value.

        Raises:
            ValueError: If the string does not match any supported shape.
        """
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except Exception as exc:  # noqa: BLE001
            msg = f"Invalid shape '{value}'. Supported: rectangle, circle, ellipse."
            raise ValueError(msg) from exc


@dataclass(frozen=True)
class BoundingBox:
    """
    Immutable data model describing a labeled bounding box.

    Coordinates are integer pixel positions (top-left inclusive, bottom-right inclusive).
    """

    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str = "Ball"
    subclass: str | None = "Ball"

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return the coordinates as a tuple (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)
