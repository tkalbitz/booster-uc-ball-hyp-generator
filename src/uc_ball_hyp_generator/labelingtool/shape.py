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
