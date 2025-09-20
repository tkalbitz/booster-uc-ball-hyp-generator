from dataclasses import dataclass


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
