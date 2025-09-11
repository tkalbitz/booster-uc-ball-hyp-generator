def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value between minimum and maximum bounds."""
    return max(min_value, min(value, max_value))
