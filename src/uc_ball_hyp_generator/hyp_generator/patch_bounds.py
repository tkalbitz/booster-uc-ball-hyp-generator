"""Shared utility functions for calculating patch bounds."""

from uc_ball_hyp_generator.hyp_generator.config import patch_height, patch_width


def get_boundary_aware_patch_bounds(
    center_x: float, center_y: float, img_width: int, img_height: int
) -> tuple[int, int]:
    """Calculate patch bounds ensuring patch stays within image boundaries.

    If patch extends beyond image border, the outer edge aligns with the border
    and start position is calculated backwards.

    Args:
        center_x: Target center x coordinate
        center_y: Target center y coordinate
        img_width: Image width
        img_height: Image height

    Returns:
        Tuple of (start_x, start_y) for patch extraction
    """
    # Initial patch position based on center
    start_x = int(center_x - patch_width / 2)
    start_y = int(center_y - patch_height / 2)

    # Adjust if patch extends beyond right/bottom borders
    if start_x + patch_width > img_width:
        start_x = img_width - patch_width
    if start_y + patch_height > img_height:
        start_y = img_height - patch_height

    # Ensure start positions are non-negative
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    return start_x, start_y
