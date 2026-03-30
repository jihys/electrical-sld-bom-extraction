"""Grid overlay drawing utilities for coordinate visualization."""

from typing import Tuple

import cv2
import numpy as np

from .gpt_utils import color_to_bgr


def draw_ruler(
    image: np.ndarray,
    step_x: int = 100,
    step_y: int = 100,
    color=(160, 160, 160),
    tick_len: int = 8,
    font_scale: float = 0.38,
) -> np.ndarray:
    """Draw ruler tick marks and coordinate numbers only along the four edges of the image.
    No lines are drawn in the interior, so the CAD drawing content is not obscured.
    """
    overlay = image.copy()
    h, w = overlay.shape[:2]
    bgr = color if isinstance(color, tuple) else color_to_bgr(color)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top and bottom ticks (x-axis)
    for x in range(0, w, step_x):
        # Top
        cv2.line(overlay, (x, 0), (x, tick_len), bgr, 1)
        cv2.putText(overlay, str(x), (x + 2, tick_len + 10),
                    font, font_scale, bgr, 1, cv2.LINE_AA)
        # Bottom
        cv2.line(overlay, (x, h - 1), (x, h - 1 - tick_len), bgr, 1)

    # Left and right ticks (y-axis)
    for y in range(0, h, step_y):
        # Left
        cv2.line(overlay, (0, y), (tick_len, y), bgr, 1)
        cv2.putText(overlay, str(y), (tick_len + 2, y + 10),
                    font, font_scale, bgr, 1, cv2.LINE_AA)
        # Right
        cv2.line(overlay, (w - 1, y), (w - 1 - tick_len, y), bgr, 1)

    return overlay


def draw_grid(
    image: np.ndarray,
    step_x: int = 100,
    step_y: int = 100,
    color="gray",
    line_width: int = 1,
    label: bool = True,
    font_scale: float = 0.45,
) -> np.ndarray:
    """Draw a uniform-spacing grid with pixel-coordinate axis labels on an image."""
    overlay = image.copy()
    h, w = overlay.shape[:2]
    bgr = color_to_bgr(color)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for x in range(0, w, step_x):
        cv2.line(overlay, (x, 0), (x, h), bgr, line_width)
        if label:
            cv2.putText(overlay, str(x), (x + 2, 18), font, font_scale, bgr, 1, cv2.LINE_AA)

    for y in range(0, h, step_y):
        cv2.line(overlay, (0, y), (w, y), bgr, line_width)
        if label:
            cv2.putText(overlay, str(y), (2, y + 18), font, font_scale, bgr, 1, cv2.LINE_AA)

    return overlay


def draw_grid_with_cell_numbers(
    image: np.ndarray,
    step_x: int = 100,
    step_y: int = 100,
    grid_color="gray",
    cell_color="darkgray",
    line_width: int = 1,
    axis_font_scale: float = 0.45,
    cell_font_scale: float = 0.35,
) -> np.ndarray:
    """Draw a grid with axis labels and (col,row) cell-center labels on an image."""
    overlay = image.copy()
    h, w = overlay.shape[:2]
    grid_bgr = color_to_bgr(grid_color)
    cell_bgr = color_to_bgr(cell_color)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for x in range(0, w, step_x):
        cv2.line(overlay, (x, 0), (x, h), grid_bgr, line_width)
        cv2.putText(overlay, str(x), (x + 2, 18), font, axis_font_scale, grid_bgr, 1, cv2.LINE_AA)

    for y in range(0, h, step_y):
        cv2.line(overlay, (0, y), (w, y), grid_bgr, line_width)
        cv2.putText(overlay, str(y), (2, y + 18), font, axis_font_scale, grid_bgr, 1, cv2.LINE_AA)

    num_cols = (w + step_x - 1) // step_x
    num_rows = (h + step_y - 1) // step_y

    for row in range(num_rows):
        for col in range(num_cols):
            cx = col * step_x + step_x // 2
            cy = row * step_y + step_y // 2
            if cx >= w or cy >= h:
                continue
            lbl = f"({col},{row})"
            (tw, th), _ = cv2.getTextSize(lbl, font, cell_font_scale, 1)
            cv2.putText(
                overlay, lbl, (cx - tw // 2, cy + th // 2),
                font, cell_font_scale, cell_bgr, 1, cv2.LINE_AA,
            )

    return overlay
