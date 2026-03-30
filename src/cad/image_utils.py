"""Image processing and GPT helper utilities."""

import base64
import json
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

GRID_STEP_X = 100
GRID_STEP_Y = 100

_GRID_COLOR = (128, 128, 128)  # BGR gray
_CELL_COLOR = (64, 64, 64)     # BGR dark gray


def draw_grid_with_cell_numbers(
    image: np.ndarray,
    step_x: int = 200,
    step_y: int = 200,
    grid_color: Tuple[int, int, int] = _GRID_COLOR,
    cell_color: Tuple[int, int, int] = _CELL_COLOR,
    line_width: int = 1,
    axis_font_scale: float = 0.55,
    cell_font_scale: float = 0.45,
) -> np.ndarray:
    """Draw grid + axis coordinate labels + (col, row) cell numbers on the image.

    GPT input overlay only — do not apply to saved/display images.
    (00_gpt_coordinate_tests.ipynb Test 5: confirmed +9.4% IoU improvement)
    """
    overlay = image.copy()
    h, w = overlay.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for x in range(0, w, step_x):
        cv2.line(overlay, (x, 0), (x, h), grid_color, line_width)
        cv2.putText(overlay, str(x), (x + 2, 18), font, axis_font_scale, grid_color, 1, cv2.LINE_AA)

    for y in range(0, h, step_y):
        cv2.line(overlay, (0, y), (w, y), grid_color, line_width)
        cv2.putText(overlay, str(y), (2, y + 18), font, axis_font_scale, grid_color, 1, cv2.LINE_AA)

    num_cols = (w + step_x - 1) // step_x
    num_rows = (h + step_y - 1) // step_y
    for row in range(num_rows):
        for col in range(num_cols):
            cx = col * step_x + step_x // 2
            cy = row * step_y + step_y // 2
            if cx >= w or cy >= h:
                continue
            label = f"({col},{row})"
            (tw, th), _ = cv2.getTextSize(label, font, cell_font_scale, 1)
            tx = cx - tw // 2
            ty = cy + th // 2
            cv2.putText(overlay, label, (tx, ty), font, cell_font_scale, cell_color, 1, cv2.LINE_AA)

    return overlay


def hex_to_bgr(color: str) -> Tuple[int, int, int]:
    """Convert hex color string to OpenCV BGR tuple."""
    h = color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_polygon(
    image: np.ndarray,
    polygon: Sequence[float],
    color: Tuple[int, int, int],
    width: int = 3,
) -> None:
    """Draw a closed polygon onto image in-place."""
    if polygon:
        pts = np.array(
            [[int(polygon[i]), int(polygon[i + 1])] for i in range(0, len(polygon), 2)],
            dtype=np.int32,
        )
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=width)


def file_to_data_url(path: Path) -> str:
    """Read an image file and return as base64 PNG data URL."""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def image_to_data_url(image: np.ndarray) -> str:
    """Convert an OpenCV BGR ndarray to a base64 PNG data URL.

    Converts BGR→RGB before encoding so GPT receives correct colors.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode(".png", rgb_image)
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def image_to_data_url_with_grid(
    image: np.ndarray,
    step_x: int = 200,
    step_y: int = 200,
) -> str:
    """Apply grid overlay and return as base64 PNG data URL for GPT input.

    The original image is not modified.
    """
    grid_image = draw_grid_with_cell_numbers(image, step_x=step_x, step_y=step_y)
    return image_to_data_url(grid_image)


def extract_json(text: str) -> Optional[Dict]:
    """Extract the first JSON object from a text string."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_bbox_pixels(
    bbox,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Convert normalized (0–1) or pixel coordinates to pixel coordinates."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
        x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


# ── Frame / title-block detection ────────────────────────────────────────────

def detect_content_bbox(
    img: np.ndarray,
    frame_span: float = 0.90,
) -> Tuple[int, int, int, int]:
    """Detect the content bounding box by stripping the outer border frame.

    Algorithm
    ---------
    Find horizontal / vertical lines that span > *frame_span* of the image
    dimension — these form the outer border frame.  Returns the rectangle
    just inside those four border lines.  If any of the four sides cannot be
    detected the full image extents (0, 0, w, h) are returned.

    Parameters
    ----------
    frame_span:
        Minimum fraction of image dimension a line must span to count as a
        frame border (default 0.90 = 90 %).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # ── Outer frame ──────────────────────────────────────────────────────────
    h_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * frame_span), 1))
    v_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * frame_span)))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_ker)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_ker)

    h_rows = np.where(h_lines.sum(axis=1) > 0)[0]
    v_cols = np.where(v_lines.sum(axis=0) > 0)[0]

    # Require all 4 sides within 5% of each edge
    _margin = 0.05
    top_rows   = h_rows[h_rows < int(h * _margin)]
    bot_rows   = h_rows[h_rows > int(h * (1 - _margin))]
    left_cols  = v_cols[v_cols < int(w * _margin)]
    right_cols = v_cols[v_cols > int(w * (1 - _margin))]

    if not all(len(x) > 0 for x in [top_rows, bot_rows, left_cols, right_cols]):
        return 0, 0, w, h  # incomplete frame — at least one side missing

    top   = int(np.max(top_rows))
    bot   = int(np.min(bot_rows))
    left  = int(np.max(left_cols))
    right = int(np.min(right_cols))

    return left, top, right, bot


def crop_to_content(
    img: np.ndarray,
    **kwargs,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Return (cropped_image, (x1, y1, x2, y2)) after removing frame/table.

    All keyword arguments are forwarded to :func:`detect_content_bbox`.
    """
    x1, y1, x2, y2 = detect_content_bbox(img, **kwargs)
    return img[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)
