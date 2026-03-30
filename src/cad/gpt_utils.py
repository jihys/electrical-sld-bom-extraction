"""GPT/OpenAI vision utility functions for coordinate and bounding box tests."""

import base64
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ── Color helpers ─────────────────────────────────────────────────────────────

_NAMED_COLORS_BGR = {
    "red":      (0, 0, 255),
    "blue":     (255, 0, 0),
    "green":    (0, 255, 0),
    "orange":   (0, 165, 255),
    "yellow":   (0, 255, 255),
    "cyan":     (255, 255, 0),
    "magenta":  (255, 0, 255),
    "purple":   (128, 0, 128),
    "gray":     (128, 128, 128),
    "grey":     (128, 128, 128),
    "darkgray": (64, 64, 64),
    "darkgrey": (64, 64, 64),
    "white":    (255, 255, 255),
    "black":    (0, 0, 0),
}


def color_to_bgr(color) -> Tuple[int, int, int]:
    if isinstance(color, tuple):
        return color
    c = color.lower().strip()
    if c in _NAMED_COLORS_BGR:
        return _NAMED_COLORS_BGR[c]
    h = c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


# ── Image encoding ────────────────────────────────────────────────────────────

def file_to_data_url(path: Path) -> str:
    """Read an image file and return as a base64 data URL."""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def image_to_data_url(image: np.ndarray) -> str:
    """Encode a numpy image array as a base64 PNG data URL.

    Passes the image directly to cv2.imencode without BGR→RGB conversion,
    preserving the original channel order as stored in memory.
    """
    _, buffer = cv2.imencode(".png", image)
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json(text: str) -> Optional[Dict]:
    """Extract and parse the first JSON object found in a text string."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ── Coordinate normalization ──────────────────────────────────────────────────

def normalize_point(point: Dict, width: int, height: int) -> Tuple[float, float]:
    """Normalize a point dict {x, y} from [0,1] to pixel coordinates if needed."""
    x = float(point.get("x"))
    y = float(point.get("y"))
    if 0 <= x <= 1 and 0 <= y <= 1:
        return x * width, y * height
    return x, y


def normalize_bbox(
    bbox: Sequence[float], width: int, height: int
) -> Tuple[float, float, float, float]:
    """Normalize a bbox [x1,y1,x2,y2] from [0,1] to pixel coordinates if needed."""
    if len(bbox) != 4:
        raise ValueError("bbox must have 4 values [x1,y1,x2,y2]")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
        x1, y1, x2, y2 = x1 * width, y1 * height, x2 * width, y2 * height
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def polygon_to_bbox(polygon: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert a flat polygon [x1,y1,...,xn,yn] to its bounding box."""
    if len(polygon) < 8 or len(polygon) % 2 != 0:
        raise ValueError("polygon must have even number of values (>= 8)")
    xs = polygon[0::2]
    ys = polygon[1::2]
    return min(xs), min(ys), max(xs), max(ys)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_point(
    image: np.ndarray, point: Tuple[float, float], color, radius: int = 8
) -> np.ndarray:
    """Draw a colored circle at the given point. Returns a copy of the image."""
    overlay = image.copy()
    x, y = int(point[0]), int(point[1])
    cv2.circle(overlay, (x, y), radius, color_to_bgr(color), thickness=4)
    return overlay


def draw_bbox(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    color,
    width: int = 4,
) -> np.ndarray:
    """Draw a colored rectangle for the given bbox. Returns a copy of the image."""
    overlay = image.copy()
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_to_bgr(color), thickness=width)
    return overlay


def show_image(image: np.ndarray, title: str) -> None:
    """Display an OpenCV BGR image using matplotlib."""
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


# ── IoU metric ────────────────────────────────────────────────────────────────

def bbox_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union between two axis-aligned bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0
