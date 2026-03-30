"""Image processing tools: crop, overlay, grid, composition."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from pydantic import Field


# ── Basic operations ──────────────────────────────────────────────────────────

def crop_image(
    image_path: Annotated[str, Field(description="Path to the source image")],
    bbox: Annotated[List[int], Field(description="[x1, y1, x2, y2] bounding box")],
    output_path: Annotated[str, Field(description="Path to save the cropped image")],
) -> str:
    """Crop a rectangular region from an image and save it."""
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cropped = img[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped)
    return output_path


def white_fill_regions(
    image_path: Annotated[str, Field(description="Path to the source image")],
    bboxes: Annotated[List[List[int]], Field(description="List of [x1,y1,x2,y2] to white-fill")],
    output_path: Annotated[str, Field(description="Path to save the result")],
) -> str:
    """White-fill specified regions (for DI 2-pass detection)."""
    img = cv2.imread(image_path)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        img[y1:y2, x1:x2] = 255
    cv2.imwrite(output_path, img)
    return output_path


# ── Grid overlay ──────────────────────────────────────────────────────────────

def draw_grid_overlay(
    image_path: Annotated[str, Field(description="Path to the source image")],
    output_path: Annotated[str, Field(description="Path to save the grid image")],
    grid_size: int = 120,
) -> str:
    """Draw a numbered grid overlay on an image for LLM spatial reference."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (128, 128, 128)

    for x in range(0, w, grid_size):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
        cv2.putText(overlay, str(x), (x + 2, 16), font, 0.45, color, 1, cv2.LINE_AA)

    for y in range(0, h, grid_size):
        cv2.line(overlay, (0, y), (w, y), color, 1)
        cv2.putText(overlay, str(y), (2, y + 16), font, 0.45, color, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, overlay)
    return output_path


# ── Bbox drawing ──────────────────────────────────────────────────────────────

def draw_bboxes_overlay(
    image_path: Annotated[str, Field(description="Path to the source image")],
    bboxes: Annotated[List[List[int]], Field(description="List of [x1,y1,x2,y2] bboxes")],
    colors: Annotated[List[str], Field(description="BGR hex colors, one per bbox")],
    labels: Annotated[List[str], Field(description="Label text for each bbox")],
    output_path: Annotated[str, Field(description="Path to save the result")],
    thickness: int = 3,
) -> str:
    """Draw labeled bounding boxes on an image."""
    img = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for bbox, color_hex, label in zip(bboxes, colors, labels):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bgr = _hex_to_bgr(color_hex)
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness)
        if label:
            cv2.putText(img, label, (x1, max(y1 - 6, 14)), font, 0.5, bgr, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    return output_path


# ── Locate / Verify input composition ─────────────────────────────────────────

def compose_locate_input(
    page_img_path: Annotated[str, Field(description="Path to the page image")],
    target_bbox: Annotated[List[int], Field(description="[x1,y1,x2,y2] of target panel name")],
    target_label: Annotated[str, Field(description="Panel name label")],
    other_panels: Annotated[Dict[str, List[int]], Field(description="Other panel name→bbox dict")],
    output_path: Annotated[str, Field(description="Output path")],
    grid_size: int = 120,
) -> str:
    """Compose the locate input image with grid + target (blue) + others (green)."""
    img = cv2.imread(page_img_path)
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Grid
    gray = (128, 128, 128)
    for x in range(0, w, grid_size):
        cv2.line(img, (x, 0), (x, h), gray, 1)
        cv2.putText(img, str(x), (x + 2, 16), font, 0.4, gray, 1, cv2.LINE_AA)
    for y in range(0, h, grid_size):
        cv2.line(img, (0, y), (w, y), gray, 1)
        cv2.putText(img, str(y), (2, y + 16), font, 0.4, gray, 1, cv2.LINE_AA)

    # Other panels (green)
    green = (0, 200, 0)
    for name, bbox in other_panels.items():
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), green, 2)
        cv2.putText(img, f"OTHER:{name}", (x1, max(y1 - 4, 14)), font, 0.4, green, 1, cv2.LINE_AA)

    # Target panel (blue)
    blue = (255, 0, 0)
    x1, y1, x2, y2 = [int(v) for v in target_bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), blue, 3)
    cv2.putText(img, f"NAME:{target_label}", (x1, max(y1 - 6, 16)), font, 0.55, blue, 2, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    return output_path


def compose_verify_input(
    page_img_path: Annotated[str, Field(description="Path to the page image")],
    panel_bbox: Annotated[List[int], Field(description="Current panel [x1,y1,x2,y2]")],
    name_bbox: Annotated[List[int], Field(description="Panel name location [x1,y1,x2,y2]")],
    panel_label: Annotated[str, Field(description="Panel name")],
    other_panels: Annotated[Dict[str, List[int]], Field(description="Neighbor panels")],
    output_path: Annotated[str, Field(description="Output path")],
    grid_size: int = 120,
) -> str:
    """Compose the verify input image with current bbox (orange) + name (blue) + neighbors."""
    img = cv2.imread(page_img_path)
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Grid
    gray = (128, 128, 128)
    for x in range(0, w, grid_size):
        cv2.line(img, (x, 0), (x, h), gray, 1)
    for y in range(0, h, grid_size):
        cv2.line(img, (0, y), (w, y), gray, 1)

    # Neighbors (green)
    green = (0, 200, 0)
    for name, bbox in other_panels.items():
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), green, 2)
        cv2.putText(img, f"OTHER:{name}", (x1, max(y1 - 4, 14)), font, 0.4, green, 1, cv2.LINE_AA)

    # Current panel bbox (orange)
    orange = (0, 165, 255)
    px1, py1, px2, py2 = [int(v) for v in panel_bbox]
    cv2.rectangle(img, (px1, py1), (px2, py2), orange, 3)
    cv2.putText(img, "PANEL", (px1, max(py1 - 6, 16)), font, 0.55, orange, 2, cv2.LINE_AA)

    # Name location (blue)
    blue = (255, 0, 0)
    nx1, ny1, nx2, ny2 = [int(v) for v in name_bbox]
    cv2.rectangle(img, (nx1, ny1), (nx2, ny2), blue, 2)
    cv2.putText(img, f"NAME:{panel_label}", (nx1, max(ny1 - 4, 14)), font, 0.45, blue, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    return output_path


# ── Image encoding ────────────────────────────────────────────────────────────

def image_to_data_url(image_path: str) -> str:
    """Read an image file and return as a base64 data URL for LLM input."""
    data = Path(image_path).read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def ndarray_to_data_url(image: np.ndarray) -> str:
    """Encode a numpy image (BGR) as a base64 PNG data URL."""
    _, buffer = cv2.imencode(".png", image)
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ── Batch locate / verify image composition ────────────────────────────────────

def compose_locate_all_input(
    page_img_path: str,
    all_name_bboxes: Dict[str, Optional[List[int]]],
    output_path: str,
    grid_size: int = 120,
) -> str:
    """Compose locate-all input: grid + ALL panel name bboxes (blue, labeled NAME:xxx).

    Used for the batch locate call that finds all panel areas in one LLM request.
    """
    img = cv2.imread(page_img_path)
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray = (128, 128, 128)
    blue = (255, 0, 0)

    for x in range(0, w, grid_size):
        cv2.line(img, (x, 0), (x, h), gray, 1)
        cv2.putText(img, str(x), (x + 2, 16), font, 0.4, gray, 1, cv2.LINE_AA)
    for y in range(0, h, grid_size):
        cv2.line(img, (0, y), (w, y), gray, 1)
        cv2.putText(img, str(y), (2, y + 16), font, 0.4, gray, 1, cv2.LINE_AA)

    for name, bbox in all_name_bboxes.items():
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), blue, 3)
            cv2.putText(img, f"NAME:{name}", (x1, max(y1 - 6, 16)),
                        font, 0.55, blue, 2, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    return output_path


def compose_verify_all_overlay(
    page_img_path: str,
    panel_bboxes: Dict[str, List[int]],
    all_name_bboxes: Dict[str, Optional[List[int]]],
    output_path: str,
    grid_size: int = 120,
) -> str:
    """Compose verify-all overlay: grid + proposed PANEL bboxes (orange) + NAME bboxes (blue).

    Used for the batch verify call to show spatial relationships between panels.
    """
    img = cv2.imread(page_img_path)
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    gray = (128, 128, 128)
    orange = (0, 165, 255)
    blue = (255, 0, 0)

    for x in range(0, w, grid_size):
        cv2.line(img, (x, 0), (x, h), gray, 1)
    for y in range(0, h, grid_size):
        cv2.line(img, (0, y), (w, y), gray, 1)

    for name, bbox in panel_bboxes.items():
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), orange, 3)
            cv2.putText(img, f"PANEL:{name}", (x1, max(y1 - 6, 16)),
                        font, 0.55, orange, 2, cv2.LINE_AA)

    for name, bbox in all_name_bboxes.items():
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), blue, 2)
            cv2.putText(img, f"NAME:{name}", (x1, max(y1 - 4, 14)),
                        font, 0.45, blue, 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
