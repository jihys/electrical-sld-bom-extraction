"""
data_types.py
-------------
Core data structures and polygon I/O utilities.
Extracted from 02_missing_image_detection.ipynb.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .image_utils import draw_polygon, hex_to_bgr


@dataclass
class ImageCropInfo:
    page_num: int
    index: int
    image_path: Path
    coords_path: Optional[Path]
    polygon: Optional[List[float]]
    bbox: Optional[Tuple[int, int, int, int]]
    is_existing: bool = True


def read_polygon_txt(path: Path) -> List[float]:
    coords: List[float] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 2:
            continue
        coords.extend([float(parts[0]), float(parts[1])])
    return coords


def write_polygon_txt(path: Path, polygon: Sequence[float]) -> None:
    lines = [f"{polygon[i]:.2f},{polygon[i + 1]:.2f}" for i in range(0, len(polygon), 2)]
    path.write_text("\n".join(lines))


def polygon_to_bbox(
    polygon: Sequence[float], image_size: Tuple[int, int], padding: int = 0
) -> Optional[Tuple[int, int, int, int]]:
    if not polygon:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    left = max(int(min(xs)) - padding, 0)
    top = max(int(min(ys)) - padding, 0)
    right = min(int(max(xs)) + padding, image_size[0])
    bottom = min(int(max(ys)) + padding, image_size[1])
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def crop_polygon(image: np.ndarray, polygon: Sequence[float]) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    bbox = polygon_to_bbox(polygon, (w, h), padding=0)
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2].copy()


def _parse_crop_index(path: Path, page_num: int) -> Optional[int]:
    match = re.match(r"images?_(\d+)_(\d+)(?:_temp)?$", path.stem)
    if not match:
        return None
    if int(match.group(1)) != page_num:
        return None
    return int(match.group(2))


def _normalize_polygon(polygon: Sequence[float]) -> Optional[List[float]]:
    if not polygon or len(polygon) < 8 or len(polygon) % 2 != 0:
        return None
    return [float(value) for value in polygon]


def draw_overlay(base_image: np.ndarray, polygons: List[Sequence[float]]) -> np.ndarray:
    overlay = base_image.copy()
    for polygon in polygons:
        draw_polygon(overlay, polygon, color=hex_to_bgr("#f97316"), width=10)
    return overlay


def collect_polygons(crops: List[ImageCropInfo]) -> List[Sequence[float]]:
    return [crop.polygon for crop in crops if crop.polygon]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def bbox_from_polygon(polygon: Sequence[float]) -> Tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) from a flat polygon [x0,y0,x1,y1,...]."""
    xs = [int(polygon[i]) for i in range(0, len(polygon), 2)]
    ys = [int(polygon[i]) for i in range(1, len(polygon), 2)]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_from_bbox(x1: int, y1: int, x2: int, y2: int) -> List[int]:
    """Return a flat 4-point polygon from a bounding box."""
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def bbox_intersect(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> Optional[Tuple[int, int, int, int]]:
    """Return intersection bbox of a and b, or None if they don't overlap."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def bbox_contains(
    outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]
) -> bool:
    """Return True if outer bbox fully contains inner bbox."""
    return (outer[0] <= inner[0] and outer[1] <= inner[1] and
            outer[2] >= inner[2] and outer[3] >= inner[3])
