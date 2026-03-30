"""Geometry helper tools: IoU, containment, deduplication."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..models.region import RegionRecord


Bbox = Tuple[int, int, int, int]


def compute_iou(a: Bbox, b: Bbox) -> float:
    """Compute Intersection-over-Union between two axis-aligned bboxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_contains(outer: Bbox, inner: Bbox) -> bool:
    """True if outer bbox fully contains inner bbox."""
    return (outer[0] <= inner[0] and outer[1] <= inner[1] and
            outer[2] >= inner[2] and outer[3] >= inner[3])


def bbox_intersect(a: Bbox, b: Bbox) -> Optional[Bbox]:
    """Return intersection bbox, or None if no overlap."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def clip_to_content_bbox(bbox: Bbox, content: Bbox) -> Bbox:
    """Clip a bbox to fit within a content boundary."""
    return (
        max(bbox[0], content[0]),
        max(bbox[1], content[1]),
        min(bbox[2], content[2]),
        min(bbox[3], content[3]),
    )


def bbox_area(bbox: Bbox) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def dedup_regions(
    regions: List[RegionRecord],
    iou_threshold: float = 0.5,
) -> List[RegionRecord]:
    """Remove overlapping regions, keeping the larger one."""
    sorted_regions = sorted(regions, key=lambda r: bbox_area(r.bbox), reverse=True)
    kept: List[RegionRecord] = []
    for region in sorted_regions:
        is_dup = False
        for k in kept:
            if compute_iou(region.bbox, k.bbox) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(region)
    return kept


def check_panel_overlaps(
    panels: List[Tuple[str, Bbox]],
) -> List[Tuple[str, str, float]]:
    """Find overlapping panel pairs. Returns [(name1, name2, iou), ...]."""
    overlaps = []
    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            iou = compute_iou(panels[i][1], panels[j][1])
            if iou > 0.05:
                overlaps.append((panels[i][0], panels[j][0], iou))
    return overlaps
