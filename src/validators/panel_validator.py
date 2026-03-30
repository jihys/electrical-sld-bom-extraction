"""Validation logic for panel crops, names, edges, and cross-panel consistency."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..models.panel import PanelCrop
from ..tools.geometry_tools import Bbox, bbox_area, bbox_contains, check_panel_overlaps, compute_iou


# ──────────────────────────────────────────────────────────────────────────────
# Crop validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_crop(
    crop: PanelCrop,
    page_width: int,
    page_height: int,
    name_bbox: Optional[Tuple[int, int, int, int]] = None,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.80,
) -> Tuple[bool, List[str]]:
    """Validate a panel crop for basic sanity.

    Returns (is_valid, list_of_issues).
    """
    issues: List[str] = []
    page_area = page_width * page_height

    area = bbox_area(crop.bbox)
    ratio = area / page_area if page_area > 0 else 0

    if ratio < min_area_ratio:
        issues.append(f"Panel area too small ({ratio:.4f} of page)")
    if ratio > max_area_ratio:
        issues.append(f"Panel area too large ({ratio:.2f} of page)")

    # Must contain name bbox
    if name_bbox and not bbox_contains(crop.bbox, name_bbox):
        issues.append("Panel bbox does not contain name bbox")

    # Width/height sanity
    w = crop.bbox[2] - crop.bbox[0]
    h = crop.bbox[3] - crop.bbox[1]
    if w <= 0 or h <= 0:
        issues.append("Panel has zero or negative dimensions")

    aspect = max(w, h) / min(w, h) if min(w, h) > 0 else float("inf")
    if aspect > 20:
        issues.append(f"Extreme aspect ratio ({aspect:.1f})")

    return (len(issues) == 0, issues)


# ──────────────────────────────────────────────────────────────────────────────
# Name validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_name_coverage(
    expected_names: List[str],
    detected_crops: List[PanelCrop],
) -> Tuple[List[str], List[str]]:
    """Check that all expected panel names have a corresponding crop.

    Returns (missing_names, extra_names).
    """
    detected_names = {c.panel_name for c in detected_crops}
    expected_set = set(expected_names)

    missing = sorted(expected_set - detected_names)
    extra = sorted(detected_names - expected_set)
    return missing, extra


# ──────────────────────────────────────────────────────────────────────────────
# Cross-panel validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_cross_panel(
    crops: List[PanelCrop],
    page_width: int,
    page_height: int,
    max_overlap_iou: float = 0.15,
    min_coverage_ratio: float = 0.30,
) -> Tuple[bool, List[str]]:
    """Validate cross-panel consistency: overlaps, gaps, coverage.

    Returns (is_valid, list_of_issues).
    """
    issues: List[str] = []

    if not crops:
        issues.append("No panels detected")
        return False, issues

    # Check pairwise overlaps
    panel_pairs = [(c.panel_name, c.bbox) for c in crops]
    overlaps = check_panel_overlaps(panel_pairs)
    for n1, n2, iou in overlaps:
        if iou > max_overlap_iou:
            issues.append(f"Panels '{n1}' and '{n2}' overlap significantly (IoU={iou:.3f})")

    # Check total coverage
    page_area = page_width * page_height
    total_panel_area = sum(bbox_area(c.bbox) for c in crops)
    coverage = total_panel_area / page_area if page_area > 0 else 0

    if coverage < min_coverage_ratio:
        issues.append(f"Low page coverage ({coverage:.2%}) — panels may be missing")

    return (len(issues) == 0, issues)


# ──────────────────────────────────────────────────────────────────────────────
# Confidence-based HITL routing
# ──────────────────────────────────────────────────────────────────────────────

def needs_human_review(
    crop: PanelCrop,
    confidence_threshold: float = 0.7,
    cross_issues: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Determine whether a panel crop needs human review.

    Returns (needs_review, reason).
    """
    if crop.status == "unverified":
        return True, f"Unverified after {crop.verify_attempts} attempts"

    if crop.confidence < confidence_threshold:
        return True, f"Low confidence ({crop.confidence:.2f} < {confidence_threshold})"

    if cross_issues:
        overlap_issues = [i for i in cross_issues if crop.panel_name in i]
        if overlap_issues:
            return True, f"Cross-panel issue: {overlap_issues[0]}"

    return False, ""
