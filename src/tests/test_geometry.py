"""Tests for geometry tools."""

import pytest
from src.tools.geometry_tools import (
    bbox_area, bbox_contains, bbox_intersect, clip_to_content_bbox,
    compute_iou, dedup_regions,
)


class TestComputeIou:
    def test_identical(self):
        assert compute_iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0

    def test_no_overlap(self):
        assert compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial_overlap(self):
        iou = compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.1 < iou < 0.2  # 25/(100+100-25) ≈ 0.143

    def test_contained(self):
        iou = compute_iou((0, 0, 100, 100), (25, 25, 75, 75))
        assert 0.2 < iou < 0.3  # 2500/10000 = 0.25


class TestBboxContains:
    def test_contains(self):
        assert bbox_contains((0, 0, 100, 100), (10, 10, 90, 90))

    def test_not_contains(self):
        assert not bbox_contains((10, 10, 90, 90), (0, 0, 100, 100))

    def test_exact_match(self):
        assert bbox_contains((0, 0, 100, 100), (0, 0, 100, 100))


class TestBboxIntersect:
    def test_overlap(self):
        result = bbox_intersect((0, 0, 10, 10), (5, 5, 15, 15))
        assert result == (5, 5, 10, 10)

    def test_no_overlap(self):
        assert bbox_intersect((0, 0, 10, 10), (20, 20, 30, 30)) is None


class TestClipToContentBbox:
    def test_clip(self):
        result = clip_to_content_bbox((-5, -5, 200, 200), (0, 0, 100, 100))
        assert result == (0, 0, 100, 100)


class TestBboxArea:
    def test_normal(self):
        assert bbox_area((0, 0, 10, 20)) == 200

    def test_zero(self):
        assert bbox_area((5, 5, 5, 10)) == 0
