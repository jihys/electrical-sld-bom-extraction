"""Tests for validators."""

import pytest
from src.models.panel import PanelCrop
from src.validators.panel_validator import (
    needs_human_review,
    validate_crop,
    validate_cross_panel,
    validate_name_coverage,
)


def _make_crop(name="P1", bbox=(100, 100, 500, 500), confidence=0.95, status="verified"):
    return PanelCrop(
        panel_name=name,
        bbox=bbox,
        confidence=confidence,
        verified_by="llm_verify",
        status=status,
    )


class TestValidateCrop:
    def test_valid_crop(self):
        crop = _make_crop()
        valid, issues = validate_crop(crop, 1000, 1000)
        assert valid
        assert not issues

    def test_too_large(self):
        crop = _make_crop(bbox=(0, 0, 950, 950))
        valid, issues = validate_crop(crop, 1000, 1000, max_area_ratio=0.8)
        assert not valid
        assert any("too large" in i for i in issues)

    def test_name_bbox_not_contained(self):
        crop = _make_crop(bbox=(200, 200, 500, 500))
        valid, issues = validate_crop(crop, 1000, 1000, name_bbox=(100, 100, 150, 150))
        assert not valid


class TestNameCoverage:
    def test_all_found(self):
        crops = [_make_crop("A"), _make_crop("B")]
        missing, extra = validate_name_coverage(["A", "B"], crops)
        assert not missing
        assert not extra

    def test_missing(self):
        crops = [_make_crop("A")]
        missing, extra = validate_name_coverage(["A", "B"], crops)
        assert missing == ["B"]


class TestNeedsHumanReview:
    def test_low_confidence(self):
        crop = _make_crop(confidence=0.5)
        needs, reason = needs_human_review(crop, 0.7)
        assert needs
        assert "confidence" in reason.lower()

    def test_unverified(self):
        crop = _make_crop(status="unverified")
        needs, reason = needs_human_review(crop, 0.7)
        assert needs

    def test_high_confidence(self):
        crop = _make_crop(confidence=0.95)
        needs, _ = needs_human_review(crop, 0.7)
        assert not needs
