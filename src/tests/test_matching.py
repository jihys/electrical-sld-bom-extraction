"""Tests for bbox matching rules."""

import pytest
from src.agents.bbox_matcher import rule_match_panel_name


def _line(text, bbox=(0, 0, 50, 20)):
    return {"content": text, "bbox": list(bbox), "polygon": []}


class TestRuleMatch:
    def test_exact_match(self):
        lines = [_line("HV 1"), _line("TR 2")]
        matches = rule_match_panel_name("HV 1", lines)
        assert len(matches) >= 1
        assert matches[0].method == "exact"

    def test_alphanum_match(self):
        lines = [_line("HV-1"), _line("TR 2")]
        matches = rule_match_panel_name("HV 1", lines)
        assert len(matches) >= 1
        assert matches[0].method == "alphanum"

    def test_no_match(self):
        lines = [_line("XYZ 999")]
        matches = rule_match_panel_name("HV 1", lines)
        assert len(matches) == 0

    def test_ocr_similar(self):
        lines = [_line("HY 14")]  # v→y OCR confusion
        matches = rule_match_panel_name("HV 14", lines)
        assert any(m.method == "ocr_similar" for m in matches)
