"""Panel bbox matching agent.

Rule-based matching of panel names to DI text-line bounding boxes,
with LLM fallback for ambiguous cases.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .llm_caller import call_llm, parse_json
from ..models.panel import BboxMatch


# ──────────────────────────────────────────────────────────────────────────────
# Text normalization
# ──────────────────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _alphanum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _norm(text))


def _strip_leading_zeros(s: str) -> str:
    return re.sub(r"(?<=[a-z])0+(\d)", r"\1", s)


def _alphanum_similar(a: str, b: str) -> bool:
    pa, pb = _strip_leading_zeros(_alphanum(a)), _strip_leading_zeros(_alphanum(b))
    if min(len(pa), len(pb)) < 3:
        return False
    if pa == pb:
        return True
    # O/0, I/1 tolerance
    def _fix(s: str) -> str:
        return s.replace("o", "0").replace("l", "1").replace("i", "1")
    return _fix(pa) == _fix(pb)


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based matching (6-pass)
# ──────────────────────────────────────────────────────────────────────────────

def rule_match_panel_name(
    panel_name: str,
    di_lines: List[Dict],
    max_merge: int = 3,
) -> List[BboxMatch]:
    """Match a panel name against DI OCR lines using a 6-pass rule engine.

    Returns a list of BboxMatch candidates (may be empty).
    """
    norm_name = _norm(panel_name)
    matches: List[BboxMatch] = []

    # Pass 1: Exact match (case-insensitive)
    for line in di_lines:
        if _norm(line["content"]) == norm_name:
            matches.append(BboxMatch(
                panel_name=panel_name,
                bbox=tuple(int(v) for v in line["bbox"]),
                method="exact",
                confidence=1.0,
                source_lines=[line["content"]],
            ))

    if matches:
        return matches

    # Pass 2: Alphanumeric match
    for line in di_lines:
        if _alphanum_similar(panel_name, line["content"]):
            matches.append(BboxMatch(
                panel_name=panel_name,
                bbox=tuple(int(v) for v in line["bbox"]),
                method="alphanum",
                confidence=0.9,
                source_lines=[line["content"]],
            ))

    if matches:
        return matches

    # Pass 3: Substring containment
    alpha_name = _alphanum(panel_name)
    for line in di_lines:
        alpha_line = _alphanum(line["content"])
        if len(alpha_name) >= 3 and (alpha_name in alpha_line or alpha_line in alpha_name):
            matches.append(BboxMatch(
                panel_name=panel_name,
                bbox=tuple(int(v) for v in line["bbox"]),
                method="substring",
                confidence=0.75,
                source_lines=[line["content"]],
            ))

    if matches:
        return matches

    # Pass 4: Multi-line merge — try merging adjacent lines
    sorted_lines = sorted(di_lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    for i in range(len(sorted_lines)):
        for j in range(i + 1, min(i + max_merge + 1, len(sorted_lines))):
            merged_text = " ".join(sorted_lines[k]["content"] for k in range(i, j + 1))
            if _alphanum_similar(panel_name, merged_text):
                bboxes = [sorted_lines[k]["bbox"] for k in range(i, j + 1)]
                merged_bbox = (
                    min(b[0] for b in bboxes),
                    min(b[1] for b in bboxes),
                    max(b[2] for b in bboxes),
                    max(b[3] for b in bboxes),
                )
                matches.append(BboxMatch(
                    panel_name=panel_name,
                    bbox=merged_bbox,
                    method="multi_line_merge",
                    confidence=0.7,
                    source_lines=[sorted_lines[k]["content"] for k in range(i, j + 1)],
                ))

    if matches:
        return matches

    # Pass 5: OCR-similar (1-char tolerance)
    for line in di_lines:
        if _ocr_similar(panel_name, line["content"]):
            matches.append(BboxMatch(
                panel_name=panel_name,
                bbox=tuple(int(v) for v in line["bbox"]),
                method="ocr_similar",
                confidence=0.6,
                source_lines=[line["content"]],
            ))

    return matches


# ──────────────────────────────────────────────────────────────────────────────
# Conflict resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_conflicts(
    all_matches: Dict[str, List[BboxMatch]],
) -> Dict[str, Optional[BboxMatch]]:
    """Resolve when multiple panel names match the same DI line.

    Keeps the highest-confidence match for each panel name.
    """
    resolved: Dict[str, Optional[BboxMatch]] = {}
    bbox_claims: Dict[tuple, List[Tuple[str, BboxMatch]]] = defaultdict(list)

    # Collect claims
    for name, candidates in all_matches.items():
        if not candidates:
            resolved[name] = None
            continue
        best = max(candidates, key=lambda m: m.confidence)
        bbox_claims[best.bbox].append((name, best))
        resolved[name] = best

    # Resolve shared bboxes — keep highest confidence
    for bbox, claimants in bbox_claims.items():
        if len(claimants) <= 1:
            continue
        sorted_claims = sorted(claimants, key=lambda c: c[1].confidence, reverse=True)
        winner_name = sorted_claims[0][0]
        for name, match in sorted_claims[1:]:
            # Try to find an alternative for the loser
            alternatives = [m for m in all_matches[name] if m.bbox != bbox]
            if alternatives:
                resolved[name] = max(alternatives, key=lambda m: m.confidence)
            else:
                resolved[name] = None

    return resolved


# ──────────────────────────────────────────────────────────────────────────────
# LLM fallback for unmatched names
# ──────────────────────────────────────────────────────────────────────────────

def llm_match_unresolved(
    unresolved_names: List[str],
    di_lines: List[Dict],
    page_img_path: str,
    llm_client,
    deployment: str,
) -> Dict[str, Optional[BboxMatch]]:
    """Use LLM to locate unresolved panel names on the page image."""
    import json

    schema = {
        "type": "object",
        "properties": {
            "matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "panel_name": {"type": "string"},
                        "found": {"type": "boolean"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "required": ["panel_name", "found"],
                },
            }
        },
        "required": ["matches"],
    }

    prompt = (
        f"다음 패널 이름들의 텍스트 위치를 이미지에서 찾아 bbox를 반환하세요.\n"
        f"패널 이름: {json.dumps(unresolved_names, ensure_ascii=False)}\n\n"
        f"반드시 JSON만 반환:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )

    raw = call_llm(
        llm_client, deployment, prompt,
        image_paths=[page_img_path],
        label="llm_match_fallback",
    )
    result = parse_json(raw)
    if not result:
        return {name: None for name in unresolved_names}

    matches: Dict[str, Optional[BboxMatch]] = {}
    for item in result.get("matches", []):
        name = item.get("panel_name", "")
        if name in unresolved_names and item.get("found") and item.get("bbox"):
            matches[name] = BboxMatch(
                panel_name=name,
                bbox=tuple(int(v) for v in item["bbox"]),
                method="llm_fallback",
                confidence=0.5,
                source_lines=[],
            )
        else:
            matches[name] = None

    for name in unresolved_names:
        if name not in matches:
            matches[name] = None

    return matches


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

_OCR_SIMILAR: Dict[str, set] = {
    "v": {"y", "u"}, "y": {"v", "u"}, "u": {"v", "y"},
    "m": {"n", "h"}, "n": {"m", "h"}, "h": {"n", "m"},
    "s": {"5"}, "5": {"s"},
    "o": {"0", "q"}, "0": {"o", "q"},
    "i": {"1", "l"}, "1": {"i", "l"}, "l": {"i", "1"},
}


def _ocr_similar(a: str, b: str) -> bool:
    """Check if two strings differ by exactly 1 OCR-confusable character."""
    pa, pb = _alphanum(a), _alphanum(b)
    if len(pa) < 4 or len(pa) != len(pb) or pa == pb:
        return False
    diffs = [(ca, cb) for ca, cb in zip(pa, pb) if ca != cb]
    if len(diffs) != 1:
        return False
    ca, cb = diffs[0]
    return cb in _OCR_SIMILAR.get(ca, set()) or ca in _OCR_SIMILAR.get(cb, set())
