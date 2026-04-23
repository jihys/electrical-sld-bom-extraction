"""
panel_bbox_matcher.py
---------------------
Rule-based and LLM-based matching of panel names to DI (Document Intelligence)
text-box bounding boxes.

Public API
----------
rule_match_panel_all(panel_name, lines, max_merge=3, x_overlap_thresh=0.25)
    -> List[dict]   # each dict: {bbox, lines, method}

resolve_bbox_conflicts(page_matches)
    -> Dict[str, List[dict]]

llm_match_panels(panel_names, di_lines, page_img, llm_client, deployment,
                 tiles=None, matched_map=None, tile_batch_size=10)
    -> Dict[str, List[dict]]
"""

from __future__ import annotations

import base64
import json
import re as _re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2


# ── Text normalisation ────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    return _re.sub(r"\s+", " ", text)


def _alphanum_only(text: str) -> str:
    return _re.sub(r"[^a-z0-9]", "", _norm(text))


def _strip_leading_zeros(s: str) -> str:
    """Strip leading zeros from digit groups after letters: ehv01 → ehv1."""
    return _re.sub(r"(?<=[a-z])0+(\d)", r"\1", s)


def _alphanum_lz(text: str) -> str:
    """Alphanumeric only with leading-zero normalisation."""
    return _strip_leading_zeros(_alphanum_only(text))


def _alphanum_similar(a: str, b: str) -> bool:
    """True if both strings are alphanumerically identical (len>=3, O/0 I/1 tolerance).

    Leading zeros in digit groups are ignored so that
    E-HV-01 == EHV-1  and  E-HV-06 == EHV-6.
    """
    pa, pb = _alphanum_lz(a), _alphanum_lz(b)
    if min(len(pa), len(pb)) < 3:
        return False
    if pa == pb:
        return True

    def _fix(s: str) -> str:
        return s.replace("o", "0").replace("l", "1").replace("i", "1")

    return _fix(pa) == _fix(pb)


# Character pairs easily visually confused in OCR (lowercase basis)
_OCR_SIMILAR: dict = {
    "v": {"y", "u"},   "y": {"v", "u"},   "u": {"v", "y"},
    "m": {"n", "h"},   "n": {"m", "h"},   "h": {"n", "m"},
    "c": {"g", "o"},   "g": {"c", "o"},
    "s": {"5"},        "5": {"s"},
    "z": {"2"},        "2": {"z"},
    "b": {"8", "6"},   "8": {"b", "6"},   "6": {"8", "b"},
    "o": {"0", "q"},   "0": {"o", "q"},   "q": {"o", "0"},
    "i": {"1", "l"},   "1": {"i", "l"},   "l": {"i", "1"},
    "d": {"0"},        "t": {"7"},        "7": {"t"},
    "e": {"f"},        "f": {"e"},
}


def _alphanum_ocr_dist1(a: str, b: str) -> bool:
    """Returns True when alphanumeric forms have the same length and differ by exactly 1 OCR-confused character.

    Conditions:
      - Alphanumeric length >= 4 (prevent false matches on very short names)
      - Same length
      - Numeric portion must match exactly (character substitution allowed only in alphabetic prefix/suffix)
      - Exactly 1 differing character that belongs to an _OCR_SIMILAR pair

    Example: HV-14 vs HY-14 → alphanum hv14/hy14 → diff: v↔y ✓
    """
    pa, pb = _alphanum_only(a), _alphanum_only(b)
    if len(pa) < 4 or len(pa) != len(pb) or pa == pb:
        return False
    # Numeric portion must match (both digit count and value)
    pa_digits = _re.sub(r"[^0-9]", "", pa)
    pb_digits = _re.sub(r"[^0-9]", "", pb)
    if pa_digits != pb_digits or not pa_digits:
        return False
    # Verify that exactly 1 position differs and it is an OCR-confused pair
    diffs = [(ca, cb) for ca, cb in zip(pa, pb) if ca != cb]
    if len(diffs) != 1:
        return False
    ca, cb = diffs[0]
    return cb in _OCR_SIMILAR.get(ca, set()) or ca in _OCR_SIMILAR.get(cb, set())


def _alphanum_ocr_dist1_lz(a: str, b: str) -> bool:
    """OCR 1-char dist with leading-zero normalisation.

    Same logic as _alphanum_ocr_dist1 but ignores leading zeros, so pairs
    with different zero-padding like E-HV-04 ↔ ENV-4 (ehv4/env4, h↔n) also match.
    Rejects if numeric portions differ to prevent false-positives.

    min_len lowered to 3 to also match short names (e.g. EHV-1).
    """
    pa, pb = _alphanum_lz(a), _alphanum_lz(b)
    if min(len(pa), len(pb)) < 3 or len(pa) != len(pb) or pa == pb:
        return False
    # Numeric portion must match (prevent false matches between different numbers)
    pa_digits = _re.sub(r"[^0-9]", "", pa)
    pb_digits = _re.sub(r"[^0-9]", "", pb)
    if pa_digits != pb_digits:
        return False
    diffs = [(ca, cb) for ca, cb in zip(pa, pb) if ca != cb]
    if len(diffs) != 1:
        return False
    ca, cb = diffs[0]
    return cb in _OCR_SIMILAR.get(ca, set()) or ca in _OCR_SIMILAR.get(cb, set())


def _is_boundary_match(needle: str, haystack: str) -> bool:
    """Word-boundary aware substring check.

    '0' in 'e-h-02' → False  (digit glued to surrounding chars)
    'e-h-06' in 'e-h-06b(u)' → False
    'e-h-06' in 'e-h-06 bus' → True
    """
    pat = r"(?<![a-z0-9\-])" + _re.escape(needle) + r"(?![a-z0-9\-])"
    return bool(_re.search(pat, haystack))


# ── BBox helpers ─────────────────────────────────────────────────────────────

# Allowed pixel tolerance when comparing DI bboxes (handles 1-2 px deviations from coordinate rendering differences)
_BBOX_ALIGN_TOL: int = 2

def _merged_bbox(lines: list) -> list:
    return [
        min(int(l["bbox"][0]) for l in lines),
        min(int(l["bbox"][1]) for l in lines),
        max(int(l["bbox"][2]) for l in lines),
        max(int(l["bbox"][3]) for l in lines),
    ]


def _x_overlap(a: list, b: list) -> float:
    """Horizontal overlap ratio of two bboxes. Allows _BBOX_ALIGN_TOL px tolerance."""
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]) + _BBOX_ALIGN_TOL)
    return overlap / max(a[2] - a[0], b[2] - b[0], 1)


# ── Serial-pattern position hints ────────────────────────────────────────────

def _parse_serial(name: str):
    """'E-H-06B' -> ('e-h-', 6, 'b')  or  None."""
    m = _re.match(r"^(.*?)(\d+)([a-z]*)$", _norm(name))
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def _cluster_by_cy(points: list, cy_tol: float = 40.0) -> List[list]:
    """Split a (num, cx, cy) list into row clusters based on cy."""
    sorted_pts = sorted(points, key=lambda p: p[2])
    clusters: List[list] = []
    for pt in sorted_pts:
        placed = False
        for cl in clusters:
            if abs(pt[2] - cl[0][2]) <= cy_tol:
                cl.append(pt)
                placed = True
                break
        if not placed:
            clusters.append([pt])
    return clusters


def _linreg_estimate(sib: list, target_num: float) -> Tuple[float, float]:
    """Estimate position of target_num via linear regression on a siblings (num, cx, cy) list."""
    nums = [s[0] for s in sib]
    cxs  = [s[1] for s in sib]
    cys  = [s[2] for s in sib]
    n    = len(nums)
    if n == 1:
        return cxs[0], cys[0]
    xm    = sum(nums) / n
    denom = sum((x - xm) ** 2 for x in nums) or 1
    cx_est = sum(cxs) / n + sum(
        (nums[i] - xm) * (cxs[i] - sum(cxs) / n) for i in range(n)
    ) / denom * (target_num - xm)
    cy_est = sum(cys) / n + sum(
        (nums[i] - xm) * (cys[i] - sum(cys) / n) for i in range(n)
    ) / denom * (target_num - xm)
    return cx_est, cy_est


def serial_position_hints(
    unmatched_names: List[str],
    matched_map: Dict[str, List[dict]],
    cy_row_tol: float = 40.0,
) -> Dict[str, Tuple[float, float]]:
    """Extrapolate expected (cx, cy) for unmatched names from matched serial siblings.

    Siblings are first clustered into Y-rows (cy within cy_row_tol).
    Only the row whose num range best contains the target num is used for
    regression — this prevents panels from unrelated rows (e.g. E-HV-001
    far to the right) from skewing the position estimate.
    """
    groups: Dict[str, list] = defaultdict(list)
    for name, hits in matched_map.items():
        if not hits:
            continue
        p = _parse_serial(name)
        if not p:
            continue
        prefix, num, _ = p
        b = hits[0]["bbox"]
        groups[prefix].append((num, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2))

    hints: Dict[str, Tuple[float, float]] = {}
    for name in unmatched_names:
        p = _parse_serial(name)
        if not p:
            continue
        prefix, num, _ = p
        all_sib = sorted(groups.get(prefix, []))
        if not all_sib:
            continue

        # Cluster siblings by Y-row and pick the best row for this num
        row_clusters = _cluster_by_cy(all_sib, cy_row_tol)

        best_cluster = None
        best_score = float("inf")
        for cl in row_clusters:
            cl_nums = [s[0] for s in cl]
            lo, hi = min(cl_nums), max(cl_nums)
            if lo <= num <= hi:
                # num is within this row's range → use it (prefer smallest range)
                score = hi - lo
            else:
                # num is outside range → penalise by distance to nearest edge
                score = min(abs(num - lo), abs(num - hi)) + (hi - lo)
            if score < best_score:
                best_score = score
                best_cluster = cl

        sib = best_cluster or all_sib
        hints[name] = _linreg_estimate(sib, num)

    return hints


# ── DI filtering ──────────────────────────────────────────────────────────────

def di_in_tile(di_lines: list, tile_bbox: tuple, min_overlap: float = 0.3) -> list:
    """Return DI lines whose bbox overlaps tile_bbox by >= min_overlap of line area.

    Expands the tile boundary by _BBOX_ALIGN_TOL px to include lines near the edge.
    """
    tx1, ty1, tx2, ty2 = tile_bbox
    tx1 -= _BBOX_ALIGN_TOL
    ty1 -= _BBOX_ALIGN_TOL
    tx2 += _BBOX_ALIGN_TOL
    ty2 += _BBOX_ALIGN_TOL
    out = []
    for l in di_lines:
        bx1, by1, bx2, by2 = l["bbox"]
        ox   = max(0, min(bx2, tx2) - max(bx1, tx1))
        oy   = max(0, min(by2, ty2) - max(by1, ty1))
        area = max((bx2 - bx1) * (by2 - by1), 1)
        if ox * oy / area >= min_overlap:
            out.append(l)
    return out


def di_in_any_tile(di_lines: list, tiles: list, min_overlap: float = 0.3) -> list:
    """Return DI lines overlapping with at least one tile (deduped).

    Used to restrict rule-based matching to panel-name regions only,
    preventing false matches from annotation text in the diagram body.
    """
    seen: set = set()
    out: list = []
    for tile in tiles:
        for l in di_in_tile(di_lines, tuple(tile), min_overlap):
            key = tuple(int(v) for v in l["bbox"])
            if key not in seen:
                seen.add(key)
                out.append(l)
    return out


def _img_crop_b64(img, bbox: list, pad: int = 60) -> str:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]
    ax1 = max(0, x1 - pad)
    ay1 = max(0, y1 - pad)
    ax2 = min(w, x2 + pad)
    ay2 = min(h, y2 + pad)
    _, buf = cv2.imencode(".jpg", img[ay1:ay2, ax1:ax2], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def _tile_img_with_matched_boxes(
    page_img,
    tile_bbox: tuple,
    matched_map: Dict[str, list],
) -> tuple:
    """Crop the tile image and overlay already-matched panel boxes in green.

    Returns
    -------
    img_b64        : base64 JPEG (annotated tile image)
    matched_in_tile: [(name, tile_local_bbox), ...]  — matched boxes contained in this tile
    """
    tx1, ty1, tx2, ty2 = [int(v) for v in tile_bbox]
    tile_img = page_img[ty1:ty2, tx1:tx2].copy()

    matched_in_tile: list = []
    for name, hits in matched_map.items():
        for h in hits:
            bx1, by1, bx2, by2 = [int(v) for v in h["bbox"]]
            if bx2 < tx1 or bx1 > tx2 or by2 < ty1 or by1 > ty2:
                continue
            lx1 = max(bx1 - tx1, 0)
            ly1 = max(by1 - ty1, 0)
            lx2 = min(bx2 - tx1, tx2 - tx1)
            ly2 = min(by2 - ty1, ty2 - ty1)
            cv2.rectangle(tile_img, (lx1, ly1), (lx2, ly2), (0, 200, 0), 2)
            cv2.putText(tile_img, name, (lx1, max(ly1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
            matched_in_tile.append((name, [lx1, ly1, lx2, ly2]))

    _, buf = cv2.imencode(".jpg", tile_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode(), matched_in_tile


# ── Rule-based matching ───────────────────────────────────────────────────────

def rule_match_panel_all(
    panel_name: str,
    lines: list,
    max_merge: int = 3,
    x_overlap_thresh: float = 0.25,
) -> List[dict]:
    """Match panel_name against DI lines using boundary-aware rules.

    Pass 1 — boundary exact/substring
    Pass 2 — alphanum normalisation  (E-HV-01 == EHV-1, O↔0, leading-zero)
    Pass 3 — token superset
    Pass 4 — vertical merge of 2-3 adjacent lines
    Pass 5 — OCR character substitution  (HV-14 ↔ HY-14, same length + 1 OCR-similar diff)
    Pass 5.5 — OCR char sub + leading-zero norm  (E-HV-04 ↔ ENV-4, h↔n)
    """
    norm_p = _norm(panel_name)
    tokens_p = set(norm_p.split())
    found: list = []
    seen_bbox: set = set()
    matched_idx: set = set()

    def _add(match: dict, idx=None):
        key = tuple(match["bbox"])
        if key not in seen_bbox:
            seen_bbox.add(key)
            found.append(match)
            if idx is not None:
                matched_idx.add(idx)

    # Pass 1 — exact match or panel name is a boundary substring of DI text
    # NOTE: the reverse direction (DI text as substring of panel name) is intentionally
    # excluded to prevent false positives like DI "1" matching panel "1 SS".
    # Partial-token coverage is handled by Pass 3+4 (token/merge).
    for i, line in enumerate(lines):
        nc = _norm(line["content"])
        if nc == norm_p or _is_boundary_match(norm_p, nc):
            _add({"bbox": _merged_bbox([line]), "lines": [line], "method": "rule-single"}, i)

    # Pass 2
    for i, line in enumerate(lines):
        if i in matched_idx:
            continue
        if _alphanum_similar(panel_name, line["content"]):
            _add({"bbox": _merged_bbox([line]), "lines": [line], "method": "rule-alphanum"}, i)

    # Pass 3+4 — Merge spatially adjacent lines then perform token/text matching
    # (Token matching must only be performed on vertically adjacent lines — prevents combining spatially separated lines)
    srt    = sorted(enumerate(lines), key=lambda x: (x[1]["bbox"][1], x[1]["bbox"][0]))
    sidx   = [p[0] for p in srt]
    slines = [p[1] for p in srt]
    for i in range(len(slines)):
        for ml in range(1, min(max_merge + 1, len(slines) - i + 1)):
            gidx = sidx[i : i + ml]
            if any(g in matched_idx for g in gidx):
                continue
            grp = slines[i : i + ml]
            # ml==1: single line → no spatial condition needed (inherently local)
            # ml>=2: vertically adjacent + horizontal overlap condition
            if ml >= 2:
                ok = all(
                    (grp[k]["bbox"][1] - grp[k - 1]["bbox"][3])
                    <= max(grp[k - 1]["bbox"][3] - grp[k - 1]["bbox"][1], 1) * 2.5
                    and _x_overlap(grp[k - 1]["bbox"], grp[k]["bbox"]) >= x_overlap_thresh
                    for k in range(1, ml)
                )
                if not ok:
                    continue
            combined = " ".join(_norm(l["content"]) for l in grp)
            method = "rule-token" if ml == 1 else f"rule-merge{ml}"
            if ml == 1:
                # Single line: token superset only (exact/boundary handled in Pass 1)
                if tokens_p and tokens_p.issubset(set(combined.split())):
                    _add({"bbox": _merged_bbox(grp), "lines": grp, "method": method},
                         gidx[0] if len(gidx) == 1 else None)
            else:
                if (
                    _is_boundary_match(norm_p, combined)
                    or _is_boundary_match(combined, norm_p)
                    or (tokens_p and tokens_p.issubset(set(combined.split())))
                    or _alphanum_similar(panel_name, " ".join(l["content"] for l in grp))
                ):
                    _add({"bbox": _merged_bbox(grp), "lines": grp, "method": method})

    # Pass 5 — OCR character substitution (same length, matching digits, 1 alphabetic OCR confusion)
    # Example: panel name HV-14 ↔ DI text HY-14  (V→Y substitution)
    for i, line in enumerate(lines):
        if i in matched_idx:
            continue
        if _alphanum_ocr_dist1(panel_name, line["content"]):
            _add({"bbox": _merged_bbox([line]), "lines": [line], "method": "rule-ocr-sub"}, i)

    # Pass 5.5 — OCR 1-char dist + leading-zero normalisation
    # Leading-zero-ignoring match: E-HV-04 ↔ ENV-4 (ehv4/env4, h↔n)
    # Supplements cases not caught by Pass 5 (e.g. length mismatch)
    for i, line in enumerate(lines):
        if i in matched_idx:
            continue
        if _alphanum_ocr_dist1_lz(panel_name, line["content"]):
            _add({"bbox": _merged_bbox([line]), "lines": [line], "method": "rule-ocr-sub-lz"}, i)

    # Pass 6 — Serial number match + similar prefix length
    # Cases where OCR has greatly distorted the prefix but the number (suffix) survived.
    # Example: panel name E-H-15 (prefix alphanum len=2) ↔ DI text GIF15 (prefix len=3)
    #          → number 15 matches, prefix alphanum length difference 1 ≤ 2 → match
    # Safety guard: number must be at least 2 digits (prevents false matches on single-digit numbers)
    p_serial = _parse_serial(panel_name)
    if p_serial and len(str(p_serial[1])) >= 2:
        p_pfx_alen = len(_alphanum_only(p_serial[0]))
        p_num, p_suf = p_serial[1], p_serial[2]
        for i, line in enumerate(lines):
            if i in matched_idx:
                continue
            l_serial = _parse_serial(line["content"])
            if not l_serial:
                continue
            l_pfx_alen = len(_alphanum_only(l_serial[0]))
            l_num, l_suf = l_serial[1], l_serial[2]
            if l_num == p_num and l_suf == p_suf and abs(p_pfx_alen - l_pfx_alen) <= 2:
                _add({"bbox": _merged_bbox([line]), "lines": [line], "method": "rule-serial-num"}, i)

    return found


# ── Bbox conflict resolver ────────────────────────────────────────────────────

METHOD_RANK = {
    "rule-single": 0, "rule-alphanum": 1, "rule-token": 2,
    "rule-merge2": 3, "rule-merge3": 4, "rule-ocr-sub": 5,
    "rule-ocr-sub-lz": 5, "rule-serial-num": 6,
    "llm": 7, "llm-validate": 8,
}


def resolve_bbox_conflicts(
    page_matches: Dict[str, List[dict]],
) -> Dict[str, List[dict]]:
    """For each bbox claimed by multiple panel names, keep the best-scoring one."""
    claims: dict = {}
    for name, hits in page_matches.items():
        for hit in hits:
            key    = tuple(hit["bbox"])
            di_txt = " ".join(l["content"] for l in hit.get("lines", []))
            score  = (
                int(_alphanum_only(name) != _alphanum_only(di_txt)),
                METHOD_RANK.get(hit.get("method", "llm"), 5),
                abs(len(_alphanum_only(name)) - len(_alphanum_only(di_txt))),
                len(name),
                name,
            )
            claims.setdefault(key, []).append((name, hit, score))

    winner = {
        key: sorted(v, key=lambda c: c[2])[0][0]
        for key, v in claims.items()
    }
    return {
        name: [h for h in hits if winner.get(tuple(h["bbox"])) == name]
        for name, hits in page_matches.items()
    }


# ── Cross-reference filter: "FROM …" / "TO: …" context detection ─────────

# Regex for cross-reference annotations surrounding a panel name.
_CROSSREF_RE = _re.compile(
    r"\b(?:from|to|ref|see|via|cable)\b",
    _re.IGNORECASE,
)


def resolve_multi_position_matches(
    page_matches: Dict[str, List[dict]],
    di_lines: list,
    page_img=None,
    llm_client=None,
    deployment: str = "",
    page_num: int = 0,
) -> Dict[str, List[dict]]:
    """When a panel name matches multiple distinct bboxes, pick the best one.

    Heuristics (applied in order):
    1. **Cross-reference penalty**: if the DI line text around the match
       contains "FROM", "TO:", "REF", etc. — it is likely a cable-route
       annotation, not the actual panel header.  Demote such candidates.
    2. **Spatial consistency**: prefer the candidate whose Y-position aligns
       with the majority of other matched panels (panel headers tend to
       share a small set of Y-rows).
    3. **LLM visual tiebreak**: if still ambiguous, send a cropped image
       with all candidate bboxes highlighted and ask the LLM to pick
       the actual panel bay header.

    Only names with ≥ 2 hits are processed; single-hit names pass through.
    """
    out: Dict[str, List[dict]] = {}

    # Collect Y-row centres of single-hit panels (reference pattern)
    ref_cy_list: list = []
    for name, hits in page_matches.items():
        if len(hits) == 1:
            bb = hits[0]["bbox"]
            ref_cy_list.append((bb[1] + bb[3]) / 2)

    for name, hits in page_matches.items():
        if len(hits) <= 1:
            out[name] = hits
            continue

        # ── Score each candidate ──────────────────────────────────────────
        scored: list = []
        for hit in hits:
            penalty = 0
            bb = hit["bbox"]
            cy = (bb[1] + bb[3]) / 2

            # 1. Cross-reference context: check the DI lines near this bbox
            #    for "FROM", "TO:", etc.
            for dl in di_lines:
                dl_bb = dl["bbox"]
                # Check lines vertically close (same row, ±20px) and
                # horizontally near (within 300px)
                if (abs((dl_bb[1] + dl_bb[3]) / 2 - cy) <= 20
                        and abs(dl_bb[0] - bb[0]) <= 300):
                    if _CROSSREF_RE.search(dl["content"]):
                        penalty += 100
                        break

            # Also check the matched DI line text itself
            line_text = " ".join(l["content"] for l in hit.get("lines", []))
            if _CROSSREF_RE.search(line_text):
                penalty += 100

            # 2. Spatial consistency: how close is this candidate's cy to
            #    the most common row among other matched panels?
            if ref_cy_list:
                min_dist = min(abs(cy - rcy) for rcy in ref_cy_list)
                penalty += min_dist  # Closer to existing rows → lower penalty

            scored.append((penalty, hit))

        scored.sort(key=lambda x: x[0])
        best_penalty = scored[0][0]
        second_penalty = scored[1][0] if len(scored) > 1 else best_penalty

        # If the best candidate is clearly better (penalty gap ≥ 30), pick it
        if second_penalty - best_penalty >= 30:
            out[name] = [scored[0][1]]
            print(
                f"  [multi-pos] page {page_num}: '{name}' — "
                f"{len(hits)} candidates, picked bbox={scored[0][1]['bbox']} "
                f"(penalty={best_penalty:.0f} vs {second_penalty:.0f})"
            )
            continue

        # ── 3. LLM tiebreak ──────────────────────────────────────────────
        if llm_client and page_img is not None:
            chosen = _llm_pick_best_position(
                name, [s[1] for s in scored], page_img,
                llm_client, deployment, page_num,
            )
            if chosen is not None:
                out[name] = [chosen]
                print(
                    f"  [multi-pos] page {page_num}: '{name}' — "
                    f"LLM picked bbox={chosen['bbox']}"
                )
                continue

        # Fallback: keep the lowest-penalty candidate
        out[name] = [scored[0][1]]
        print(
            f"  [multi-pos] page {page_num}: '{name}' — "
            f"fallback to lowest penalty bbox={scored[0][1]['bbox']}"
        )

    return out


def _llm_pick_best_position(
    panel_name: str,
    candidates: List[dict],
    page_img,
    llm_client,
    deployment: str,
    page_num: int,
) -> Optional[dict]:
    """Ask LLM to pick the correct bbox among multiple candidates.

    Each candidate bbox is drawn on the page image with a label (A, B, C, …).
    The LLM sees the image and picks the one that is the actual panel bay header.
    """
    import numpy as np

    h, w = page_img.shape[:2]
    # Compute a crop region that includes all candidates + context
    pad = 150
    all_x1 = max(0, min(c["bbox"][0] for c in candidates) - pad)
    all_y1 = max(0, min(c["bbox"][1] for c in candidates) - pad)
    all_x2 = min(w, max(c["bbox"][2] for c in candidates) + pad)
    all_y2 = min(h, max(c["bbox"][3] for c in candidates) + pad)
    crop = page_img[int(all_y1):int(all_y2), int(all_x1):int(all_x2)].copy()

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    colors = [(0, 0, 255), (255, 0, 0), (0, 180, 0), (255, 165, 0), (128, 0, 128)]
    bbox_descs: list = []
    for i, cand in enumerate(candidates):
        bb = cand["bbox"]
        lx1 = int(bb[0] - all_x1)
        ly1 = int(bb[1] - all_y1)
        lx2 = int(bb[2] - all_x1)
        ly2 = int(bb[3] - all_y1)
        color = colors[i % len(colors)]
        label = labels[i]
        cv2.rectangle(crop, (lx1, ly1), (lx2, ly2), color, 3)
        cv2.putText(crop, label, (lx1, max(ly1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        line_text = " ".join(l["content"] for l in cand.get("lines", []))
        bbox_descs.append(f"  {label}: bbox={list(bb)}, OCR text=\"{line_text}\"")

    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode()

    prompt = (
        f"Panel name: \"{panel_name}\"\n\n"
        f"This name appears at {len(candidates)} locations on the SLD page.\n"
        "The image shows all candidate locations marked with colored boxes and labels.\n\n"
        "Candidates:\n" + "\n".join(bbox_descs) + "\n\n"
        "Rules:\n"
        "- Panel bay header names are printed at the TOP of rectangular panel boxes/sections.\n"
        "- Cross-reference annotations like 'FROM ... (E-H-01B)' or 'TO: PANEL-X' "
        "are NOT panel headers — they are cable routing labels.\n"
        "- The correct match is the one that labels an actual panel bay section.\n\n"
        "Which label (A, B, C, …) is the actual panel bay header?\n"
        "Respond with JSON: {\"choice\": \"A\"}"
    )

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}},
    ]
    try:
        import time as _time
        _t0 = _time.time()
        resp = llm_client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        _elapsed = round(_time.time() - _t0, 3)
        from .panel_utils import log_llm_call
        log_llm_call("multi_pos_tiebreak", _elapsed, 1,
                     reasoning_effort="none", source="panel_bbox_matcher")
        body = json.loads(resp.choices[0].message.content)
        choice = body.get("choice", "A").upper().strip()
        idx = labels.index(choice) if choice in labels else -1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    except Exception as exc:
        print(f"  [multi-pos] LLM tiebreak error: {exc}")

    return None


# ── LLM batch call ────────────────────────────────────────────────────────────

def _llm_call_batch(
    names: List[str],
    di_lines_sub: list,
    img_b64: Optional[str],
    llm_client,
    deployment: str,
    hints: dict,
    matched_context: Optional[list] = None,  # [(name, tile_local_bbox), ...]
) -> dict:
    hints_text = ""
    if hints:
        h_lines = [
            f'  "{n}": estimated position ~ ({cx:.0f}, {cy:.0f})'
            for n, (cx, cy) in hints.items()
            if n in names
        ]
        if h_lines:
            hints_text = (
                "\n\nPosition hints (estimated from serial pattern of already-matched panels):\n"
                + "\n".join(h_lines)
                + "\nUse these to narrow down candidate OCR lines."
            )

    matched_text = ""
    if matched_context:
        m_lines = [
            f'  "{name}"  bbox={bbox}'
            for name, bbox in matched_context
        ]
        matched_text = (
            "\n\nAlready matched panels in this tile (shown in GREEN on the image):\n"
            + "\n".join(m_lines)
            + "\nUse these as spatial and visual reference to find the missing panel names below."
        )

    lines_text = "\n".join(
        f'{i + 1}. "{l["content"]}"  bbox={l["bbox"]}'
        for i, l in enumerate(di_lines_sub)
    )
    prompt = (
        "You are given a tile image from an electrical Single Line Diagram (SLD).\n"
        "Panel bay header names are printed in rectangular boxes.\n"
        "Already-found panel boxes are highlighted in GREEN on the image.\n\n"
        "Your task: find the OCR text box (from the list below) that corresponds to each MISSING panel name.\n"
        "Use the spatial pattern of green boxes, the name pattern (serial numbering), "
        "and the OCR bbox positions to infer where the missing panels are.\n\n"
        "Missing panel names to find:\n"
        + "\n".join(f'- "{n}"' for n in names)
        + "\n\nOCR lines (index. text  bbox=[x1,y1,x2,y2] in tile-local pixels):\n"
        + lines_text
        + matched_text
        + hints_text
        + "\n\nRules:\n"
        "- OCR may have O<->0, I<->1, shifted hyphens — match visually.\n"
        "- Same name can appear MULTIPLE times.\n"
        "- 'E-H-01' and 'E-H-01A' are DIFFERENT names.\n"
        "- Only match names that are visually present in THIS tile.\n"
        "- occurrences:[] if the panel name is not in this tile.\n\n"
        "Respond with valid JSON in this format:\n"
        '{"matches":{"<name>":{"occurrences":[{"line_indices":[<1-based>],'
        '"confidence":"high|medium|low"}]}}}'
    )
    content: list = [{"type": "text", "text": prompt}]
    if img_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"},
        })
    import time as _time
    _t0 = _time.time()
    resp = llm_client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    _elapsed = round(_time.time() - _t0, 3)
    from .panel_utils import log_llm_call
    log_llm_call("bbox_matcher_llm", _elapsed, int(img_b64 is not None),
                 reasoning_effort="none", source="panel_bbox_matcher")
    return json.loads(resp.choices[0].message.content)


# ── LLM fallback matching ─────────────────────────────────────────────────────

def llm_match_panels(
    panel_names: List[str],
    di_lines: list,
    page_img,
    llm_client,
    deployment: str,
    tiles: Optional[list] = None,
    matched_map: Optional[Dict[str, list]] = None,
    tile_batch_size: int = 10,
) -> Dict[str, List[dict]]:
    """LLM fallback matching with serial hints and optional tile splitting.

    If len(panel_names) >= tile_batch_size and tiles is given, each tile's DI
    lines and cropped image are sent to a separate LLM call to improve accuracy
    and reduce cost.
    """
    if not panel_names:
        return {}

    matched_map = matched_map or {}
    hints = serial_position_hints(panel_names, matched_map)
    if hints:
        print(f"    [serial hints] {len(hints)}/{len(panel_names)} names have position estimates")

    # ── tile-split path ───────────────────────────────────────────────────────
    # When many names are unmatched (>= tile_batch_size): run LLM per tile.
    # Provide as context: image with already-matched boxes overlaid on each tile +
    # DI lines + already-found box positions, so the LLM can infer unmatched panels
    # using spatial/visual/name patterns.
    if tiles and len(panel_names) >= tile_batch_size:
        print(
            f"    [LLM] {len(panel_names)} names >= {tile_batch_size}"
            f" → tile-level inference ({len(tiles)} tiles)"
        )
        out: Dict[str, list] = {n: [] for n in panel_names}
        remaining = list(panel_names)  # Track names not yet found

        for ti, tile in enumerate(tiles):
            if not remaining:
                break
            tile_di = di_in_tile(di_lines, tile)
            if not tile_di:
                print(f"    [LLM] tile {ti}: no DI lines — skip")
                continue

            # Overlay already-found panel boxes on the tile image (green)
            img_b64 = None
            matched_in_tile: list = []
            if page_img is not None:
                img_b64, matched_in_tile = _tile_img_with_matched_boxes(
                    page_img, tile, matched_map
                )

            tile_hints = {n: hints[n] for n in remaining if n in hints}
            print(
                f"    [LLM] tile {ti}: {len(remaining)} names to try, "
                f"{len(tile_di)} DI lines, {len(matched_in_tile)} known boxes"
            )
            parsed = _llm_call_batch(
                remaining, tile_di, img_b64, llm_client, deployment,
                tile_hints, matched_context=matched_in_tile if matched_in_tile else None,
            )

            newly_found = []
            for name in remaining:
                occs = parsed.get("matches", {}).get(name, {}).get("occurrences") or []
                seen_b: set = set()
                for occ in occs:
                    idxs = [i - 1 for i in (occ.get("line_indices") or []) if 1 <= i <= len(tile_di)]
                    if idxs:
                        grp = [tile_di[i] for i in idxs]
                        key = tuple(_merged_bbox(grp))
                        if key not in seen_b:
                            seen_b.add(key)
                            out[name].append({
                                "bbox": _merged_bbox(grp),
                                "lines": grp,
                                "method": "llm",
                                "confidence": occ.get("confidence", "?"),
                            })
                if out[name]:
                    newly_found.append(name)

            if newly_found:
                print(f"      found: {newly_found}")
                # Add newly found boxes to matched_map (used as context in the next tile)
                for name in newly_found:
                    matched_map[name] = out[name]
                remaining = [n for n in remaining if n not in newly_found]

        return out

    # ── single-batch path ─────────────────────────────────────────────────────
    img_b64 = None
    if page_img is not None and di_lines:
        pad = 80
        ax1 = max(0, min(int(l["bbox"][0]) for l in di_lines) - pad)
        ay1 = max(0, min(int(l["bbox"][1]) for l in di_lines) - pad)
        ax2 = min(page_img.shape[1], max(int(l["bbox"][2]) for l in di_lines) + pad)
        ay2 = min(page_img.shape[0], max(int(l["bbox"][3]) for l in di_lines) + pad)
        _, buf = cv2.imencode(".jpg", page_img[ay1:ay2, ax1:ax2], [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf).decode()

    parsed = _llm_call_batch(panel_names, di_lines, img_b64, llm_client, deployment, hints)
    out = {}
    for name in panel_names:
        occs = parsed.get("matches", {}).get(name, {}).get("occurrences") or []
        hits: list = []
        seen_b: set = set()
        for occ in occs:
            idxs = [i - 1 for i in (occ.get("line_indices") or []) if 1 <= i <= len(di_lines)]
            if idxs:
                grp = [di_lines[i] for i in idxs]
                key = tuple(_merged_bbox(grp))
                if key not in seen_b:
                    seen_b.add(key)
                    hits.append({
                        "bbox": _merged_bbox(grp),
                        "lines": grp,
                        "method": "llm",
                        "confidence": occ.get("confidence", "?"),
                    })
        out[name] = hits
    return out


# ── Serial outlier correction ─────────────────────────────────────────────────

def correct_serial_outliers(
    panel_bbox_matches: Dict[int, Dict[str, List[dict]]],
    result: dict,
) -> Tuple[Dict, dict, List[str]]:
    """Detect and correct OCR digit errors in serial panel names.

    Strategy: names sharing the same Y-row (bbox center-Y within 15 px) and
    the same serial prefix are sorted by center-X.  Their serial numbers
    should be monotonically increasing.  When a number is out of order, common
    OCR substitutions are tried (1→0 being most frequent) and accepted when
    the corrected number falls strictly between its spatial neighbours.

    Example row (sorted by X):
        E-H-02, E-H-03, E-H-04, [E-H-15], E-H-07
        E-H-15 is between E-H-04 and E-H-07 → try 15→05 → 4 < 5 < 7 ✓
        Correction applied: E-H-15 → E-H-05
    """
    # (bad_digit, good_digit) pairs in priority order
    _DIGIT_SUBS = [("1", "0"), ("6", "0"), ("8", "0"), ("5", "6"), ("0", "6"), ("0", "8")]

    def _last_digits_sub(name: str, cand: int) -> str:
        """Replace the last digit group in *name* with zero-padded *cand*."""
        m = _re.search(r"(\d+)(?=[^0-9]*$)", name)
        if not m:
            return name
        return name[: m.start()] + str(cand).zfill(len(m.group())) + name[m.end() :]

    corrections: List[str] = []

    for pn in sorted(panel_bbox_matches.keys()):
        page_matches = panel_bbox_matches[pn]

        # Build: prefix -> [(num, suffix, name, cx, cy), ...]
        groups: Dict[str, list] = defaultdict(list)
        for name, hits in page_matches.items():
            if not hits:
                continue
            p = _parse_serial(name)
            if not p:
                continue
            prefix, num, suffix = p
            bb = hits[0]["bbox"]
            cx = (bb[0] + bb[2]) / 2
            cy = (bb[1] + bb[3]) / 2
            groups[prefix].append([num, suffix, name, cx, cy])

        for prefix, members in groups.items():
            if len(members) < 2:
                continue

            # Split into Y-rows (center-Y within 15 px of row anchor)
            members.sort(key=lambda m: (m[4], m[3]))
            rows: List[list] = []
            for m in members:
                placed = False
                for row in rows:
                    if abs(m[4] - row[0][4]) <= 15:
                        row.append(m)
                        placed = True
                        break
                if not placed:
                    rows.append([m])

            for row in rows:
                if len(row) < 2:
                    continue
                row.sort(key=lambda m: m[3])  # sort by cx

                # Detect row direction: +1 = increasing left→right, -1 = decreasing
                nums_init = [m[0] for m in row]
                inc = sum(1 for j in range(len(nums_init) - 1) if nums_init[j] < nums_init[j + 1])
                dec = sum(1 for j in range(len(nums_init) - 1) if nums_init[j] > nums_init[j + 1])
                direction = +1 if inc >= dec else -1

                changed = True
                while changed:
                    changed = False
                    nums = [m[0] for m in row]

                    for i in range(len(nums) - 1):
                        # Skip pairs that are correctly ordered for this direction
                        if direction == +1 and nums[i] <= nums[i + 1]:
                            continue
                        if direction == -1 and nums[i] >= nums[i + 1]:
                            continue

                        orig_num, suffix, orig_name, cx, cy = row[i]
                        next_num = nums[i + 1]
                        next_name = row[i + 1][2]
                        all_known_alpha = {
                            _alphanum_only(n) for n in page_matches
                            if n not in (orig_name, next_name)
                        }

                        # ── 1. Swap detection ─────────────────────────────────
                        # Adjacent pair in wrong order → swap their bbox assignments
                        if direction == +1:
                            expected_prev = nums[i - 1] if i > 0 else orig_num - 1
                            expected_next = nums[i + 2] if i + 2 < len(nums) else next_num + 1
                            pair_ok_swapped = expected_prev < next_num and orig_num < expected_next
                        else:
                            expected_prev = nums[i - 1] if i > 0 else orig_num + 1
                            expected_next = nums[i + 2] if i + 2 < len(nums) else next_num - 1
                            pair_ok_swapped = expected_prev > next_num and orig_num > expected_next

                        if pair_ok_swapped:
                            # Swap bbox assignments between the two names
                            page_matches[orig_name], page_matches[next_name] = (
                                page_matches[next_name], page_matches[orig_name]
                            )
                            row[i][3], row[i + 1][3] = row[i + 1][3], row[i][3]  # swap cx
                            row[i][4], row[i + 1][4] = row[i + 1][4], row[i][4]  # swap cy
                            msg = (
                                f"  page {pn}: swapped bbox  '{orig_name}' ↔ '{next_name}'"
                                f"  (direction={'↑' if direction==1 else '↓'},"
                                f" spatial neighbours: {nums[i-1] if i>0 else '?'}…{nums[i+2] if i+2<len(nums) else '?'})"
                            )
                            corrections.append(msg)
                            print(msg)
                            changed = True
                            break

                        # ── 2. Single-element rename (digit sub + ±1) ─────────
                        if direction == +1:
                            prev_num = nums[i - 1] if i > 0 else 0
                            lo, hi = prev_num, next_num   # need lo < cand < hi
                        else:
                            prev_num = nums[i - 1] if i > 0 else orig_num + 1
                            lo, hi = next_num, prev_num   # need lo < cand < hi

                        s = str(orig_num)
                        # Build correction candidates: digit substitutions + ±1
                        candidates: List[int] = []
                        for pos in range(len(s)):
                            for bad, good in _DIGIT_SUBS:
                                if s[pos] == bad:
                                    try:
                                        candidates.append(int(s[:pos] + good + s[pos + 1:]))
                                    except ValueError:
                                        pass
                        candidates += [orig_num + 1, orig_num - 1]

                        corrected = False
                        for cand_num in candidates:
                            if not (lo < cand_num < hi):
                                continue
                            new_name = _last_digits_sub(orig_name, cand_num)
                            if new_name == orig_name:
                                continue
                            if _alphanum_only(new_name) in all_known_alpha:
                                continue
                            hits = page_matches.pop(orig_name)
                            page_matches[new_name] = hits
                            names_list = result["by_page"].get(str(pn), [])
                            if orig_name in names_list:
                                names_list[names_list.index(orig_name)] = new_name
                            src = (
                                f"±1" if cand_num in (orig_num + 1, orig_num - 1)
                                else f"digit sub"
                            )
                            msg = (
                                f"  page {pn}: '{orig_name}' → '{new_name}'"
                                f"  ({src}, direction={'↑' if direction==1 else '↓'},"
                                f" neighbours: {lo}…{hi})"
                            )
                            corrections.append(msg)
                            print(msg)
                            row[i][0] = cand_num
                            row[i][2] = new_name
                            corrected = True
                            changed = True
                            break
                        if changed:
                            break  # restart while loop

    return panel_bbox_matches, result, corrections


# ── Serial gap-fill by spatial position ──────────────────────────────────────

def fill_serial_gaps_by_position(
    panel_bbox_matches: Dict[int, Dict[str, List[dict]]],
    result: dict,
    di_lines_by_page: Dict[int, list],
    x_tol: float = 30.0,
    y_tol: float = 15.0,
) -> Tuple[Dict, dict, List[str]]:
    """Assign unmatched serial names to DI lines at their expected spatial position.

    After rule-based and LLM matching, some panels remain unmatched because
    OCR garbled their name so badly that string similarity fails
    (e.g. "E-H-06" read as "EIN", "E-H-08" read as "EI-O").

    Algorithm:
      1. For each page, build serial groups from *matched* panels.
      2. For each unmatched panel in the same series, estimate the expected
         (cx, cy) by linear interpolation/extrapolation from matched siblings.
      3. Search raw DI lines (di_lines_by_page) for a line whose centre is
         within (x_tol, y_tol) of the expected position AND that is not yet
         claimed by any matched panel.
      4. Assign that DI line's bbox as a new match (method="serial_gap_fill").

    Parameters
    ----------
    x_tol : max horizontal distance in pixels between expected and actual DI cx
    y_tol : max vertical distance in pixels between serial-row cy and DI line cy
    """
    fills: List[str] = []

    for pn in sorted(panel_bbox_matches.keys()):
        page_matches = panel_bbox_matches[pn]
        di_lines_all = di_lines_by_page.get(pn, [])
        if not di_lines_all:
            continue

        # Collect already-claimed bboxes (as int-tuples) to avoid re-assigning
        claimed: set = set()
        for hits in page_matches.values():
            for h in hits:
                claimed.add(tuple(int(v) for v in h["bbox"]))

        # Build serial groups from matched entries only
        groups: Dict[str, list] = defaultdict(list)
        for name, hits in page_matches.items():
            p = _parse_serial(name)
            if not p:
                continue
            prefix, num, suffix = p
            if hits:
                bb = hits[0]["bbox"]
                cx = (bb[0] + bb[2]) / 2
                cy = (bb[1] + bb[3]) / 2
                groups[prefix].append({"num": num, "suffix": suffix,
                                        "name": name, "cx": cx, "cy": cy, "matched": True})
            else:
                groups[prefix].append({"num": num, "suffix": suffix,
                                        "name": name, "cx": None, "cy": None, "matched": False})

        for prefix, members in groups.items():
            unmatched_members = [m for m in members if not m["matched"]]
            matched_members   = [m for m in members if m["matched"]]
            if not unmatched_members or not matched_members:
                continue

            # Sort matched by serial number, build cx→num linear model
            matched_members.sort(key=lambda m: m["num"])
            ref_cy = sum(m["cy"] for m in matched_members) / len(matched_members)

            # Simple linear fit: cx = a * num + b  (using at least 2 points)
            if len(matched_members) >= 2:
                xs = [m["num"] for m in matched_members]
                ys = [m["cx"] for m in matched_members]
                n = len(xs)
                sx, sy = sum(xs), sum(ys)
                sxy = sum(x * y for x, y in zip(xs, ys))
                sxx = sum(x * x for x in xs)
                denom = n * sxx - sx * sx
                if denom != 0:
                    a = (n * sxy - sx * sy) / denom
                    b = (sy - a * sx) / n
                else:
                    a = 0
                    b = sy / n
            else:
                # Only one reference: can't fit a line, skip this group
                continue

            for um in unmatched_members:
                expected_cx = a * um["num"] + b
                # Find closest unclaimed DI line near expected position on the same row
                best = None
                best_dx = float("inf")
                for dl in di_lines_all:
                    bb = dl["bbox"]
                    dcx = (bb[0] + bb[2]) / 2
                    dcy = (bb[1] + bb[3]) / 2
                    dx = abs(dcx - expected_cx)
                    dy = abs(dcy - ref_cy)
                    if dx <= x_tol and dy <= y_tol and dx < best_dx:
                        key = tuple(int(v) for v in bb)
                        if key not in claimed:
                            best = dl
                            best_dx = dx
                if best is None:
                    continue
                # Assign
                match_entry = {
                    "bbox": [int(v) for v in best["bbox"]],
                    "lines": [best],
                    "method": "serial_gap_fill",
                    "content_found": best.get("content", ""),
                }
                page_matches[um["name"]] = [match_entry]
                claimed.add(tuple(match_entry["bbox"]))
                names_list = result["by_page"].get(str(pn), [])
                msg = (
                    f"  page {pn}: '{um['name']}' gap-filled"
                    f"  expected_cx={expected_cx:.0f}"
                    f"  found='{best.get('content','')}' @ {match_entry['bbox']}"
                )
                fills.append(msg)
                print(msg)

    return panel_bbox_matches, result, fills

# ── Noise DI line filter ──────────────────────────────────────────────────────

# Filter pattern list (DI line is excluded if any one pattern matches)
_NOISE_PATTERNS: list[str] = [
    r"VCB",                        # 1. Contains VCB
    r"REACTOR",                    # Contains REACTOR
    r"CT[x×]\d+",                  # 2. CTx1, CTx3, etc.
    r"\d+\s*/\s*\d+\s*A\b",       # 3/6. number/numberA  (e.g. 630/5A)
    r"\d+(?:\.\d+)?\s*V\s*/\s*\d+(?:\.\d+)?\s*V\b",  # numberV/numberV (e.g. 220V/110V)
    r"\d+(?:\.\d+)?\s*VA\b",      # 4. numberVA
    r"\d+(?:\.\d+)?\s*kA\b",      # 5. numberkA
    r"ZCT",                        # 7. Contains ZCT
    r"MCCB",                       # 8. MCCB
    r"\d+(?:\.\d+)?\s*AF\b",      # 9. numberAF
    r"\d+(?:\.\d+)?\s*%",         # 10/11. number% or number %
    # Existing electrical unit patterns
    r"\d+(?:\.\d+)?\s*kW\b",
    r"\d+(?:\.\d+)?\s*MW\b",
    r"\d+(?:\.\d+)?\s*MVA\b",
    r"\d+(?:\.\d+)?\s*kV\b",
    r"\d+(?:\.\d+)?\s*kVA\b",
    r"\d+(?:\.\d+)?\s*Hz\b",
    # Additional circuit component/terminology patterns
    r"QPTR",
    r"CAPACITOR",
    r"\d+(?:\.\d+)?\s*kVAR\b",
    r"\bFROM\b",
    r"\bTO\b",
    r"VOLTAGE",
    r"DETECTOR",
    r"LED",
    r"SA[x×]\d+",                  # SAxnumber (e.g. SAx3)
    r"ALARM",
    r"\bSEE\b",
    r"\bD\.C\b",                   # Contains D.C
    # Additional electrical/cable specification patterns
    r"\d+(?:\.\d+)?\s*[-]\s*kW\b", # number-kW  (e.g. 100-kW)
    r"\d+[x×]CT\b",                # numberxCT  (e.g. 3xCT)
    r"\d+(?:\.\d+)?\s*S\.R\b",    # numberS.R  (e.g. 5.5S.R)
    r"S\.R\b",                     # S.R standalone
    r"\d+\.\d+\s*k\b",            # number.numberk  (e.g. 22.9k, 1.5k)
    r"\d+\s*/\s*\d+(?!\s*[AaVv])", # number/number (plain fraction without units; A/V-suffixed handled by existing patterns)
    r"\d+C[x×]\d+(?:-\d+)?",      # numberCxnumber or numberCxnumber-number  (cable spec, e.g. 3Cx16-2)
    r"(?<![A-Za-z0-9_-])\d+\s*A\b",  # numberA  (current value, e.g. 600A, 100A) — excluded if preceded by letter/- (protects panel name suffix)
    r"VT[x×]\d+",                  # VTxnumber  (e.g. VTx1, VTx2)
    r"%Z\s*=",                      # %Z= impedance value (e.g. %Z=5.23)
]

_NOISE_TERMS = _re.compile("|".join(_NOISE_PATTERNS), _re.IGNORECASE)


def merge_split_di_lines(
    lines: list,
    y_gap_factor: float = 1.5,
    x_overlap_thresh: float = 0.3,
) -> list:
    """Merge vertically adjacent DI lines that together form one label.

    Two lines are merged when:
    - The vertical gap between them is ≤ y_gap_factor × (average line height)
    - Their horizontal overlap ratio is ≥ x_overlap_thresh

    Merged lines get:
    - content  : upper + " " + lower (stripped)
    - bbox     : tight union of both bboxes
    - crop_idx : taken from upper line (they should be the same tile)
    - _merged  : True  (flag so callers can tell)
    """
    if not lines:
        return lines

    # Sort top-to-bottom, then left-to-right
    srt = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    used = [False] * len(srt)
    out: list = []

    for i, lo in enumerate(srt):
        if used[i]:
            continue
        x1_a, y1_a, x2_a, y2_a = lo["bbox"]
        h_a = max(y2_a - y1_a, 1)
        merged = False
        for j in range(i + 1, len(srt)):
            if used[j]:
                continue
            x1_b, y1_b, x2_b, y2_b = srt[j]["bbox"]
            # Only look at lines that start below the current one
            if y1_b < y2_a - _BBOX_ALIGN_TOL:
                continue
            # Stop scanning if the next line is too far down
            h_b = max(y2_b - y1_b, 1)
            avg_h = (h_a + h_b) / 2
            if y1_b - y2_a > y_gap_factor * avg_h:
                break
            # Check horizontal overlap
            overlap = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
            span   = max(x2_a - x1_a, x2_b - x1_b, 1)
            if overlap / span < x_overlap_thresh:
                continue
            # Merge
            merged_line = {
                "content":  lo["content"].strip() + " " + srt[j]["content"].strip(),
                "bbox":     [min(x1_a, x1_b), min(y1_a, y1_b),
                             max(x2_a, x2_b), max(y2_a, y2_b)],
                "crop_idx": lo.get("crop_idx"),
                "_merged":  True,
            }
            if "tile_idx" in lo:
                merged_line["tile_idx"] = lo["tile_idx"]
            used[j] = True
            out.append(merged_line)
            merged = True
            break
        if not merged:
            out.append(lo)

    return out


def _filter_noise_di_lines(lines: list) -> list:
    """Remove DI lines containing circuit component/value labels (VCB, CTx1, numberA, MCCB, etc.).

    Pre-excludes lines with electrical parameter values that are not panel names, before matching.
    """
    return [l for l in lines if not _NOISE_TERMS.search(l.get("content", ""))]


def _filter_frequent_di_lines(
    lines: list,
    min_count: int = 5,
    panel_name_tokens: "set | None" = None,
) -> list:
    """Remove lines where the same content appears min_count or more times on a single page.

    Texts that repeatedly appear like legend/title/unit labels are likely not panel names,
    so they are pre-excluded based on a histogram threshold.

    If panel_name_tokens is provided, lines whose normalized content is in that set
    are preserved regardless of frequency. This prevents high-frequency deletion of
    the top prefix token of 2-line panel names (e.g. 'GCP' + '1A').
    Example: GCP 1A~5A → even if 'GCP' appears 5 times, keep it if 'gcp' is in panel_name_tokens.
    """
    from collections import Counter
    freq = Counter(_norm(l.get("content", "")) for l in lines)
    protected = panel_name_tokens or set()
    return [
        l for l in lines
        if freq[_norm(l.get("content", ""))] < min_count
        or _norm(l.get("content", "")) in protected
    ]


# ── High-level orchestrator ───────────────────────────────────────────────────

def _group_di_lines_by_tile(
    di_lines_by_page: Dict[int, list],
    tiles_by_page: dict,
) -> Dict[tuple, list]:
    """Group di_lines_by_page by (page_num, tile_idx) key.

    If a line has a 'tile_idx' field, that value is used;
    otherwise tile_idx is estimated via tiles_by_page bbox overlap (for legacy DI compatibility).
    """
    groups: Dict[tuple, list] = {}
    for pn, lines in di_lines_by_page.items():
        tiles = tiles_by_page.get(pn) or []
        for line in lines:
            if "tile_idx" in line:
                key = (pn, line["tile_idx"])
                groups.setdefault(key, []).append(line)
            elif tiles:
                bx1, by1, bx2, by2 = line["bbox"]
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                for ti, (tx1, ty1, tx2, ty2) in enumerate(tiles):
                    if (tx1 - _BBOX_ALIGN_TOL <= cx <= tx2 + _BBOX_ALIGN_TOL
                            and ty1 - _BBOX_ALIGN_TOL <= cy <= ty2 + _BBOX_ALIGN_TOL):
                        groups.setdefault((pn, ti), []).append(line)
                        break
                else:
                    groups.setdefault((pn, -1), []).append(line)
            else:
                groups.setdefault((pn, 0), []).append(line)
    return groups


def run_bbox_matching(
    result: dict,
    di_lines_by_page: Dict[int, list],
    page_images: dict,
    tiles_by_page: dict,
    llm_client,
    deployment: str,
    output_root=None,
    notebook_dir=None,
    stages_out: Optional[dict] = None,
) -> tuple:
    """Run full 7-B pipeline: rule-based (per-tile) → merge → conflict resolve → LLM fallback.

    Rule-based matching runs independently per tile, then all tile results are
    merged per page.  LLM fallback also runs per tile for unmatched names.

    Parameters
    ----------
    result          : panel_names dict with ``result["by_page"]``
    di_lines_by_page: DI lines per page (full-page coords, tile_idx per line)
    page_images     : BGR ndarray per page
    tiles_by_page   : tile bboxes per page  [(tx1,ty1,tx2,ty2), ...]
    llm_client      : AzureOpenAI client
    deployment      : deployment name
    output_root     : Path used for loading panel_names.json (optional)
    notebook_dir    : Path used for relative display in log messages (optional)
    stages_out      : optional dict to capture intermediate snapshots.
                      If provided, will be populated with keys:
                        "1_rule_only"   — after Pass 1-6 + serial-align + conflict resolve
                        "2_llm_fallback"— after LLM fallback + serial correction + gap-fill

    Returns
    -------
    (panel_bbox_matches, result)
        panel_bbox_matches : Dict[int, Dict[str, List[dict]]]
        result             : same dict as input (unchanged here, returned for convenience)
    """
    from collections import Counter as _Counter

    if output_root is not None:
        import json as _json
        _result_path = output_root / "panel_names.json"
        with open(_result_path, encoding="utf-8") as _f:
            result = _json.load(_f)
        _rel = _result_path.resolve().relative_to(notebook_dir) if notebook_dir else _result_path
        print(f"Loaded result from {_rel}")
        print(f"Pages: {sorted(result['by_page'].keys())}")

    # ── Merge 2-line split DI entries into single bounding boxes ─────────────
    merged_di: Dict[int, list] = {}
    for pn, lines in di_lines_by_page.items():
        merged = merge_split_di_lines(lines)
        n_merged = sum(1 for l in merged if l.get("_merged"))
        if n_merged:
            print(f"  page {pn}: merged {n_merged} split DI line pair(s)")
        merged_di[pn] = merged
    di_lines_by_page = merged_di

    # ── Dedup: remove bare name when (UPPER)/(LOWER) suffixed variant exists ──
    for _ps in list(result["by_page"].keys()):
        _names = result["by_page"][_ps]
        _suffixed_bases: set = set()
        for _n in _names:
            _m = _re.match(r'^(.+?)\s*\((?:UPPER|LOWER)\)$', _n, _re.IGNORECASE)
            if _m:
                _suffixed_bases.add(_m.group(1).strip())
        if _suffixed_bases:
            _before = len(_names)
            _names = [_n for _n in _names if _n not in _suffixed_bases]
            _removed = _before - len(_names)
            if _removed:
                print(f"  page {_ps}: removed {_removed} bare name(s) superseded by (UPPER)/(LOWER) variants")
            result["by_page"][_ps] = _names

    # ── DI-guided supplement: discover panel names the LLM missed ─────────────
    # If a DI line contains an already-extracted name preceded by a valid
    # alphanumeric prefix (e.g. "NGR-E-TR-01B" when "E-TR-01B" is known),
    # add the longer form as an additional panel name.
    for _ps in list(result["by_page"].keys()):
        _pn = int(_ps)
        _names = result["by_page"][_ps]
        _names_set = set(_names)
        _di_lines = di_lines_by_page.get(_pn, [])
        _new_names: list = []
        for _name in _names:
            _esc = _re.escape(_name)
            _pat = _re.compile(
                r'([A-Z0-9][-A-Z0-9]*[-_ ])' + _esc + r'(?:\s|$)',
                _re.IGNORECASE,
            )
            for _dl in _di_lines:
                _dt = _dl["content"].strip()
                _m = _pat.search(_dt)
                if _m:
                    _full = _m.group(0).rstrip()
                    _prefix = _m.group(1).rstrip('-_ ')
                    if (_full not in _names_set
                            and _re.match(r'^[A-Z0-9][-A-Z0-9]*$', _prefix, _re.IGNORECASE)):
                        _new_names.append(_full)
                        _names_set.add(_full)
        if _new_names:
            result["by_page"][_ps].extend(_new_names)
            print(f"  page {_ps}: added {len(_new_names)} DI-discovered name(s): {_new_names}")

    # ── Group DI lines by (page_num, tile_idx) ────────────────────────────────
    tile_di_groups = _group_di_lines_by_tile(di_lines_by_page, tiles_by_page)

    panel_bbox_matches: Dict[int, Dict[str, List[dict]]] = {}
    unmatched_by_page: Dict[int, list] = {}

    # ── Rule-based matching: per tile → merge ─────────────────────────────────
    print("\nRule-based matching (per tile → merge):")
    for page_str, panel_names in result["by_page"].items():
        pn = int(page_str)
        panel_bbox_matches[pn] = {name: [] for name in panel_names}

        tile_indices = sorted({k[1] for k in tile_di_groups if k[0] == pn})
        if not tile_indices:
            tile_indices = [0]

        for ti in tile_indices:
            tile_lines = tile_di_groups.get((pn, ti), [])
            if not tile_lines:
                continue
            for name in panel_names:
                ms = rule_match_panel_all(name, tile_lines)
                panel_bbox_matches[pn][name].extend(ms)

        unmatched = [n for n, ms in panel_bbox_matches[pn].items() if not ms]
        unmatched_by_page[pn] = unmatched
        hits_n = sum(len(ms) for ms in panel_bbox_matches[pn].values())
        total_di = sum(len(tile_di_groups.get((pn, ti), [])) for ti in tile_indices)
        print(
            f"  page {pn:2d}: {len(panel_names)-len(unmatched)}/{len(panel_names)} matched  "
            f"hits={hits_n}  tiles={len(tile_indices)}  DI={total_di}  "
            f"unmatched={unmatched or '[]'}"
        )


    # ── serial-align: supplement unmatched panels (based on sibling panel coordinates) ─────
    # Search for additional unmatched panels from DI lines near cx/cy of already-matched siblings.
    # Already-matched panels are never touched.
    _ALIGN_TOL = 2.0
    for pn, page_matches in panel_bbox_matches.items():
        di_lines = di_lines_by_page.get(pn, [])
        if not di_lines:
            continue

        # Collect sibling centers of all already-matched panels
        sibling_centers: Dict[str, list] = defaultdict(list)  # prefix -> [(cx, cy)]
        for name, hits in page_matches.items():
            if not hits:
                continue
            p = _parse_serial(name)
            if not p:
                continue
            for h in hits:
                bb = h["bbox"]
                sibling_centers[p[0]].append(((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2))

        # Search sibling-alignment-based DI lines only for unmatched panels
        for name, hits in page_matches.items():
            if hits:
                continue  # Already matched → do not touch
            p = _parse_serial(name)
            if not p:
                continue
            siblings = sibling_centers.get(p[0], [])
            if not siblings:
                continue  # No siblings → skip

            # Collect DI lines near sibling centers then retry rule matching
            nearby = [
                l for l in di_lines
                if any(
                    abs((l["bbox"][0] + l["bbox"][2]) / 2 - scx) <= _ALIGN_TOL
                    or abs((l["bbox"][1] + l["bbox"][3]) / 2 - scy) <= _ALIGN_TOL
                    for scx, scy in siblings
                )
            ]
            if not nearby:
                continue
            ms = rule_match_panel_all(name, nearby)
            if ms:
                for m in ms:
                    m["method"] = "rule-serial-align"
                page_matches[name].extend(ms)
                unmatched_by_page[pn] = [n for n in unmatched_by_page[pn] if n != name]
                print(
                    f"  [serial-align] page {pn}: '{name}' FOUND"
                    f"  ({len(ms)} hit(s) via sibling alignment)"
                )

    print("\nResolving bbox conflicts:")
    for pn in list(panel_bbox_matches):
        before = sum(len(ms) for ms in panel_bbox_matches[pn].values())
        panel_bbox_matches[pn] = resolve_bbox_conflicts(panel_bbox_matches[pn])
        after = sum(len(ms) for ms in panel_bbox_matches[pn].values())
        if before != after:
            print(f"  page {pn:2d}: {before-after} duplicate-bbox claim(s) removed")
        unmatched_by_page[pn] = [n for n, ms in panel_bbox_matches[pn].items() if not ms]

    # ── Snapshot: rule-only state (before LLM fallback) ──────────────────────
    if stages_out is not None:
        import copy as _copy
        stages_out["1_rule_only"] = _copy.deepcopy(panel_bbox_matches)

    # ── LLM fallback: page-level (since we don't know which tile contains unmatched names)
    # Rule-based was done per tile; for names still unmatched, use the full DI lines.
    print("\nLLM fallback:")
    has_fallback = False
    for pn, names in sorted(unmatched_by_page.items()):
        if not names:
            continue
        has_fallback = True
        print(f"  page {pn}: {names}")
        di_lines_all = di_lines_by_page.get(pn, [])
        tiles = tiles_by_page.get(pn) or []
        llm_res = llm_match_panels(
            names, di_lines_all,
            page_images.get(pn), llm_client, deployment,
            tiles=tiles,
            matched_map={n: ms for n, ms in panel_bbox_matches[pn].items() if ms},
        )
        for name, hits in llm_res.items():
            panel_bbox_matches[pn][name] = hits
            for h in hits:
                print(f"    '{name}' -> {h['bbox']}  conf={h.get('confidence', '?')}")
            if not hits:
                print(f"    '{name}' -> NO MATCH")
    if not has_fallback:
        print("  (none)")

    for pn in list(panel_bbox_matches):
        panel_bbox_matches[pn] = resolve_bbox_conflicts(panel_bbox_matches[pn])

    # ── Resolve multi-position matches: when one name matches multiple bboxes ─
    multi_pos_pages = {
        pn for pn, pm in panel_bbox_matches.items()
        if any(len(ms) > 1 for ms in pm.values())
    }
    if multi_pos_pages:
        print("\nResolving multi-position matches:")
        for pn in sorted(multi_pos_pages):
            panel_bbox_matches[pn] = resolve_multi_position_matches(
                panel_bbox_matches[pn],
                di_lines_by_page.get(pn, []),
                page_img=page_images.get(pn),
                llm_client=llm_client,
                deployment=deployment,
                page_num=pn,
            )

    # Serial name correction: fix OCR digit errors using spatial row order
    panel_bbox_matches, result, serial_corrections = correct_serial_outliers(
        panel_bbox_matches, result
    )
    if serial_corrections:
        print("\nSerial name corrections applied:")
        for c in serial_corrections:
            print(c)

    # Gap-fill: assign unmatched serial names to DI lines at expected position
    panel_bbox_matches, result, gap_fills = fill_serial_gaps_by_position(
        panel_bbox_matches, result, di_lines_by_page
    )
    if gap_fills:
        print("\nSerial gap-fills applied:")
        for g in gap_fills:
            print(g)

    # ── Snapshot: after LLM fallback + serial correction + gap-fill ──────────
    if stages_out is not None:
        import copy as _copy
        stages_out["2_llm_fallback"] = _copy.deepcopy(panel_bbox_matches)

    print("\n" + "=" * 60)
    total_names = sum(len(v) for v in result["by_page"].values())
    total_matched = sum(
        1 for pd in panel_bbox_matches.values() for ms in pd.values() if ms
    )
    total_hits = sum(len(ms) for pd in panel_bbox_matches.values() for ms in pd.values())
    print(f"Total: {total_matched}/{total_names} matched  |  {total_hits} bbox hits")
    mc = _Counter(
        m.get("method")
        for pd in panel_bbox_matches.values()
        for ms in pd.values()
        for m in ms
    )
    print("By method:", dict(mc))
    print("=" * 60)
    for pn in sorted(panel_bbox_matches.keys()):
        for name, ms in panel_bbox_matches[pn].items():
            if ms:
                for i, m in enumerate(ms):
                    c = f"  conf={m['confidence']}" if "confidence" in m else ""
                    print(f"  page {pn:2d}  '{name}' #{i+1}  [{m['method']}]{c}  {m['bbox']}")
            else:
                print(f"  page {pn:2d}  '{name}'  [UNMATCHED]  -")

    return panel_bbox_matches, result


# ── DI lines loader ───────────────────────────────────────────────────────────

def load_di_lines(
    crops: list,
    di_dir: "Path",
    notebook_dir: "Path" = None,
) -> Dict[int, list]:
    """Load DI JSON files and translate bbox coords to full-page space.

    Parameters
    ----------
    crops       : List[(page_num, crop_idx, png_path, crop_bbox)]
    di_dir      : Path to ``outputs/missing_image_detection_di``
    notebook_dir: used only for relative path logging

    Returns
    -------
    di_lines_by_page : Dict[int, List[dict]]
        Each entry: {content, bbox (full-page), crop_idx}
    """
    import json as _json

    di_lines_by_page: Dict[int, list] = {}
    for page_num, crop_idx, png_path, crop_bbox in crops:
        di_json_path = di_dir / f"page{page_num}" / f"{png_path.stem}_di.json"
        if not di_json_path.exists():
            rel = di_json_path.resolve().relative_to(notebook_dir) if notebook_dir else di_json_path
            print(f"  [WARN] DI JSON not found: {rel}")
            continue
        with open(di_json_path, encoding="utf-8") as f:
            di_data = _json.load(f)
        cx1, cy1 = crop_bbox[0], crop_bbox[1]
        for line in di_data.get("lines", []):
            bx1, by1, bx2, by2 = line["bbox"]
            di_lines_by_page.setdefault(page_num, []).append({
                "content":  line["content"],
                "bbox":     [cx1 + bx1, cy1 + by1, cx1 + bx2, cy1 + by2],
                "crop_idx": crop_idx,
            })
    for pn, lines in sorted(di_lines_by_page.items()):
        print(f"  page {pn:2d}: {len(lines)} DI lines loaded")
    return di_lines_by_page


# ── Results saver ─────────────────────────────────────────────────────────────

def _save_bbox_by_image(
    panel_bbox_matches: Dict[int, Dict[str, list]],
    crops: list,
    output_root: "Path",
    notebook_dir: "Path" = None,
):
    """Save panel_bbox_matches_by_image.json with image_N_M.png-local coordinates.

    Structure:
        {
          "by_page": {
            "<page_num>": {
              "image_N_M.png": {
                "<panel_name>": [{"bbox": [x1,y1,x2,y2], "method": "..."}]
              }
            }
          }
        }

    The bbox coords are in local pixel space of each image_N_M.png crop.
    """
    import json as _json

    # Build offset map: (page_num, crop_idx) -> (cx1, cy1)
    # crops: List[(page_num, crop_idx, png_path, crop_bbox)]
    offset_map: dict = {}
    for page_num, crop_idx, png_path, crop_bbox in crops:
        cx1 = int(crop_bbox[0]) if crop_bbox is not None else 0
        cy1 = int(crop_bbox[1]) if crop_bbox is not None else 0
        offset_map[(int(page_num), int(crop_idx))] = (cx1, cy1)

    by_image: dict = {"by_page": {}}

    for pn in sorted(panel_bbox_matches.keys()):
        page_key = str(pn)
        by_image["by_page"][page_key] = {}
        for name, hits in panel_bbox_matches[pn].items():
            # ── Rank multi-candidate bboxes by text-length ratio ─────
            # Principle: a true panel label has DI text ≈ the panel name
            # (ratio close to 1.0), while a cross-reference note embeds
            # the name inside longer text (ratio << 1.0).
            # This is keyword-free and works across any drawing style.
            name_alen = max(len(_alphanum_only(name)), 1)

            def _match_quality(h):
                di_text = " ".join(l.get("content", "") for l in h.get("lines", []))
                di_alen = max(len(_alphanum_only(di_text)), 1)
                # Higher ratio → DI text is mostly the panel name → better match
                return -(name_alen / di_alen)  # negate for ascending sort

            sorted_hits = sorted((hits or []), key=_match_quality)
            for hit in sorted_hits:
                lines = hit.get("lines") or []
                if not lines:
                    continue
                crop_idx = lines[0].get("crop_idx")
                if crop_idx is None:
                    continue
                key = (int(pn), int(crop_idx))
                if key not in offset_map:
                    continue
                cx1, cy1 = offset_map[key]
                x1, y1, x2, y2 = [int(v) for v in hit["bbox"]]
                local_bbox = [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]
                crop_key = f"image_{pn}_{crop_idx}.png"
                page_data = by_image["by_page"][page_key]
                page_data.setdefault(crop_key, {}).setdefault(name, []).append({
                    "bbox": local_bbox,
                    "method": hit.get("method", ""),
                })

    out_path = output_root / "panel_bbox_matches_by_image.json"
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(by_image, f, ensure_ascii=False, indent=2)
    rel = out_path.resolve().relative_to(notebook_dir) if notebook_dir else out_path
    print(f"Saved: {rel}")


def save_bbox_results(
    result: dict,
    panel_bbox_matches: Dict[int, Dict[str, list]],
    output_root: "Path",
    notebook_dir: "Path" = None,
    preview_chars: int = 3000,
    crops: list = None,
):
    """Save panel_bbox_matches.json (full-page coords) and optionally
    panel_bbox_matches_by_image.json (image_N_M.png local coords) to output_root.

    NOTE: panel_names.json is intentionally NOT overwritten here.
    It is written once in Step 6 (save_panel_names) from the accurate
    tile-extraction result and must not be modified by bbox matching
    or validation steps which may incorrectly remove/add panel names.

    Parameters
    ----------
    result              : panel names dict (not saved — kept for API compat)
    panel_bbox_matches  : Dict[page_num -> {name -> [hit, ...]}}
    output_root         : directory to write JSON files
    notebook_dir        : for relative-path logging
    preview_chars       : how many chars of bbox JSON to print (0 = silent)
    crops               : if provided, also saves panel_bbox_matches_by_image.json
                          with coordinates in image_N_M.png local space
    """
    import json as _json

    save_data: dict = {}
    for pn in sorted(panel_bbox_matches.keys()):
        save_data[str(pn)] = {}
        for name, ms in panel_bbox_matches[pn].items():
            save_data[str(pn)][name] = (
                [{"bbox": m["bbox"], "method": m["method"]} for m in ms]
                if ms else []
            )
    bbox_path = output_root / "panel_bbox_matches.json"
    with open(bbox_path, "w", encoding="utf-8") as f:
        _json.dump(save_data, f, ensure_ascii=False, indent=2)
    rel_b = bbox_path.resolve().relative_to(notebook_dir) if notebook_dir else bbox_path
    print(f"Saved: {rel_b}")
    if preview_chars:
        print(_json.dumps(save_data, ensure_ascii=False, indent=2)[:preview_chars])

    if crops is not None:
        _save_bbox_by_image(panel_bbox_matches, crops, output_root, notebook_dir)
