"""SVG Grid-based Panel Area Detection.

Uses PDF vector line segments (SVG) to build an H/V grid and then assigns
panel boundaries based on panel-name seed positions from Step 3.

This is a lightweight, LLM-free alternative to the LLM-based panel area
detection.  It works well when the SLD has clear dash-dot / solid / dotted
boundary lines separating panel regions.

Adapted from electronic-single-line-diagram-detection/1_panel_detection.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import cv2
import fitz  # PyMuPDF
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Boundary Style Strategy (simplified — "line" style only)
# ──────────────────────────────────────────────────────────────────────────────

class _LineStyle:
    """Solid-line boundary style parameters."""
    key = "line"
    def use_all_layer_fallback(self): return True
    def block_thick_strokes(self): return True
    def axis_tolerance_px(self): return 6
    def min_segment_px(self): return 8
    def thick_stroke_width_threshold(self): return 4.0
    def thick_block_position_tolerance_px(self): return 14
    def line_cluster_seed_min_len_px(self): return 150
    def line_parallel_pos_gap_px(self): return 70
    def line_parallel_min_overlap_ratio(self): return 0.55
    def line_parallel_thick_avg_len(self): return 650.0
    def line_parallel_thin_avg_len(self): return 220.0
    def max_avg_seg_len_for_boundary_px(self): return float("inf")


_STYLE = _LineStyle()


# ──────────────────────────────────────────────────────────────────────────────
# SVG Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_affine(transform_str: str) -> list:
    if not transform_str:
        return [1, 0, 0, 1, 0, 0]
    m = re.match(r'matrix\(([^)]+)\)', transform_str)
    if m:
        v = list(map(float, re.split(r'[\s,]+', m.group(1).strip())))
        if len(v) == 6:
            return v
    m2 = re.match(r'translate\(([^,)]+),?\s*([^)]*)\)', transform_str)
    if m2:
        tx = float(m2.group(1))
        ty = float(m2.group(2)) if m2.group(2) else 0.0
        return [1, 0, 0, 1, tx, ty]
    return [1, 0, 0, 1, 0, 0]


def _mat_mul(a: list, b: list) -> list:
    return [
        a[0]*b[0]+a[2]*b[1], a[1]*b[0]+a[3]*b[1],
        a[0]*b[2]+a[2]*b[3], a[1]*b[2]+a[3]*b[3],
        a[0]*b[4]+a[2]*b[5]+a[4], a[1]*b[4]+a[3]*b[5]+a[5],
    ]


def _apply_mat(mat, x, y):
    a, b, c, d, e, f = mat
    return a*x + c*y + e, b*x + d*y + f


def _parse_path_segments(d: str, mat: list, img_w, img_h, cx0, cy0, sx, sy):
    segs = []
    tokens = re.findall(r'[MmHhVvLlZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d)
    cur_x = cur_y = 0.0
    cmd = 'M'
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in 'MmHhVvLlZz':
            cmd = t; i += 1; continue
        try:
            if cmd in ('M', 'm'):
                x = float(tokens[i]); y = float(tokens[i+1]); i += 2
                if cmd == 'm': x += cur_x; y += cur_y
                cur_x, cur_y = x, y
                cmd = 'L' if cmd == 'M' else 'l'
            elif cmd in ('L', 'l'):
                x = float(tokens[i]); y = float(tokens[i+1]); i += 2
                if cmd == 'l': x += cur_x; y += cur_y
                segs.append((cur_x, cur_y, x, y))
                cur_x, cur_y = x, y
            elif cmd in ('H', 'h'):
                x = float(tokens[i]); i += 1
                if cmd == 'h': x += cur_x
                segs.append((cur_x, cur_y, x, cur_y))
                cur_x = x
            elif cmd in ('V', 'v'):
                y = float(tokens[i]); i += 1
                if cmd == 'v': y += cur_y
                segs.append((cur_x, cur_y, cur_x, y))
                cur_y = y
            elif cmd in ('Z', 'z'):
                i += 1
            else:
                i += 1
        except (IndexError, ValueError):
            i += 1
    result = []
    for x1, y1, x2, y2 in segs:
        sx1, sy1 = _apply_mat(mat, x1, y1)
        sx2, sy2 = _apply_mat(mat, x2, y2)
        result.append(((sx1 - cx0) * sx, (sy1 - cy0) * sy,
                        (sx2 - cx0) * sx, (sy2 - cy0) * sy))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Extract SVG Segments
# ──────────────────────────────────────────────────────────────────────────────

def _extract_svg_segments(pdf_path: str, page_number: int,
                          clip_coords: list,
                          img_w: int, img_h: int) -> Tuple[list, list]:
    """Extract H/V line segments from PDF SVG (Layer 3 priority, all-layer fallback)."""
    cx0, cy0, cx1, cy1 = clip_coords
    clip_w = cx1 - cx0
    clip_h = cy1 - cy0
    px_per_pt_x = img_w / clip_w
    px_per_pt_y = img_h / clip_h

    pdf_doc = fitz.open(pdf_path)
    pg = pdf_doc[page_number - 1]
    svg_raw = pg.get_svg_image(matrix=fitz.Identity)
    pdf_doc.close()
    svg_text = svg_raw if isinstance(svg_raw, str) else svg_raw.decode("utf-8", errors="replace")

    svg_text_p = svg_text.replace("xmlns:inkscape=", "xmlns:inks=")
    svg_text_p = svg_text_p.replace("inkscape:", "inks:")
    INK_NS = "http://www.inkscape.org/namespaces/inkscape"

    root = ET.fromstring(svg_text_p)

    h_segs, v_segs = [], []
    thick_h, thick_v = [], []
    AXIS_TOL = _STYLE.axis_tolerance_px()
    MIN_SEG = _STYLE.min_segment_px()
    MAX_SW = _STYLE.thick_stroke_width_threshold()

    def _get_sw(node):
        sw = node.get("stroke-width")
        if sw:
            try: return float(sw)
            except Exception: pass
        st_attr = node.get("style", "")
        m = re.search(r"stroke-width\s*:\s*([0-9.]+)", st_attr)
        if m:
            try: return float(m.group(1))
            except Exception: pass
        return None

    def _ok_len(length): return length > MIN_SEG

    def _traverse(node, in_l3, parent_mat):
        tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
        local_mat = _parse_affine(node.get("transform", ""))
        cur_mat = _mat_mul(parent_mat, local_mat)
        lbl = node.get(f"{{{INK_NS}}}label") or node.get("inks:label", "") or ""
        is_l3 = in_l3 or (lbl == "3")
        if tag == "path" and is_l3:
            sw = _get_sw(node)
            d = node.get("d", "")
            path_mat = _mat_mul(parent_mat, _parse_affine(node.get("transform", "")))
            scale = math.sqrt(abs(path_mat[0]*path_mat[3] - path_mat[1]*path_mat[2])) or 1.0
            effective_sw = (sw or 0) * scale
            is_thick = _STYLE.block_thick_strokes() and sw is not None and effective_sw >= MAX_SW
            raw = _parse_path_segments(d, path_mat, img_w, img_h, cx0, cy0, px_per_pt_x, px_per_pt_y)
            for x1, y1, x2, y2 in raw:
                dx, dy = abs(x2-x1), abs(y2-y1)
                if dy < AXIS_TOL and _ok_len(dx):
                    seg = (round((y1+y2)/2), min(x1,x2), max(x1,x2))
                    (thick_h if is_thick else h_segs).append(seg)
                elif dx < AXIS_TOL and _ok_len(dy):
                    seg = (round((x1+x2)/2), min(y1,y2), max(y1,y2))
                    (thick_v if is_thick else v_segs).append(seg)
        for child in node:
            _traverse(child, is_l3, cur_mat if tag == "g" else parent_mat)

    _traverse(root, False, [1,0,0,1,0,0])
    h_segs = [s for s in h_segs if 0 <= s[0] <= img_h and _ok_len(s[2]-s[1])]
    v_segs = [s for s in v_segs if 0 <= s[0] <= img_w and _ok_len(s[2]-s[1])]

    if not h_segs and not v_segs and _STYLE.use_all_layer_fallback():
        def _traverse_all(node, parent_mat):
            tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
            local_mat = _parse_affine(node.get("transform", ""))
            cur_mat = _mat_mul(parent_mat, local_mat)
            if tag == "path":
                sw = _get_sw(node)
                d = node.get("d", "")
                path_mat = _mat_mul(parent_mat, _parse_affine(node.get("transform", "")))
                scale = math.sqrt(abs(path_mat[0]*path_mat[3] - path_mat[1]*path_mat[2])) or 1.0
                effective_sw = (sw or 0) * scale
                is_thick = _STYLE.block_thick_strokes() and sw is not None and effective_sw >= MAX_SW
                raw = _parse_path_segments(d, path_mat, img_w, img_h, cx0, cy0, px_per_pt_x, px_per_pt_y)
                for x1, y1, x2, y2 in raw:
                    dx, dy = abs(x2-x1), abs(y2-y1)
                    if dy < AXIS_TOL and _ok_len(dx):
                        seg = (round((y1+y2)/2), min(x1,x2), max(x1,x2))
                        (thick_h if is_thick else h_segs).append(seg)
                    elif dx < AXIS_TOL and _ok_len(dy):
                        seg = (round((x1+x2)/2), min(y1,y2), max(y1,y2))
                        (thick_v if is_thick else v_segs).append(seg)
            for child in node:
                _traverse_all(child, cur_mat if tag == "g" else parent_mat)
        _traverse_all(root, [1,0,0,1,0,0])
        h_segs = [s for s in h_segs if 0 <= s[0] <= img_h and _ok_len(s[2]-s[1])]
        v_segs = [s for s in v_segs if 0 <= s[0] <= img_w and _ok_len(s[2]-s[1])]
        thick_h = [s for s in thick_h if 0 <= s[0] <= img_h and _ok_len(s[2]-s[1])]
        thick_v = [s for s in thick_v if 0 <= s[0] <= img_w and _ok_len(s[2]-s[1])]

    if _STYLE.block_thick_strokes():
        bl_h = sorted({int(s[0]) for s in thick_h})
        bl_v = sorted({int(s[0]) for s in thick_v})
        tol = _STYLE.thick_block_position_tolerance_px()
        def _is_bl(pos, blocked):
            return any(abs(pos - b) <= tol for b in blocked)
        if bl_h:
            h_segs = [s for s in h_segs if not _is_bl(int(s[0]), bl_h)]
        if bl_v:
            v_segs = [s for s in v_segs if not _is_bl(int(s[0]), bl_v)]

    print(f"    [SVG] Segments: H={len(h_segs)}, V={len(v_segs)}")
    return h_segs, v_segs


# ──────────────────────────────────────────────────────────────────────────────
# Grid Building
# ──────────────────────────────────────────────────────────────────────────────

def _cluster_lines(segs, tol=25, min_segs=3, min_span=100,
                   min_density=0.20, min_avg_seg_len=12.0):
    if not segs:
        return []
    positions = sorted(s[0] for s in segs)
    groups, cur = [], [positions[0]]
    for v in positions[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            groups.append(cur); cur = [v]
    groups.append(cur)
    result = []
    for g in groups:
        pos = round(sum(g) / len(g))
        nearby = [s for s in segs if abs(s[0]-pos) <= tol]
        span_min = min(s[1] for s in nearby)
        span_max = max(s[2] for s in nearby)
        span = span_max - span_min
        avg_len = sum(s[2]-s[1] for s in nearby) / len(nearby)
        total_len = sum(s[2]-s[1] for s in nearby)
        density = total_len / max(span, 1)
        if len(nearby) >= min_segs and span >= min_span and density >= min_density and avg_len >= min_avg_seg_len:
            result.append((pos, round(span_min), round(span_max), round(avg_len)))
    return sorted(result)


def _build_line_segments(segs, lines, tol=25, merge_gap=1, min_seg_len=4):
    line_segments = []
    for pos, *_ in lines:
        intervals = []
        for seg in segs:
            if abs(seg[0]-pos) <= tol:
                s = int(round(min(seg[1], seg[2])))
                e = int(round(max(seg[1], seg[2])))
                if e - s >= min_seg_len:
                    intervals.append((s, e))
        if not intervals:
            line_segments.append([int(pos), []])
            continue
        intervals.sort()
        merged = [list(intervals[0])]
        for s, e in intervals[1:]:
            if s <= merged[-1][1] + merge_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        line_segments.append([int(pos), merged])
    return line_segments


def _prune_close(lines, min_gap=28):
    if not lines: return []
    s = sorted(lines, key=lambda x: x[0])
    pruned = [s[0]]
    for cur in s[1:]:
        prev = pruned[-1]
        if abs(cur[0]-prev[0]) < min_gap:
            if (cur[2]-cur[1], cur[3]) > (prev[2]-prev[1], prev[3]):
                pruned[-1] = cur
        else:
            pruned.append(cur)
    return pruned


def _drop_fragmented(lines, min_avg=200.0):
    return [l for l in lines if float(l[3]) >= min_avg]


def _drop_frame_like(h_lines, v_lines, w, h):
    hk = []
    for y, s0, s1, avg in h_lines:
        span = float(s1)-float(s0)
        yp = float(y)/max(1.0, float(h))
        if not (span >= 0.90*w and (yp <= 0.03 or yp >= 0.97)):
            hk.append((y, s0, s1, avg))
    vk = []
    for x, s0, s1, avg in v_lines:
        span = float(s1)-float(s0)
        xp = float(x)/max(1.0, float(w))
        if not (span >= 0.90*h and (xp <= 0.08 or xp >= 0.92)):
            vk.append((x, s0, s1, avg))
    return hk, vk


def _drop_thick_parallel(lines, pos_gap=70, min_overlap_ratio=0.55,
                         thick_avg=650.0, thin_avg=220.0):
    if not lines: return []
    kept = []
    for i, cur in enumerate(lines):
        cpos, c0, c1, cavg = int(cur[0]), int(cur[1]), int(cur[2]), float(cur[3])
        if cavg < thick_avg:
            kept.append(cur); continue
        cspan = max(1, c1-c0)
        has_thin = False
        for j, alt in enumerate(lines):
            if i == j: continue
            apos, a0, a1, aavg = int(alt[0]), int(alt[1]), int(alt[2]), float(alt[3])
            if abs(apos-cpos) > pos_gap: continue
            if aavg > thin_avg: continue
            overlap = max(0, min(c1,a1)-max(c0,a0))
            if overlap/cspan >= min_overlap_ratio:
                has_thin = True; break
        if not has_thin:
            kept.append(cur)
    return kept


def _drop_thick_middle_vertical(lines, width):
    """Drop thick V lines near the page center that have thinner parallels."""
    if not lines:
        return []
    kept = []
    for i, cur in enumerate(lines):
        cx, c0, c1, cavg = int(cur[0]), int(cur[1]), int(cur[2]), float(cur[3])
        is_middle = (0.45 * width) <= cx <= (0.65 * width)
        if not (is_middle and cavg >= 700.0):
            kept.append(cur); continue
        has_alt = False
        for j, alt in enumerate(lines):
            if i == j: continue
            ax = int(alt[0])
            if abs(ax - cx) > 60: continue
            overlap = max(0, min(c1, int(alt[2])) - max(c0, int(alt[1])))
            if overlap <= 0: continue
            base = max(1, c1 - c0)
            if overlap / base >= 0.5 and float(alt[3]) <= 200.0:
                has_alt = True; break
        if not has_alt:
            kept.append(cur)
    return kept


def _drop_repetitive_lines(lines, min_gap_ratio=0.08, max_rep_count=4):
    """Drop dense repetitive lines (table/BOM rows) that form regular patterns.

    If more than max_rep_count consecutive lines have similar spacing,
    they are likely table rows and should be removed.
    """
    if len(lines) < max_rep_count + 1:
        return lines
    positions = [int(l[0]) for l in lines]
    # Find groups of lines with regular spacing
    to_remove = set()
    for i in range(len(positions)):
        # Look for sequences of regularly spaced lines starting at i
        if i + max_rep_count > len(positions):
            break
        gaps = [positions[j+1] - positions[j] for j in range(i, min(i + max_rep_count + 2, len(positions) - 1))]
        if len(gaps) < max_rep_count:
            continue
        # Check if gaps are similar (within 20%)
        avg_gap = sum(gaps) / len(gaps)
        if avg_gap <= 0:
            continue
        all_similar = all(abs(g - avg_gap) / avg_gap < 0.20 for g in gaps)
        if all_similar:
            for j in range(i, i + len(gaps) + 1):
                to_remove.add(j)
    if not to_remove:
        return lines
    return [l for i, l in enumerate(lines) if i not in to_remove]


def _filter_structural_boundaries(h_lines, v_lines, img_w, img_h,
                                  h_min_span_ratio=0.30, v_min_span_ratio=0.40):
    """Keep only structurally significant lines that span a large portion of the page.

    Panel boundary lines typically span across a significant portion of the
    page width (for H lines) or height (for V lines), while table/BOM/circuit
    lines are localized.
    """
    h_min_span = img_w * h_min_span_ratio
    v_min_span = img_h * v_min_span_ratio

    h_filtered = [l for l in h_lines if (l[2] - l[1]) >= h_min_span]
    v_filtered = [l for l in v_lines if (l[2] - l[1]) >= v_min_span]

    # Only apply if we have enough structural lines remaining
    if len(h_filtered) >= 3:
        h_lines = h_filtered
    if len(v_filtered) >= 3:
        v_lines = v_filtered

    # For V lines: also prune closely spaced lines (likely table columns)
    # Use 6% of page width as minimum gap
    v_min_gap = max(50, int(img_w * 0.06))
    v_lines = _prune_close(v_lines, min_gap=v_min_gap)

    return h_lines, v_lines


def _build_grid(h_segs, v_segs, img_w, img_h):
    """Build grid with multi-strategy approach: try line-style first, then dash-dot fallback."""
    min_seed_len = _STYLE.line_cluster_seed_min_len_px()  # 150px
    h_segs_long = [s for s in h_segs if (s[2]-s[1]) >= min_seed_len]

    # --- Phase 1: Line-style clustering (solid boundary lines) ---
    h_lines = _cluster_lines(h_segs_long, tol=25, min_segs=1, min_span=200,
                              min_density=0.60, min_avg_seg_len=100.0)
    v_lines = _cluster_lines(v_segs, tol=25, min_segs=1, min_span=200,
                              min_density=0.60, min_avg_seg_len=50.0)
    if len(h_lines) < 4:
        h2 = _cluster_lines(h_segs_long, tol=25, min_segs=1, min_span=100,
                             min_density=0.40, min_avg_seg_len=60.0)
        if len(h2) > len(h_lines): h_lines = h2
    if len(v_lines) < 2:
        v2 = _cluster_lines(v_segs, tol=25, min_segs=1, min_span=100,
                             min_density=0.40, min_avg_seg_len=30.0)
        if len(v2) > len(v_lines): v_lines = v2

    # For V lines: also try dash-dot style clustering,
    # because panel boundary V lines often have low avg_seg_len (dash-dot).
    v_dd = _cluster_lines(v_segs, tol=25, min_segs=5, min_span=120,
                           min_density=0.20, min_avg_seg_len=12.0)
    if len(v_dd) > len(v_lines):
        v_lines = v_dd  # Replace: Phase 2 finds more lines including boundaries

    # --- Apply filters ---
    h_lines = _prune_close(h_lines, 28)
    v_lines = _prune_close(v_lines, 28)
    if h_lines and max(float(l[3]) for l in h_lines) > 200:
        h_lines = _drop_fragmented(h_lines, 200.0)
    # Note: don't drop_fragmented V lines — dash-dot V boundaries have low avg
    h_lines, v_lines = _drop_frame_like(h_lines, v_lines, img_w, img_h)
    v_lines = _drop_thick_middle_vertical(v_lines, img_w)
    v_lines = _drop_thick_parallel(v_lines,
                                    pos_gap=_STYLE.line_parallel_pos_gap_px(),
                                    thick_avg=_STYLE.line_parallel_thick_avg_len(),
                                    thin_avg=_STYLE.line_parallel_thin_avg_len())
    h_lines = _drop_thick_parallel(h_lines,
                                    pos_gap=_STYLE.line_parallel_pos_gap_px(),
                                    thick_avg=_STYLE.line_parallel_thick_avg_len(),
                                    thin_avg=_STYLE.line_parallel_thin_avg_len())

    # --- Structural boundary filter: remove localized table/BOM/circuit lines ---
    # (Applied before Phase 2 for H lines only)
    h_struct, _ = _filter_structural_boundaries(h_lines, [], img_w, img_h)
    if len(h_struct) >= 3:
        h_lines = h_struct
    h_lines = _drop_repetitive_lines(h_lines)

    # --- Phase 2: Post-filter fallback (dash-dot style if line-style was filtered out) ---
    if len(h_lines) < 2:
        h_dd = _cluster_lines(h_segs, tol=25, min_segs=5, min_span=120,
                               min_density=0.20, min_avg_seg_len=12.0)
        if len(h_dd) < 4:
            h_dd2 = _cluster_lines(h_segs, tol=25, min_segs=3, min_span=80,
                                    min_density=0.10, min_avg_seg_len=8.0)
            if len(h_dd2) > len(h_dd): h_dd = h_dd2
        h_dd = _prune_close(h_dd, 28)
        h_dd, _ = _drop_frame_like(h_dd, [], img_w, img_h)
        if len(h_dd) > len(h_lines):
            h_lines = h_dd

    if len(v_lines) < 6:
        # V lines: use broader clustering since Phase 1 may miss dash-dot lines
        v_dd = _cluster_lines(v_segs, tol=25, min_segs=5, min_span=120,
                               min_density=0.20, min_avg_seg_len=12.0)
        if len(v_dd) < 4:
            v_dd2 = _cluster_lines(v_segs, tol=25, min_segs=3, min_span=80,
                                    min_density=0.10, min_avg_seg_len=8.0)
            if len(v_dd2) > len(v_dd): v_dd = v_dd2
        v_dd = _prune_close(v_dd, 28)
        _, v_dd = _drop_frame_like([], v_dd, img_w, img_h)
        if len(v_dd) > len(v_lines):
            v_lines = v_dd

    # --- Post Phase 2: Apply structural filter to V lines ---
    _, v_struct = _filter_structural_boundaries([], v_lines, img_w, img_h)
    if len(v_struct) >= 3:
        v_lines = v_struct
    v_lines = _drop_repetitive_lines(v_lines)
    # Also apply frame-like filter to Phase 2 V lines
    _, v_lines = _drop_frame_like([], v_lines, img_w, img_h)

    h_line_segments = _build_line_segments(h_segs, h_lines, tol=25)
    v_line_segments = _build_line_segments(v_segs, v_lines, tol=25)
    h_pct = [round(l[0]/img_h*100, 2) for l in h_lines]
    v_pct = [round(l[0]/img_w*100, 2) for l in v_lines]
    grid = {
        "h_lines": h_lines, "v_lines": v_lines,
        "h_pct": h_pct, "v_pct": v_pct,
        "h_line_segments": h_line_segments,
        "v_line_segments": v_line_segments,
    }
    print(f"    [SVG] Grid: H={len(h_lines)} lines, V={len(v_lines)} lines")
    return grid


# ──────────────────────────────────────────────────────────────────────────────
# Segment map / support helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_segment_map(line_segments):
    sm = {}
    if not line_segments:
        return sm
    for item in line_segments:
        try:
            pos = int(item[0])
            segs = []
            for seg in (item[1] if len(item) > 1 else []):
                if len(seg) < 2: continue
                s0, s1 = int(seg[0]), int(seg[1])
                if s1 < s0: s0, s1 = s1, s0
                segs.append((s0, s1))
            sm[pos] = segs
        except Exception:
            continue
    return sm


def _collect_intervals(sm, pos_key, pos_tol=2):
    intervals = []
    p = int(round(pos_key))
    for k in range(p - pos_tol, p + pos_tol + 1):
        intervals.extend(sm.get(k, []))
    return intervals


def _covers_value(intervals, value, pad=5.0):
    if value is None: return False
    for s0, s1 in intervals:
        if (s0-pad) <= value <= (s1+pad):
            return True
    return False


def _overlaps_range(intervals, r0, r1, pad=0.0):
    if r1 < r0: r0, r1 = r1, r0
    for s0, s1 in intervals:
        if (s1+pad) >= r0 and (s0-pad) <= r1:
            return True
    return False


def _overlap_ratio(intervals, r0, r1):
    if r1 < r0: r0, r1 = r1, r0
    base = max(1.0, float(r1-r0))
    ov = 0.0
    for s0, s1 in intervals:
        o0 = max(float(s0), float(r0))
        o1 = min(float(s1), float(r1))
        if o1 > o0: ov += (o1-o0)
    return min(1.0, ov/base)


def _interval_overlap_len(intervals, r0, r1):
    if r1 < r0: r0, r1 = r1, r0
    ov = 0.0
    for s0, s1 in intervals:
        o0 = max(float(s0), float(r0))
        o1 = min(float(s1), float(r1))
        if o1 > o0: ov += (o1-o0)
    return ov


def _line_support_len(sm, pos_key, r0, r1, pos_tol=2):
    return _interval_overlap_len(_collect_intervals(sm, pos_key, pos_tol), r0, r1)


def _pick_boundary_with_support(pos_px, lines, direction, sm,
                                support_range, pos_tol=2,
                                min_support_ratio=0.45, min_support_px=50.0,
                                exclude_pos=None):
    r0, r1 = support_range
    if r1 < r0: r0, r1 = r1, r0
    base = max(1.0, float(r1-r0))
    min_sup = max(float(min_support_px), base * float(min_support_ratio))
    candidates = []
    for l in lines:
        pos = int(l[0])
        if exclude_pos is not None and pos == int(exclude_pos): continue
        if direction in ("above", "left") and not (pos < pos_px): continue
        if direction in ("below", "right") and not (pos > pos_px): continue
        candidates.append(pos)
    candidates.sort(key=lambda p: abs(p-pos_px))
    for pos in candidates:
        if _line_support_len(sm, pos, r0, r1, pos_tol) >= min_sup:
            return pos
    return None


def _find_nearest_boundary(pos_px, lines, direction,
                           orth_pos=None, orth_range=None,
                           line_segments=None,
                           segment_pad=5.0, pos_tol=2,
                           min_overlap_ratio=0.2):
    """Find nearest grid line in the given direction, preferring closer lines.

    Uses distance-first priority: among all valid candidate lines,
    pick the closest one. Segment support is used as a tiebreaker, not a filter,
    because boundary lines may have gaps at circuit connection points.
    """
    sm = _build_segment_map(line_segments)
    candidates = []  # (distance, support_score, pos)
    for l in lines:
        pos, span_min, span_max = int(l[0]), l[1], l[2]
        if direction in ("above", "left") and not (pos < pos_px):
            continue
        if direction in ("below", "right") and not (pos > pos_px):
            continue
        dist = abs(pos - pos_px)
        # Compute support score: how well the line covers the seed position
        support = 0
        if sm:
            intervals = _collect_intervals(sm, pos, pos_tol)
            if not intervals:
                support = 0
            elif orth_pos is not None and _covers_value(intervals, orth_pos, pad=segment_pad):
                support = 3  # strict: covers seed position
            elif orth_range is not None:
                r0, r1 = orth_range
                if _overlaps_range(intervals, r0, r1):
                    support = 2  # overlaps seed range
                elif _overlap_ratio(intervals, r0, r1) >= min_overlap_ratio:
                    support = 1  # partial overlap
                else:
                    support = 0.5  # has segments but not covering seed
            else:
                support = 0.5  # has segments
        else:
            span_pad = 12.0
            if orth_pos is not None and (span_min - span_pad <= orth_pos <= span_max + span_pad):
                support = 3
            else:
                support = 0.5
        if support > 0:
            candidates.append((dist, -support, pos))

    if not candidates:
        return None
    # Sort by distance first (closest), then by support (higher is better via negative)
    candidates.sort()
    return candidates[0][2]


# ──────────────────────────────────────────────────────────────────────────────
# Panel Boundary Assignment
# ──────────────────────────────────────────────────────────────────────────────

def _assign_panel_boundaries(seeds: list, grid: dict,
                             img_w: int, img_h: int) -> list:
    """Assign panel bounding boxes using seed positions + grid lines."""
    h_lines = grid["h_lines"]
    v_lines = grid["v_lines"]
    h_line_segments = grid.get("h_line_segments", [])
    v_line_segments = grid.get("v_line_segments", [])
    h_sm = _build_segment_map(h_line_segments)
    v_sm = _build_segment_map(v_line_segments)

    EDGE_PAD = 5
    snap_tol_pct = 3.0
    segment_pad = 5.0
    pos_tol = 2
    min_overlap_ratio = 0.2
    pair_len_ratio_min = 0.60
    min_support_ratio = 0.45
    min_support_px = 50.0

    panels = []
    for seed in seeds:
        name = seed["panel_name"]
        sx = seed["seed_x"]
        sy = seed["seed_y"]

        # Nudge seed past nearby grid lines so the panel name position
        # (which is typically AT the top/left boundary) ends up inside the cell.
        snap_tol = max(50, int(min(img_h, img_w) * 0.01))
        for l in h_lines:
            if 0 < l[0] - sy < snap_tol:
                sy = l[0] + snap_tol
                break
        for l in v_lines:
            if 0 < l[0] - sx < snap_tol:
                sx = l[0] + snap_tol
                break

        x_range = (max(0.0, sx-20.0), min(float(img_w-1), sx+20.0))
        y_range = (max(0.0, sy-20.0), min(float(img_h-1), sy+20.0))

        # Top
        top = _find_nearest_boundary(sy, h_lines, "above",
                                      orth_pos=sx, orth_range=x_range,
                                      line_segments=h_line_segments,
                                      segment_pad=segment_pad, pos_tol=pos_tol,
                                      min_overlap_ratio=min_overlap_ratio)
        top = top if top is not None else EDGE_PAD

        # Bottom
        bot = _find_nearest_boundary(sy, h_lines, "below",
                                      orth_pos=sx, orth_range=x_range,
                                      line_segments=h_line_segments,
                                      segment_pad=segment_pad, pos_tol=pos_tol,
                                      min_overlap_ratio=min_overlap_ratio)
        bot = bot if bot is not None else (img_h - EDGE_PAD)

        y_panel_range = (max(0.0, top-20.0), min(float(img_h-1), bot+20.0))

        # Left
        lft = _find_nearest_boundary(sx, v_lines, "left",
                                      orth_pos=sy, orth_range=y_panel_range,
                                      line_segments=v_line_segments,
                                      segment_pad=segment_pad, pos_tol=pos_tol,
                                      min_overlap_ratio=min_overlap_ratio)
        lft = lft if lft is not None else EDGE_PAD

        # Right
        rgt = _find_nearest_boundary(sx, v_lines, "right",
                                      orth_pos=sy, orth_range=y_panel_range,
                                      line_segments=v_line_segments,
                                      segment_pad=segment_pad, pos_tol=pos_tol,
                                      min_overlap_ratio=min_overlap_ratio)
        rgt = rgt if rgt is not None else (img_w - EDGE_PAD)

        # Validate seed inside bbox
        if not (lft <= sx <= rgt and top <= sy <= bot):
            top = _find_nearest_boundary(sy, h_lines, "above",
                                          orth_pos=sx, orth_range=x_range,
                                          line_segments=h_line_segments,
                                          segment_pad=segment_pad, pos_tol=pos_tol) or EDGE_PAD
            bot = _find_nearest_boundary(sy, h_lines, "below",
                                          orth_pos=sx, orth_range=x_range,
                                          line_segments=h_line_segments,
                                          segment_pad=segment_pad, pos_tol=pos_tol) or (img_h-EDGE_PAD)
            y_panel_range = (max(0.0, top-20.0), min(float(img_h-1), bot+20.0))
            lft = _find_nearest_boundary(sx, v_lines, "left",
                                          orth_pos=sy, orth_range=y_panel_range,
                                          line_segments=v_line_segments,
                                          segment_pad=segment_pad) or EDGE_PAD
            rgt = _find_nearest_boundary(sx, v_lines, "right",
                                          orth_pos=sy, orth_range=y_panel_range,
                                          line_segments=v_line_segments,
                                          segment_pad=segment_pad) or (img_w-EDGE_PAD)

        # Support length consistency check — disabled in favor of distance-first
        # boundary selection which already handles boundary gaps well.
        # The old support check could cause cascading errors when V lines
        # include non-boundary lines (table columns, bus bars), leading to
        # narrow x-ranges that falsely invalidate correct H boundaries.

        panels.append({
            "panel_name": name,
            "x0": int(min(lft,rgt)), "y0": int(min(top,bot)),
            "x1": int(max(lft,rgt)), "y1": int(max(top,bot)),
            "seed_x": round(sx), "seed_y": round(sy),
        })

    return panels


def _resolve_duplicates(panels, img_w, img_h):
    """Resolve overlapping panels by subdividing shared bboxes based on seeds."""
    seen = set()
    unique = []
    for p in panels:
        if p["panel_name"] not in seen:
            seen.add(p["panel_name"])
            unique.append(p)

    # Group panels with identical or near-identical bboxes
    iou_thresh = 0.85
    groups = []
    used = [False] * len(unique)
    for i, p in enumerate(unique):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(unique)):
            if used[j]:
                continue
            # Check IoU
            xi0 = max(p["x0"], unique[j]["x0"])
            yi0 = max(p["y0"], unique[j]["y0"])
            xi1 = min(p["x1"], unique[j]["x1"])
            yi1 = min(p["y1"], unique[j]["y1"])
            inter = max(0, xi1 - xi0) * max(0, yi1 - yi0)
            a1 = max(1, (p["x1"] - p["x0"]) * (p["y1"] - p["y0"]))
            a2 = max(1, (unique[j]["x1"] - unique[j]["x0"]) * (unique[j]["y1"] - unique[j]["y0"]))
            iou = inter / max(1, a1 + a2 - inter)
            if iou >= iou_thresh:
                group.append(j)
                used[j] = True
        groups.append(group)

    result = []
    for group in groups:
        if len(group) == 1:
            result.append(unique[group[0]])
            continue

        # Multiple panels share the same bbox — subdivide
        members = [unique[i] for i in group]
        x0 = min(m["x0"] for m in members)
        y0 = min(m["y0"] for m in members)
        x1 = max(m["x1"] for m in members)
        y1 = max(m["y1"] for m in members)

        seeds_x = [m["seed_x"] for m in members]
        seeds_y = [m["seed_y"] for m in members]
        x_spread = max(seeds_x) - min(seeds_x)
        y_spread = max(seeds_y) - min(seeds_y)

        if y_spread > x_spread:
            # Split horizontally (seeds stacked vertically)
            sorted_members = sorted(members, key=lambda m: m["seed_y"])
            n = len(sorted_members)
            for idx, m in enumerate(sorted_members):
                if idx == 0:
                    new_y0 = y0
                else:
                    prev_sy = sorted_members[idx - 1]["seed_y"]
                    new_y0 = int((prev_sy + m["seed_y"]) / 2)
                if idx == n - 1:
                    new_y1 = y1
                else:
                    next_sy = sorted_members[idx + 1]["seed_y"]
                    new_y1 = int((m["seed_y"] + next_sy) / 2)
                r = dict(m)
                r["x0"], r["y0"], r["x1"], r["y1"] = x0, new_y0, x1, new_y1
                result.append(r)
        else:
            # Split vertically (seeds stacked horizontally)
            sorted_members = sorted(members, key=lambda m: m["seed_x"])
            n = len(sorted_members)
            for idx, m in enumerate(sorted_members):
                if idx == 0:
                    new_x0 = x0
                else:
                    prev_sx = sorted_members[idx - 1]["seed_x"]
                    new_x0 = int((prev_sx + m["seed_x"]) / 2)
                if idx == n - 1:
                    new_x1 = x1
                else:
                    next_sx = sorted_members[idx + 1]["seed_x"]
                    new_x1 = int((m["seed_x"] + next_sx) / 2)
                r = dict(m)
                r["x0"], r["y0"], r["x1"], r["y1"] = new_x0, y0, new_x1, y1
                result.append(r)

    # Clamp to image bounds
    final = []
    for p in result:
        x0 = max(0, min(img_w-1, int(p["x0"])))
        x1 = max(0, min(img_w-1, int(p["x1"])))
        y0 = max(0, min(img_h-1, int(p["y0"])))
        y1 = max(0, min(img_h-1, int(p["y1"])))
        if x1 <= x0: x1 = min(img_w-1, x0+1)
        if y1 <= y0: y1 = min(img_h-1, y0+1)
        r = dict(p)
        r["x0"], r["x1"], r["y0"], r["y1"] = x0, x1, y0, y1
        final.append(r)
    return sorted(final, key=lambda p: (p["y0"], p["x0"]))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_panels_via_svg(
    pdf_path: str,
    page_number: int,
    page_png_path: str,
    panel_names: List[str],
    name_bboxes: Dict[str, Optional[List[int]]],
    output_dir: str,
) -> Tuple[List[dict], dict]:
    """Detect panel areas using SVG grid lines.

    Args:
        pdf_path: Path to the source PDF file.
        page_number: 1-based page number.
        page_png_path: Path to the rendered page PNG.
        panel_names: List of panel names (from Step 3).
        name_bboxes: Dict mapping panel name → [x1,y1,x2,y2] bbox in page
                     pixel coords (from Step 3 matches). May be None.
        output_dir: Directory for debug output.

    Returns:
        (panels, info) where panels is a list of dicts with keys:
            panel_name, bbox (x0,y0,x1,y1), seed_x, seed_y
        and info is a dict with grid statistics and quality metrics.
    """
    img = cv2.imread(page_png_path)
    if img is None:
        print(f"[SVG Panel] Cannot load image: {page_png_path}")
        return [], {"error": "cannot load image"}
    img_h, img_w = img.shape[:2]

    # Get page clip coordinates (full page in PDF points)
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[page_number - 1]
    rect = page.rect
    clip_coords = [rect.x0, rect.y0, rect.x1, rect.y1]
    pdf_doc.close()

    print(f"[SVG Panel] Page {page_number}: {img_w}x{img_h}px, "
          f"{len(panel_names)} names, clip={clip_coords}")

    # Step 1: Extract SVG segments
    h_segs, v_segs = _extract_svg_segments(pdf_path, page_number,
                                            clip_coords, img_w, img_h)
    if not h_segs and not v_segs:
        print("[SVG Panel] No SVG segments found — cannot proceed")
        return [], {"error": "no segments", "h_segs": 0, "v_segs": 0}

    # Step 2: Build grid
    grid = _build_grid(h_segs, v_segs, img_w, img_h)
    if not grid["h_lines"] and not grid["v_lines"]:
        print("[SVG Panel] No grid lines found — cannot proceed")
        return [], {"error": "no grid", "h_segs": len(h_segs), "v_segs": len(v_segs)}

    # Step 3: Build seeds from name bboxes
    seeds = []
    for name in panel_names:
        bbox = name_bboxes.get(name)
        if bbox and len(bbox) >= 4:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            seeds.append({"panel_name": name, "seed_x": cx, "seed_y": cy})
        else:
            print(f"    [SVG Panel] Skipping '{name}' — no bbox available")

    if not seeds:
        print("[SVG Panel] No seeds available — cannot assign boundaries")
        return [], {"error": "no seeds", "grid_h": len(grid["h_lines"]),
                    "grid_v": len(grid["v_lines"])}

    # Step 4: Assign panel boundaries
    panels_raw = _assign_panel_boundaries(seeds, grid, img_w, img_h)
    panels = _resolve_duplicates(panels_raw, img_w, img_h)

    # Save debug visualization
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _save_debug_viz(page_png_path, panels, grid, output_dir, img_w, img_h)

    # Quality metrics
    valid_count = 0
    for p in panels:
        w = p["x1"] - p["x0"]
        h = p["y1"] - p["y0"]
        area_ratio = (w * h) / (img_w * img_h)
        if 0.002 < area_ratio < 0.5 and w > 50 and h > 50:
            valid_count += 1

    info = {
        "method": "svg_grid",
        "h_segs": len(h_segs),
        "v_segs": len(v_segs),
        "grid_h_lines": len(grid["h_lines"]),
        "grid_v_lines": len(grid["v_lines"]),
        "total_panels": len(panels),
        "valid_panels": valid_count,
        "requested_panels": len(panel_names),
        "matched_ratio": valid_count / max(1, len(panel_names)),
    }

    print(f"[SVG Panel] Result: {len(panels)} panels detected, "
          f"{valid_count}/{len(panel_names)} valid")
    for p in panels:
        print(f"    {p['panel_name']:>12s}: "
              f"({p['x0']},{p['y0']}) → ({p['x1']},{p['y1']})  "
              f"{p['x1']-p['x0']}x{p['y1']-p['y0']}")

    return panels, info


def _save_debug_viz(png_path, panels, grid, output_dir, img_w, img_h):
    """Save debug visualization showing grid lines and panel boxes."""
    try:
        img = cv2.imread(png_path)
        if img is None:
            return
        # Draw grid lines
        for l in grid["h_lines"]:
            y = int(l[0])
            cv2.line(img, (0, y), (img_w, y), (200, 200, 200), 1)
        for l in grid["v_lines"]:
            x = int(l[0])
            cv2.line(img, (x, 0), (x, img_h), (200, 200, 200), 1)
        # Draw panels
        colors = [(255,60,60),(60,100,255),(40,180,40),(255,160,0),
                  (180,60,220),(0,200,200),(200,80,80),(80,200,120)]
        for i, p in enumerate(panels):
            c = colors[i % len(colors)]
            cv2.rectangle(img, (p["x0"],p["y0"]), (p["x1"],p["y1"]), c, 3)
            cv2.putText(img, p["panel_name"], (p["x0"]+4, p["y0"]+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        out_path = str(Path(output_dir) / "svg_panel_debug.png")
        # Resize for manageable file size
        scale = min(1.0, 4000 / max(img_w, img_h))
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.imwrite(out_path, img)
        print(f"    [SVG Panel] Debug viz saved: {out_path}")
    except Exception as e:
        print(f"    [SVG Panel] Debug viz error: {e}")
