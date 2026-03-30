"""
figure_subcropping.py
----------------------
Sub-cropping pipeline functions extracted from 04_figure_subcropping.ipynb.

Supports modes: SVG, LLM, SMART.

Globals that were notebook-level constants become explicit function parameters.
LLM-calling functions require (llm_client, deployment) to be passed.
Path-loading helpers require outputs_dir to be passed.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Core geometry
# ─────────────────────────────────────────────────────────────────────────────

def cuts_to_slices(length: int, cuts: List[int], overlap: int) -> List[Tuple[int, int]]:
    """Convert cut positions to (start, end) slice pairs with overlap.

    Each slice is [start, end) in pixel coordinates.
    """
    boundaries = [0] + sorted(cuts) + [length]
    slices = []
    for i in range(len(boundaries) - 1):
        s = max(0, boundaries[i] - (overlap if i > 0 else 0))
        e = min(length, boundaries[i + 1] + (overlap if i < len(boundaries) - 2 else 0))
        slices.append((s, e))
    return slices


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — SVG / DI data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_polygon_txt(txt_path: Path) -> Tuple[int, int, int, int]:
    """4-corner polygon txt → (x1, y1, x2, y2) bounding box in page px."""
    coords = []
    for line in txt_path.read_text().strip().splitlines():
        x, y = [float(v) for v in line.split(',')]
        coords.append((x, y))
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def load_crop_bbox_in_page(
    page_num: int,
    crop_idx: int,
    figure_crops_dir: Path,
) -> Optional[Tuple[int, int, int, int]]:
    """Load the full-page px bounding box from figure_crops/pageN/image_N_M.txt."""
    txt_path = figure_crops_dir / f'page{page_num}' / f'image_{page_num}_{crop_idx}.txt'
    if txt_path.exists():
        return _load_polygon_txt(txt_path)
    return None


def load_text_bboxes_in_page(
    page_num: int,
    outputs_dir: Path,
) -> List[Tuple[int, int, int, int]]:
    """Load all text bounding boxes (full-page px) for a given page."""
    text_dir = outputs_dir / f'page{page_num}' / 'text'
    if not text_dir.exists():
        return []
    bboxes = []
    for txt_file in sorted(text_dir.glob('images_*.txt')):
        try:
            bboxes.append(_load_polygon_txt(txt_file))
        except Exception:
            continue
    return bboxes


def text_bboxes_to_crop_local(
    page_bboxes: List[Tuple[int, int, int, int]],
    crop_bbox: Tuple[int, int, int, int],
) -> List[Tuple[int, int, int, int]]:
    """Convert full-page px text bboxes to crop-local coords (only those inside crop)."""
    cx1, cy1, cx2, cy2 = crop_bbox
    local = []
    for tx1, ty1, tx2, ty2 in page_bboxes:
        if tx2 < cx1 or tx1 > cx2 or ty2 < cy1 or ty1 > cy2:
            continue
        local.append((tx1 - cx1, ty1 - cy1, tx2 - cx1, ty2 - cy1))
    return local


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — SVG mode: separator-based tiling
# ─────────────────────────────────────────────────────────────────────────────


def svg_compute_tiles(
    img_arr_rgb: np.ndarray,
    img_w: int,
    img_h: int,
    svg_sep_positions: Dict,
    text_bboxes_crop: List[Tuple[int, int, int, int]],
    min_tiles: int,
    max_tiles: int,
    overlap: int,
    text_margin: int,
    tolerance_ratio: float,
    ws_remove_fraction: float = 0.20,
    ws_tile_max_white: float = 0.97,
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """Return tile bboxes using adaptive per-strip secondary cuts (no LLM).

    Algorithm:
    1. Detect rectangular whitespace blocks globally (highest-priority cut candidates).
    2. Determine primary axis (vertical for wide images, horizontal for tall).
    3. Find primary cuts using WS block edges > WS band mids > SVG separators.
    4. For each primary strip: detect per-strip WS blocks + bands; find secondary
       cuts with require_evidence=True — strips without a WS block / band / SVG line
       are kept as a single tile (no forced ideal-position cut).
    5. Filter near-white tiles.
    """
    h_all = svg_sep_positions.get('h', [])
    v_all = svg_sep_positions.get('v', [])

    clean_h = [y for y in h_all
               if not any(ty1 - text_margin <= y <= ty2 + text_margin
                          for _, ty1, _, ty2 in text_bboxes_crop)]
    clean_v = [x for x in v_all
               if not any(tx1 - text_margin <= x <= tx2 + text_margin
                          for tx1, _, tx2, _ in text_bboxes_crop)]

    gray = np.mean(img_arr_rgb.astype(np.float32) / 255.0, axis=2)

    # Detect rectangular whitespace blocks first (highest-priority cut candidates)
    ws_blocks = _detect_ws_blocks(gray, min_area_fraction=0.02, white_threshold=0.95)
    h_block_edges, v_block_edges = _ws_blocks_to_cut_edges(ws_blocks, img_w, img_h, edge_margin=50)

    # Also detect 1D whitespace bands as fallback
    h_ws_bands, v_ws_bands = _detect_ws_bands(gray, min_fraction=ws_remove_fraction)
    h_ws_mids = [(s + e) // 2 for s, e in h_ws_bands]
    v_ws_mids = [(s + e) // 2 for s, e in v_ws_bands]

    area = img_w * img_h
    total_tiles = min(max_tiles, max(min_tiles, int(np.sqrt(area / (1500 * 1500))) + min_tiles - 1))
    if img_w >= img_h:
        n_v = max(1, total_tiles // 2) if img_h >= 500 else max(1, total_tiles - 1)
        n_h = 1 if img_h >= 500 else 0
    else:
        n_h = max(1, total_tiles // 2) if img_w >= 500 else max(1, total_tiles - 1)
        n_v = 1 if img_w >= 500 else 0
    if img_w >= 2500 and img_h >= 2000:
        n_v = max(2, total_tiles // 2)
        n_h = max(1, total_tiles // 2)

    tol_h = int(img_h * tolerance_ratio)
    tol_v = int(img_w * tolerance_ratio)

    primary_axis = 'v' if img_w >= img_h else 'h'

    if primary_axis == 'v':
        x_cuts = _smart_pick_cuts(img_w, n_v, v_block_edges, v_ws_mids, clean_v, v_all, tol_v)
        x_slices = cuts_to_slices(img_w, x_cuts, overlap)
        tiles, strip_y_cuts, strip_x_cuts = [], [], []
        y_cuts: List[int] = []
        for x1, x2 in x_slices:
            strip_gray = gray[:, x1:x2]
            strip_blocks = _detect_ws_blocks(strip_gray, min_area_fraction=0.02)
            strip_h_edges, _ = _ws_blocks_to_cut_edges(strip_blocks, x2 - x1, img_h, edge_margin=30)
            sh_ws_bands, _ = _detect_ws_bands(strip_gray, min_fraction=ws_remove_fraction)
            sh_ws_mids = [(s + e) // 2 for s, e in sh_ws_bands]
            y_c = _smart_pick_cuts(img_h, n_h, strip_h_edges, sh_ws_mids, clean_h, h_all, tol_h,
                                   require_evidence=True)
            strip_y_cuts.append(y_c)
            for y1, y2 in cuts_to_slices(img_h, y_c, overlap):
                t = (x1, y1, x2, y2)
                if not _is_tile_white(img_arr_rgb[y1:y2, x1:x2], ws_tile_max_white):
                    tiles.append(t)
        print(f'  [SVG] primary=V  x_cuts={x_cuts}')
        for i, (x1, x2) in enumerate(x_slices):
            print(f'    strip x={x1}-{x2}: y_cuts={strip_y_cuts[i]}')
    else:
        y_cuts = _smart_pick_cuts(img_h, n_h, h_block_edges, h_ws_mids, clean_h, h_all, tol_h)
        y_slices = cuts_to_slices(img_h, y_cuts, overlap)
        tiles, strip_x_cuts, strip_y_cuts = [], [], []
        x_cuts: List[int] = []
        for y1, y2 in y_slices:
            strip_gray = gray[y1:y2, :]
            strip_blocks = _detect_ws_blocks(strip_gray, min_area_fraction=0.02)
            _, strip_v_edges = _ws_blocks_to_cut_edges(strip_blocks, img_w, y2 - y1, edge_margin=30)
            _, sv_ws_bands = _detect_ws_bands(strip_gray, min_fraction=ws_remove_fraction)
            sv_ws_mids = [(s + e) // 2 for s, e in sv_ws_bands]
            x_c = _smart_pick_cuts(img_w, n_v, strip_v_edges, sv_ws_mids, clean_v, v_all, tol_v,
                                   require_evidence=True)
            strip_x_cuts.append(x_c)
            for x1_t, x2_t in cuts_to_slices(img_w, x_c, overlap):
                t = (x1_t, y1, x2_t, y2)
                if not _is_tile_white(img_arr_rgb[y1:y2, x1_t:x2_t], ws_tile_max_white):
                    tiles.append(t)
        print(f'  [SVG] primary=H  y_cuts={y_cuts}')
        for i, (y1, y2) in enumerate(y_slices):
            print(f'    strip y={y1}-{y2}: x_cuts={strip_x_cuts[i]}')

    print(f'  [SVG] WS blocks={len(ws_blocks)} → H-edges:{len(h_block_edges)} V-edges:{len(v_block_edges)}')
    print(f'  [SVG] → {len(tiles)} tiles')

    payload = {
        'primary_axis': primary_axis,
        'ws_blocks': [list(b) for b in ws_blocks],
        'h_block_edges': h_block_edges, 'v_block_edges': v_block_edges,
        'h_ws_bands': h_ws_bands, 'v_ws_bands': v_ws_bands,
        'h_all': h_all, 'v_all': v_all, 'h_clean': clean_h, 'v_clean': clean_v,
        'x_cuts': x_cuts,
        'y_cuts': y_cuts,
        'strip_y_cuts': strip_y_cuts,
        'strip_x_cuts': strip_x_cuts,
    }
    return tiles, payload


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — LLM mode helpers
# ─────────────────────────────────────────────────────────────────────────────

LLM_CUT_SCHEMA = {
    "type": "object",
    "properties": {
        "x_cuts_pct": {
            "type": "array",
            "items": {"type": "number"},
            "description": "List of vertical cut positions (% of image width). Empty array if none."
        },
        "y_cuts_pct": {
            "type": "array",
            "items": {"type": "number"},
            "description": "List of horizontal cut positions (% of image height). Empty array if none."
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of split rationale and boundary evidence."
        },
    },
    "required": ["x_cuts_pct", "y_cuts_pct", "reasoning"],
}


def _draw_grid_overlay(img_arr: np.ndarray, step: int) -> np.ndarray:
    """Draw a pixel coordinate grid on the image for LLM coordinate reference."""
    vis = img_arr.copy()
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (120, 120, 120)
    for x in range(0, w, step):
        cv2.line(vis, (x, 0), (x, h), color, 1)
        cv2.putText(vis, str(x), (x + 2, 20), font, 0.5, color, 1)
    for y in range(0, h, step):
        cv2.line(vis, (0, y), (w, y), color, 1)
        cv2.putText(vis, str(y), (2, y + 18), font, 0.5, color, 1)
    return vis


def _img_to_data_url(img_bgr: np.ndarray) -> str:
    """Encode a BGR numpy array as a PNG data URL."""
    _, buf = cv2.imencode('.png', img_bgr)
    b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    return f'data:image/png;base64,{b64}'


def _build_llm_split_prompt(img_w: int, img_h: int, min_t: int, max_t: int) -> str:
    return (
        f"You are analyzing a large CAD / engineering drawing image ({img_w}×{img_h} px).\n\n"
        "The image has a pixel-coordinate grid overlay. "
        "Grid lines are drawn every 200 px and labeled with their pixel coordinates.\n\n"
        "## Task\n"
        f"Split this image into **{min_t}–{max_t} sub-regions** for detailed downstream LLM analysis.\n"
        "Identify **natural visual boundaries** where the cut lines should be placed:\n"
        "- Wide whitespace / blank bands\n"
        "- Structural separators: thick lines, title borders, frame edges\n"
        "- Logical groupings: separate drawing panels, circuit sections, note blocks\n\n"
        "## Output format\n"
        "Return cut positions as **percentages (0–100)** of the image dimension:\n"
        "- `x_cuts_pct`: positions that create vertical slices (left↔right splits)\n"
        "- `y_cuts_pct`: positions that create horizontal slices (top↔bottom splits)\n"
        "- If only one axis needs splitting, leave the other list empty `[]`.\n"
        "- Ensure cuts are spread so no single tile exceeds ~40% of the full image.\n\n"
        "## Rules\n"
        "- Do NOT cut through dense content (circuit symbols, text blocks).\n"
        "- Prefer cuts that fall inside visible whitespace bands.\n"
        "- Each resulting tile must contain meaningful content.\n\n"
        "Return JSON only matching this schema:\n"
        + json.dumps(LLM_CUT_SCHEMA, indent=2)
    )


def llm_compute_tiles(
    img_arr_bgr: np.ndarray,
    img_w: int,
    img_h: int,
    grid_step: int,
    overlap: int,
    min_tiles: int,
    max_tiles: int,
    reasoning_effort: Optional[str],
    llm_client,
    deployment: str,
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """Call GPT to get cut positions, then convert to tile bboxes."""
    grid_img = _draw_grid_overlay(img_arr_bgr, grid_step)
    prompt = _build_llm_split_prompt(img_w, img_h, min_tiles, max_tiles)

    response = llm_client.responses.create(
        model=deployment,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",  "text": prompt},
                {"type": "input_image", "image_url": _img_to_data_url(grid_img)},
            ],
        }],
        **{"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {"temperature": 0},
    )

    raw = response.output_text.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    payload = json.loads(raw)

    x_cuts_px = [int(p / 100 * img_w) for p in sorted(float(v) for v in payload.get('x_cuts_pct', []))]
    y_cuts_px = [int(p / 100 * img_h) for p in sorted(float(v) for v in payload.get('y_cuts_pct', []))]

    tiles = [(x1, y1, x2, y2)
             for y1, y2 in cuts_to_slices(img_h, y_cuts_px, overlap)
             for x1, x2 in cuts_to_slices(img_w, x_cuts_px, overlap)]
    return tiles, payload



# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — SMART mode helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_ws_blocks(
    img_gray: np.ndarray,
    min_area_fraction: float = 0.02,
    white_threshold: float = 0.95,
) -> List[Tuple[int, int, int, int]]:
    """Detect rectangular whitespace blocks using contour detection.
    
    Args:
        img_gray: 2D float array [0,1] (1=white)
        min_area_fraction: minimum block area as fraction of total image area
        white_threshold: pixel intensity threshold for whitespace (0-1 scale)
    
    Returns:
        List of (x1, y1, x2, y2) bounding boxes for whitespace blocks,
        sorted by area (largest first).
    """
    h, w = img_gray.shape
    min_area = int(w * h * min_area_fraction)
    
    # Binarize: 255 for whitespace, 0 for content
    binary = ((img_gray > white_threshold).astype(np.uint8) * 255)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blocks = []
    for contour in contours:
        x1, y1, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area >= min_area:
            x2, y2 = x1 + bw, y1 + bh
            blocks.append((x1, y1, x2, y2, area))
    
    # Sort by area (largest first) and return without area
    blocks.sort(key=lambda b: b[4], reverse=True)
    return [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in blocks]


def _ws_blocks_to_cut_edges(
    blocks: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    edge_margin: int = 50,
) -> Tuple[List[int], List[int]]:
    """Extract horizontal and vertical cut candidate positions from whitespace block edges.
    
    Args:
        blocks: List of (x1, y1, x2, y2) whitespace blocks
        img_w, img_h: Image dimensions
        edge_margin: Ignore edges within this distance from image borders
    
    Returns:
        (h_edges, v_edges) — lists of y and x positions for potential cuts
    """
    h_edges = set()  # y positions (horizontal cuts)
    v_edges = set()  # x positions (vertical cuts)
    
    for x1, y1, x2, y2 in blocks:
        # Top and bottom edges (horizontal cuts)
        if y1 > edge_margin:
            h_edges.add(y1)
        if y2 < img_h - edge_margin:
            h_edges.add(y2)
        
        # Left and right edges (vertical cuts)
        if x1 > edge_margin:
            v_edges.add(x1)
        if x2 < img_w - edge_margin:
            v_edges.add(x2)
    
    return sorted(h_edges), sorted(v_edges)


def _detect_ws_bands(
    img_gray: np.ndarray,
    row_threshold: float = 0.99,
    min_fraction: float = 0.20,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Detect contiguous near-white bands >= min_fraction of image dimension.

    img_gray: 2D float array [0,1] (1=white)
    Returns (h_bands, v_bands) — list of (start_px, end_px) tuples.
    """
    h, w = img_gray.shape

    def _bands(profile, length):
        out, in_b, start = [], False, 0
        for i, v in enumerate(profile):
            if v >= row_threshold and not in_b:
                start, in_b = i, True
            elif v < row_threshold and in_b:
                if (i - start) / length >= min_fraction:
                    out.append((start, i - 1))
                in_b = False
        if in_b and (length - start) / length >= min_fraction:
            out.append((start, length - 1))
        return out

    row_white = np.mean(img_gray > 0.95, axis=1)
    col_white = np.mean(img_gray > 0.95, axis=0)
    return _bands(row_white, h), _bands(col_white, w)


def _is_tile_white(tile_rgb: np.ndarray, threshold: float) -> bool:
    """True if >= threshold fraction of tile pixels are near-white."""
    gray = np.mean(tile_rgb.astype(np.float32) / 255.0, axis=2)
    return float(np.mean(gray > 0.95)) >= threshold


def _smart_pick_cuts(
    length: int,
    n_cuts: int,
    ws_block_edges: List[int],
    ws_mids: List[int],
    svg_clean: List[int],
    svg_all: List[int],
    tolerance: int,
    require_evidence: bool = False,
) -> List[int]:
    """Pick n_cuts with priority: ws_block_edges > whitespace-mid > SVG(text-clean) > SVG(all) > ideal.

    ws_block_edges: cut positions from rectangular whitespace block boundaries (NEW: highest priority)
    ws_mids: cut positions from 1D whitespace band midpoints (fallback if no blocks)
    
    require_evidence=True: skip a cut position if no WS band or SVG line is nearby.
    Use this for per-strip secondary cuts so strips without natural boundaries
    are not split unnecessarily.
    """
    if n_cuts <= 0:
        return []
    ideal = [int(length * (i + 1) / (n_cuts + 1)) for i in range(n_cuts)]
    chosen = []
    for pos in ideal:
        for candidates in [ws_block_edges, ws_mids, svg_clean, svg_all]:
            near = [c for c in candidates if abs(c - pos) <= tolerance]
            if near:
                chosen.append(min(near, key=lambda c: abs(c - pos)))
                break
        else:
            if not require_evidence:
                chosen.append(pos)   # ideal fallback
            # else: no evidence → skip this cut
    return sorted(set(chosen))


def _var_slices(
    length: int,
    cuts: List[int],
    before_exp: List[int],
    after_exp: List[int],
) -> List[Tuple[int, int]]:
    """Build tile slices with per-boundary variable overlap.

    before_exp[i]: extra px tile-i extends past cut-i into tile-(i+1)
    after_exp[i]:  extra px tile-(i+1) extends before cut-i into tile-i
    """
    boundaries = [0] + sorted(cuts) + [length]
    n = len(boundaries) - 1
    slices = []
    for i in range(n):
        s = boundaries[i]
        e = boundaries[i + 1]
        if i > 0:
            s = max(0, s - after_exp[i - 1])
        if i < n - 1:
            e = min(length, e + before_exp[i])
        slices.append((s, e))
    return slices


def _draw_tiles_overlay(
    img_bgr: np.ndarray,
    tiles: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int],
    label_prefix: str = '',
    thickness: int = 3,
) -> np.ndarray:
    """Draw tile bboxes on a copy of img_bgr."""
    vis = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (x1, y1, x2, y2) in enumerate(tiles):
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label = f'{label_prefix}sub{i + 1:02d}'
        cv2.putText(vis, label, (x1 + 6, y1 + 28), font, 0.8, color, 2)
    return vis


def _llm_check_cut(
    img_bgr: np.ndarray,
    cut_pos: int,
    axis: str,
    strip_px: int,
    max_expand: int,
    llm_client,
    deployment: str,
) -> Tuple[int, int, bool, str]:
    """Show LLM a strip around the cut; ask if any panel is truncated.

    Returns (expand_before, expand_after, is_truncated, reasoning).
    """
    h, w = img_bgr.shape[:2]
    if axis == 'h':
        y1 = max(0, cut_pos - strip_px)
        y2 = min(h, cut_pos + strip_px)
        vis = img_bgr[y1:y2, :].copy()
        cy = cut_pos - y1
        cv2.line(vis, (0, cy), (w, cy), (0, 0, 255), 4)
        cv2.putText(vis, 'CUT', (8, max(cy - 10, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        x1 = max(0, cut_pos - strip_px)
        x2 = min(w, cut_pos + strip_px)
        vis = img_bgr[:, x1:x2].copy()
        cx = cut_pos - x1
        cv2.line(vis, (cx, 0), (cx, h), (0, 0, 255), 4)
        cv2.putText(vis, 'CUT', (cx + 5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    prompt = (
        "This is an electrical SLD (Single Line Diagram) drawing.\n"
        "The red CUT line marks a proposed tile-split boundary.\n\n"
        "## Task\n"
        "Determine whether this cut line slices through an **electrical panel region**.\n\n"
        "## What counts as a panel boundary (DO consider these)\n"
        "- Solid rectangular frames enclosing a panel section\n"
        "- Dashed rectangular lines separating panel sections\n"
        "- Section label boxes at the top of a panel (e.g. 'MCC-1', 'E-H-08B')\n"
        "- Corner bracket marks (┐└) that anchor panel corners\n\n"
        "## What is NOT a panel boundary (DO NOT confuse these)\n"
        "- Thin single-line electrical wiring between components\n"
        "- Bus bars (horizontal/vertical thick lines connecting multiple breakers)\n"
        "- Connection lines between circuit breakers, transformers, meters\n"
        "- Leader lines and dimension lines\n\n"
        "## Decision rule\n"
        "Set panel_truncated=true ONLY if the cut goes through a panel's enclosing frame/label,\n"
        "NOT just because electrical wiring crosses the cut line.\n\n"
        f"JSON: {{\"panel_truncated\": bool, "
        f"\"expand_before_px\": int (0~{max_expand}), "
        f"\"expand_after_px\": int (0~{max_expand}), "
        f"\"reasoning\": str}}"
    )

    _, buf = cv2.imencode('.png', vis)
    b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
    data_url = f'data:image/png;base64,{b64}'

    response = llm_client.responses.create(
        model=deployment,
        input=[{
            'role': 'user',
            'content': [
                {'type': 'input_text',  'text': prompt},
                {'type': 'input_image', 'image_url': data_url},
            ],
        }],
        temperature=0,
    )
    raw = response.output_text.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    result = json.loads(raw)

    before = min(max_expand, max(0, int(result.get('expand_before_px', 0))))
    after  = min(max_expand, max(0, int(result.get('expand_after_px',  0))))
    return before, after, bool(result.get('panel_truncated', False)), result.get('reasoning', '')


def smart_compute_tiles_full(
    img_arr_rgb: np.ndarray,
    img_arr_bgr: np.ndarray,
    img_w: int,
    img_h: int,
    sep_pos: Dict,
    text_bboxes_crop: List[Tuple[int, int, int, int]],
    temp_dir: Optional[Path],
    source_stem: str,
    # ── config (notebook-level constants become parameters with defaults) ─────
    min_tiles: int = 3,
    max_tiles: int = 6,
    ws_remove_fraction: float = 0.20,
    ws_tile_max_white: float = 0.97,
    overlap_initial_px: int = 200,
    llm_adjust_overlap: bool = True,
    overlap_strip_px: int = 350,
    overlap_max_px: int = 500,
    svg_cut_tolerance: float = 0.30,
    svg_text_margin_px: int = 20,
    llm_client=None,
    deployment: str = '',
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]], Dict]:
    """Full SMART pipeline.

    Returns
    -------
    (final_tiles, initial_tiles, payload)
      final_tiles   : kept tiles after LLM overlap adjustment + whitespace filtering
      initial_tiles : tiles before LLM adjustment (for overlay comparison)
      payload       : debug info dict
    """
    gray = np.mean(img_arr_rgb.astype(np.float32) / 255.0, axis=2)

    # ── Phase 1: Detect whitespace — blocks (2D) + bands (1D fallback) ───────
    # First detect rectangular whitespace blocks and extract their edges
    ws_blocks = _detect_ws_blocks(gray, min_area_fraction=0.02, white_threshold=0.95)
    h_block_edges, v_block_edges = _ws_blocks_to_cut_edges(ws_blocks, img_w, img_h, edge_margin=50)
    
    # Also detect 1D whitespace bands as fallback
    h_ws_bands, v_ws_bands = _detect_ws_bands(gray, min_fraction=ws_remove_fraction)
    h_ws_mids = [(s + e) // 2 for s, e in h_ws_bands]
    v_ws_mids = [(s + e) // 2 for s, e in v_ws_bands]
    
    print(f'  [SMART] WS blocks: {len(ws_blocks)} found → H-edges:{len(h_block_edges)} V-edges:{len(v_block_edges)}')
    print(f'          WS bands:  H:{len(h_ws_bands)} V:{len(v_ws_bands)}')
    if ws_blocks:
        print(f'          Top 3 WS blocks (px): {[f"({x1},{y1})-({x2},{y2})" for x1, y1, x2, y2 in ws_blocks[:3]]}')

    h_svg = sep_pos.get('h', [])
    v_svg = sep_pos.get('v', [])
    h_svg_clean = [y for y in h_svg
                   if not any(ty1 - svg_text_margin_px <= y <= ty2 + svg_text_margin_px
                              for _, ty1, _, ty2 in text_bboxes_crop)]
    v_svg_clean = [x for x in v_svg
                   if not any(tx1 - svg_text_margin_px <= x <= tx2 + svg_text_margin_px
                              for tx1, _, tx2, _ in text_bboxes_crop)]

    # ── Phase 2: Determine primary axis and cut counts ────────────────────────
    area = img_w * img_h
    total_tiles = min(max_tiles, max(min_tiles,
                                     int(np.sqrt(area / (1500 * 1500))) + min_tiles - 1))
    if img_w >= img_h:
        n_v = max(1, total_tiles // 2) if img_h >= 500 else max(1, total_tiles - 1)
        n_h = 1 if img_h >= 500 else 0
    else:
        n_h = max(1, total_tiles // 2) if img_w >= 500 else max(1, total_tiles - 1)
        n_v = 1 if img_w >= 500 else 0
    if img_w >= 2500 and img_h >= 2000:
        n_v = max(2, total_tiles // 2)
        n_h = max(1, total_tiles // 2)

    tol_h = int(img_h * svg_cut_tolerance)
    tol_v = int(img_w * svg_cut_tolerance)

    # Primary axis: vertical for wide images, horizontal for tall images.
    # Primary cuts use global WS blocks + bands + SVG (with ideal fallback).
    # Secondary cuts per strip use require_evidence=True: only cut if a local
    # WS band or SVG separator is found — avoids forcing cuts through content.
    primary_axis = 'v' if img_w >= img_h else 'h'

    if primary_axis == 'v':
        x_cuts = _smart_pick_cuts(img_w, n_v, v_block_edges, v_ws_mids, v_svg_clean, v_svg, tol_v)
        y_cuts: List[int] = []
    else:
        y_cuts = _smart_pick_cuts(img_h, n_h, h_block_edges, h_ws_mids, h_svg_clean, h_svg, tol_h)
        x_cuts: List[int] = []

    print(f'  [SMART] primary={primary_axis.upper()}')
    print(f'    SVG h:{len(h_svg)}(clean:{len(h_svg_clean)}) v:{len(v_svg)}(clean:{len(v_svg_clean)})'
          f'  text:{len(text_bboxes_crop)}')
    if primary_axis == 'v':
        print(f'    primary x_cuts={x_cuts}  (secondary y per strip, evidence-only)')
    else:
        print(f'    primary y_cuts={y_cuts}  (secondary x per strip, evidence-only)')

    # ── Phase 3: Initial tiles — adaptive per-strip secondary cuts ────────────
    if primary_axis == 'v':
        p_slices_init = cuts_to_slices(img_w, x_cuts, overlap_initial_px)
        initial_all, strip_sec_cuts = [], []
        for x1, x2 in p_slices_init:
            strip_gray = gray[:, x1:x2]
            strip_rgb = img_arr_rgb[:, x1:x2]
            # Detect WS blocks in strip
            strip_blocks = _detect_ws_blocks(strip_gray, min_area_fraction=0.02)
            strip_h_edges, _ = _ws_blocks_to_cut_edges(strip_blocks, x2-x1, img_h, edge_margin=30)
            # Fallback: 1D bands
            sh_ws_bands, _ = _detect_ws_bands(strip_gray, min_fraction=ws_remove_fraction)
            sh_ws_mids = [(s + e) // 2 for s, e in sh_ws_bands]
            
            y_c = _smart_pick_cuts(img_h, n_h, strip_h_edges, sh_ws_mids, h_svg_clean, h_svg, tol_h,
                                   require_evidence=True)
            strip_sec_cuts.append(y_c)
            for y1, y2 in cuts_to_slices(img_h, y_c, overlap_initial_px):
                initial_all.append((x1, y1, x2, y2))
            print(f'    strip x={x1}-{x2}: ws_blocks={len(strip_blocks)} h_edges={strip_h_edges} → y_cuts={y_c}')
    else:
        p_slices_init = cuts_to_slices(img_h, y_cuts, overlap_initial_px)
        initial_all, strip_sec_cuts = [], []
        for y1, y2 in p_slices_init:
            strip_gray = gray[y1:y2, :]
            strip_rgb = img_arr_rgb[y1:y2, :]
            # Detect WS blocks in strip
            strip_blocks = _detect_ws_blocks(strip_gray, min_area_fraction=0.02)
            _, strip_v_edges = _ws_blocks_to_cut_edges(strip_blocks, img_w, y2-y1, edge_margin=30)
            # Fallback: 1D bands
            _, sv_ws_bands = _detect_ws_bands(strip_gray, min_fraction=ws_remove_fraction)
            sv_ws_mids = [(s + e) // 2 for s, e in sv_ws_bands]
            
            x_c = _smart_pick_cuts(img_w, n_v, strip_v_edges, sv_ws_mids, v_svg_clean, v_svg, tol_v,
                                   require_evidence=True)
            strip_sec_cuts.append(x_c)
            for x1_t, x2_t in cuts_to_slices(img_w, x_c, overlap_initial_px):
                initial_all.append((x1_t, y1, x2_t, y2))
            print(f'    strip y={y1}-{y2}: ws_blocks={len(strip_blocks)} v_edges={strip_v_edges} → x_cuts={x_c}')

    initial_tiles = [t for t in initial_all
                     if not _is_tile_white(img_arr_rgb[t[1]:t[3], t[0]:t[2]], ws_tile_max_white)]

    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        vis_init = _draw_tiles_overlay(img_arr_bgr, initial_all, (0, 180, 255), 'init-')
        for t in initial_all:
            if t not in initial_tiles:
                x1, y1, x2, y2 = t
                cv2.rectangle(vis_init, (x1, y1), (x2, y2), (0, 0, 200), 3)
                cv2.putText(vis_init, 'WS-skip', (x1 + 6, y1 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        cv2.imwrite(str(temp_dir / f'{source_stem}_01_initial.png'), vis_init)
        print(f'  → initial overlay saved: {temp_dir.name}/{source_stem}_01_initial.png')

    # ── Phase 4: LLM overlap check — PRIMARY cuts only ───────────────────────
    # Secondary cuts per strip use fixed overlap_initial_px (no LLM needed).
    p_before = [overlap_initial_px] * len(x_cuts if primary_axis == 'v' else y_cuts)
    p_after  = [overlap_initial_px] * len(x_cuts if primary_axis == 'v' else y_cuts)
    primary_cuts = x_cuts if primary_axis == 'v' else y_cuts
    llm_axis     = 'v' if primary_axis == 'v' else 'h'

    if llm_adjust_overlap and llm_client is not None and primary_cuts:
        print(f'  [SMART/LLM] {len(primary_cuts)} primary {llm_axis.upper()}-cuts → '
              f'{len(primary_cuts)} LLM calls, strip±{overlap_strip_px}px each')
        t0 = time.time()
        for i, cp in enumerate(primary_cuts):
            bef, aft, trunc, reason = _llm_check_cut(
                img_arr_bgr, cp, llm_axis, overlap_strip_px, overlap_max_px,
                llm_client, deployment)
            p_before[i] = max(overlap_initial_px, bef)
            p_after[i]  = max(overlap_initial_px, aft)
            print(f'    {llm_axis}-cut {cp}: truncated={trunc} expand=({bef},{aft})px — {reason[:70]}')
        print(f'  LLM overlap check done ({time.time() - t0:.1f}s)')

    # ── Phase 5: Final tiles with adjusted primary overlap ────────────────────
    p_slices_fin = _var_slices(
        img_w if primary_axis == 'v' else img_h,
        primary_cuts, p_before, p_after,
    )
    final_all = []
    for i, (p1, p2) in enumerate(p_slices_fin):
        sec_cuts = strip_sec_cuts[i]
        if primary_axis == 'v':
            for y1, y2 in cuts_to_slices(img_h, sec_cuts, overlap_initial_px):
                final_all.append((p1, y1, p2, y2))
        else:
            for x1, x2 in cuts_to_slices(img_w, sec_cuts, overlap_initial_px):
                final_all.append((x1, p1, x2, p2))

    final_tiles = [t for t in final_all
                   if not _is_tile_white(img_arr_rgb[t[1]:t[3], t[0]:t[2]], ws_tile_max_white)]
    discarded_final = [t for t in final_all if t not in final_tiles]
    print(f'  Final: {len(final_all)} tiles → {len(discarded_final)} WS-discarded → {len(final_tiles)} kept')

    if temp_dir is not None:
        vis_fin = _draw_tiles_overlay(img_arr_bgr, final_all, (0, 200, 0), 'fin-')
        for t in discarded_final:
            x1, y1, x2, y2 = t
            cv2.rectangle(vis_fin, (x1, y1), (x2, y2), (0, 0, 180), 3)
            cv2.putText(vis_fin, 'WS-skip', (x1 + 6, y1 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2)
        cv2.imwrite(str(temp_dir / f'{source_stem}_02_final.png'), vis_fin)
        print(f'  → final overlay saved:   {temp_dir.name}/{source_stem}_02_final.png')

    payload = {
        'primary_axis': primary_axis,
        'ws_blocks': [list(b) for b in ws_blocks],
        'h_block_edges': h_block_edges, 'v_block_edges': v_block_edges,
        'h_ws_bands': h_ws_bands, 'v_ws_bands': v_ws_bands,
        'h_svg': h_svg, 'v_svg': v_svg,
        'x_cuts': x_cuts,
        'y_cuts': y_cuts,
        'strip_sec_cuts': strip_sec_cuts,
        'primary_expand_before': p_before,
        'primary_expand_after':  p_after,
        'discarded_whitespace': [list(t) for t in discarded_final],
    }
    return final_tiles, initial_tiles, payload


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Panel-name extraction tiling
# ─────────────────────────────────────────────────────────────────────────────

def _best_cut_counts(
    img_w: int,
    img_h: int,
    target_ratios: Tuple[float, float] = (3 / 4, 4 / 3),
    max_splits: int = 8,
    min_nv: int = 0,
    min_nh: int = 0,
) -> Tuple[int, int]:
    """Choose (n_v_cuts, n_h_cuts) so tiles approximate a target ratio.

    Only considers (nv, nh) combinations where nv >= min_nv and nh >= min_nh,
    enforcing a minimum number of cuts (e.g. to keep tile size under a threshold).
    Among valid combinations, picks the one whose tile ratio is closest to any
    value in target_ratios.

    target_ratios: (portrait_ratio, landscape_ratio) e.g. (0.75, 1.33)
    """
    import math
    best_nv, best_nh, best_err = max(1, min_nv), min_nh, float("inf")
    for nv in range(min_nv, max_splits + 1):
        for nh in range(min_nh, max_splits + 1):
            if nv == 0 and nh == 0:
                continue
            tw = img_w / (nv + 1)
            th = img_h / (nh + 1)
            ratio = tw / max(th, 1)
            err = min(abs(ratio - r) for r in target_ratios)
            if err < best_err:
                best_err, best_nv, best_nh = err, nv, nh
    return best_nv, best_nh


def trim_tile_whitespace(
    tile_img: np.ndarray,
    white_threshold: float = 0.97,
    min_content_frac: float = 0.03,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Trim near-white borders from a tile image.

    Returns:
        (trimmed_image, (left, top, right, bottom)) — offsets from original tile corners.
        If the whole tile appears white, returns the original tile unchanged.
    """
    gray = np.mean(tile_img.astype(np.float32) / 255.0, axis=2)
    h, w = gray.shape

    row_content = np.mean(gray < white_threshold, axis=1)  # fraction of non-white per row
    col_content = np.mean(gray < white_threshold, axis=0)

    rows_with_content = np.where(row_content >= min_content_frac)[0]
    cols_with_content = np.where(col_content >= min_content_frac)[0]

    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        return tile_img, (0, 0, w, h)

    top    = int(rows_with_content[0])
    bottom = int(rows_with_content[-1]) + 1
    left   = int(cols_with_content[0])
    right  = int(cols_with_content[-1]) + 1

    trimmed = tile_img[top:bottom, left:right]
    return trimmed, (left, top, right, bottom)


def tile_page(
    img_arr_rgb: np.ndarray,
    pdf_path: Path,
    page_num: int,
    overlap_px: int = 200,
    split_threshold: int = 1000,
    target_ratios: Tuple[float, float] = (3 / 4, 4 / 3),
    ws_white_threshold: float = 0.95,
    text_margin_px: int = 30,
    ws_tile_max_white: float = 0.97,
    full_page_w: Optional[int] = None,
    full_page_h: Optional[int] = None,
    crop_bbox_in_page: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """Compute tile bboxes for a page crop for panel-name extraction.

    Rules:
    - Only split an axis if that dimension >= split_threshold (default 1000 px).
    - When splitting, use overlap_px on each side of the cut.
    - Cut positions are chosen by priority:
        ws_block_edges > whitespace_band_mids > svg_clean > svg_all > ideal centre.
    - Cutting through text is avoided via svg text-margin filtering.
    - Tiles that are almost entirely white are dropped.
    - Targets tile aspect ratio closest to target_ratios (portrait 3:4 or landscape 4:3).

    Args:
        img_arr_rgb       : RGB image of the crop (may be a sub-crop of the full page)
        pdf_path          : path to the source PDF (for SVG separator extraction)
        page_num          : 1-based PDF page number
        full_page_w/h     : full page PNG dimensions — required when img is a sub-crop
                            so SVG coordinates can be correctly mapped into crop space
        crop_bbox_in_page : (x1,y1,x2,y2) of this crop in full-page px;
                            if None, the image is assumed to be the entire page

    Returns:
        tiles  : list of (x1, y1, x2, y2) in image-local px
        payload: diagnostic dict (cut positions, whitespace info, etc.)
    """
    try:
        from svg_utils import detect_separator_lines, separator_positions_in_crop
    except ImportError:
        from src.svg_utils import detect_separator_lines, separator_positions_in_crop

    img_arr_bgr = cv2.cvtColor(img_arr_rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w = img_arr_rgb.shape[:2]

    # ── 1. Decide how many cuts per axis ─────────────────────────────────────
    # Minimum cuts needed so each tile side < split_threshold
    # e.g. 3414px / 1000px threshold → need at least ceil(3414/1000)-1 = 3 cuts
    import math
    min_n_v = max(0, math.ceil(img_w / split_threshold) - 1) if img_w >= split_threshold else 0
    min_n_h = max(0, math.ceil(img_h / split_threshold) - 1) if img_h >= split_threshold else 0

    if min_n_v == 0 and min_n_h == 0:
        # Image is small enough on both axes — return as a single tile
        return [(0, 0, img_w, img_h)], {'n_v': 0, 'n_h': 0, 'note': 'too small to split'}

    # Upper bound: explore up to the larger of the two minimums per axis
    # (allows balancing w vs h cuts for better ratio, without adding excess tiles)
    max_splits = max(min_n_v, min_n_h)
    n_v, n_h = _best_cut_counts(
        img_w, img_h, target_ratios,
        max_splits=max_splits,
        min_nv=min_n_v,
        min_nh=min_n_h,
    )
    if img_w < split_threshold:
        n_v = 0
    if img_h < split_threshold:
        n_h = 0

    print(f"  tile_page: {img_w}×{img_h} → n_v={n_v}, n_h={n_h}")

    # ── 2. SVG separator positions ────────────────────────────────────────────
    svg_h: List[int] = []
    svg_v: List[int] = []
    try:
        sep = detect_separator_lines(pdf_path, page_num)
        if crop_bbox_in_page is not None:
            # img is a sub-crop: use actual page dimensions for the coordinate mapping
            _fpw = full_page_w if full_page_w is not None else img_w
            _fph = full_page_h if full_page_h is not None else img_h
            svg_pos = separator_positions_in_crop(sep, _fpw, _fph, crop_bbox_in_page)
        else:
            svg_pos = separator_positions_in_crop(sep, img_w, img_h, (0, 0, img_w, img_h))
        svg_h = svg_pos.get('h', [])
        svg_v = svg_pos.get('v', [])
    except Exception as e:
        print(f"  [WARN] SVG separator detection failed: {e}")

    # ── 3. Whitespace bands & blocks ─────────────────────────────────────────
    gray = np.mean(img_arr_rgb.astype(np.float32) / 255.0, axis=2)
    ws_blocks = _detect_ws_blocks(gray, min_area_fraction=0.01, white_threshold=ws_white_threshold)
    h_block_edges, v_block_edges = _ws_blocks_to_cut_edges(ws_blocks, img_w, img_h, edge_margin=50)
    h_ws_bands, v_ws_bands = _detect_ws_bands(gray, min_fraction=0.15)
    h_ws_mids = [(s + e) // 2 for s, e in h_ws_bands]
    v_ws_mids = [(s + e) // 2 for s, e in v_ws_bands]

    # Text-safe SVG positions (exclude positions that fall inside text bboxes)
    # For now we use all SVG positions; a future enhancement can integrate PDF text coords.
    svg_h_clean = svg_h
    svg_v_clean = svg_v

    tol_h = max(50, int(img_h * 0.08))
    tol_v = max(50, int(img_w * 0.08))

    # ── 4. Pick cut positions ─────────────────────────────────────────────────
    x_cuts = _smart_pick_cuts(img_w, n_v, v_block_edges, v_ws_mids, svg_v_clean, svg_v, tol_v)
    y_cuts = _smart_pick_cuts(img_h, n_h, h_block_edges, h_ws_mids, svg_h_clean, svg_h, tol_h)

    print(f"  x_cuts={x_cuts}  y_cuts={y_cuts}")

    # ── 5. Build slices with overlap ─────────────────────────────────────────
    x_slices = cuts_to_slices(img_w, x_cuts, overlap_px)
    y_slices = cuts_to_slices(img_h, y_cuts, overlap_px)

    # ── 6. Build tile list, discard near-white tiles ─────────────────────────
    tiles: List[Tuple[int, int, int, int]] = []
    for y1, y2 in y_slices:
        for x1, x2 in x_slices:
            if not _is_tile_white(img_arr_rgb[y1:y2, x1:x2], ws_tile_max_white):
                tiles.append((x1, y1, x2, y2))

    print(f"  → {len(tiles)} tiles (after dropping all-white)")

    payload = {
        'n_v': n_v, 'n_h': n_h,
        'x_cuts': x_cuts, 'y_cuts': y_cuts,
        'x_slices': x_slices, 'y_slices': y_slices,
        'v_block_edges': v_block_edges, 'h_block_edges': h_block_edges,
        'v_ws_mids': v_ws_mids, 'h_ws_mids': h_ws_mids,
        'svg_v': svg_v, 'svg_h': svg_h,
    }
    return tiles, payload


# ── High-level orchestrators ──────────────────────────────────────────────────

import re as _re_sub
import cv2 as _cv2_sub


def find_page_png(page_num: int, pages_dir: "Path"):
    """Return the full-page PNG path for page_num, or None if not found."""
    nested = pages_dir / f"page{page_num}" / f"page{page_num}.png"
    if nested.exists():
        return nested
    flat = pages_dir / f"page{page_num}.png"
    if flat.exists():
        return flat
    return None


def discover_crops(
    figure_crops_dir: "Path",
    pages_dir: "Path",
    test_pages=None,
) -> tuple:
    """Discover crop PNGs + load full-page images.

    Returns
    -------
    crops       : List[(page_num, crop_idx, png_path, bbox_in_page)]
    page_images : Dict[int, np.ndarray]  (BGR)
    """
    import numpy as np

    crops = []
    for page_dir in sorted(
        figure_crops_dir.iterdir(),
        key=lambda p: int(_re_sub.sub(r"\D", "", p.name) or 0),
    ):
        m = _re_sub.match(r"page(\d+)$", page_dir.name)
        if not m:
            continue
        page_num = int(m.group(1))
        if test_pages is not None and page_num not in test_pages:
            continue
        for png in sorted(page_dir.glob(f"image_{page_num}_*.png")):
            if "overlay" in png.name:
                continue
            cm = _re_sub.match(r"image_\d+_(\d+)\.png$", png.name)
            if not cm:
                continue
            crop_idx = int(cm.group(1))
            txt = png.with_suffix(".txt")
            if not txt.exists():
                print(f"  [WARN] No bbox txt for {png} — skip")
                continue
            bbox = _load_polygon_txt(txt)
            crops.append((page_num, crop_idx, png, bbox))

    page_nums = sorted({c[0] for c in crops})
    page_images = {}
    for pn in page_nums:
        pp = find_page_png(pn, pages_dir)
        if pp and pp.exists():
            page_images[pn] = _cv2_sub.imread(str(pp))
        else:
            print(f"  [WARN] Full-page PNG not found for page {pn}")

    print(f"Found {len(crops)} crops across {len(page_nums)} pages:")
    for page_num, crop_idx, png, bbox in crops:
        img = page_images.get(page_num)
        page_size = f"{img.shape[1]}×{img.shape[0]}" if img is not None else "?"
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        print(
            f"  page {page_num:2d}, crop {crop_idx}: {png.name}"
            f"  bbox={bbox}  ({bw}×{bh}px)  fullpage={page_size}"
        )
    return crops, page_images


def tile_crops(
    crops: list,
    page_images: dict,
    pdf_path: "Path",
    overlap_px: int = 200,
    split_threshold: int = 1000,
    target_ratios=(3 / 4, 4 / 3),
    ws_tile_max_white: float = 0.97,
    per_page_dpi: "Optional[Dict[int, int]]" = None,
    split_threshold_inch: "Optional[float]" = None,
) -> tuple:
    """Run tile_page() over all crops and translate to full-page coordinates.

    When *split_threshold_inch* and *per_page_dpi* are both provided, the
    pixel threshold is computed per page as ``split_threshold_inch * page_dpi``.
    Otherwise *split_threshold* (fixed pixel value) is used for all pages.

    Returns
    -------
    tiles_by_page  : Dict[int, List[Tuple[int,int,int,int]]]
    tile_payloads  : Dict[int, List[dict]]
    """
    tiles_by_page: dict = {}
    tile_payloads: dict = {}

    for page_num, crop_idx, png_path, bbox_in_page in crops:
        crop_bgr = _cv2_sub.imread(str(png_path))
        if crop_bgr is None:
            print(f"[page {page_num}, crop {crop_idx}] Cannot load {png_path} — skip")
            continue
        crop_rgb = _cv2_sub.cvtColor(crop_bgr, _cv2_sub.COLOR_BGR2RGB)
        ch, cw = crop_bgr.shape[:2]

        full_img = page_images.get(page_num)
        fp_w = full_img.shape[1] if full_img is not None else None
        fp_h = full_img.shape[0] if full_img is not None else None

        # DPI-adaptive split threshold
        if split_threshold_inch is not None and per_page_dpi is not None:
            page_dpi = per_page_dpi.get(page_num, 288)
            effective_threshold = int(split_threshold_inch * page_dpi)
        else:
            effective_threshold = split_threshold
            page_dpi = None

        print(f"\n[page {page_num}, crop {crop_idx}] crop={cw}×{ch}  bbox={bbox_in_page}"
              f"  split_threshold={effective_threshold}px"
              + (f" ({split_threshold_inch}in × {page_dpi}dpi)" if page_dpi else ""))
        local_tiles, payload = tile_page(
            crop_rgb,
            pdf_path=pdf_path,
            page_num=page_num,
            overlap_px=overlap_px,
            split_threshold=effective_threshold,
            target_ratios=target_ratios,
            ws_tile_max_white=ws_tile_max_white,
            full_page_w=fp_w,
            full_page_h=fp_h,
            crop_bbox_in_page=bbox_in_page,
        )

        cx1, cy1 = bbox_in_page[0], bbox_in_page[1]
        full_tiles = [
            (cx1 + tx1, cy1 + ty1, cx1 + tx2, cy1 + ty2)
            for tx1, ty1, tx2, ty2 in local_tiles
        ]
        tiles_by_page.setdefault(page_num, []).extend(full_tiles)
        tile_payloads.setdefault(page_num, []).append(payload)

        for i, (tx1, ty1, tx2, ty2) in enumerate(full_tiles):
            tw, th = tx2 - tx1, ty2 - ty1
            ratio = tw / th if th else float("nan")
            print(f"  tile {i+1}: [{tx1},{ty1},{tx2},{ty2}]  {tw}×{th}  ratio={ratio:.3f}")

    total = sum(len(t) for t in tiles_by_page.values())
    print(f"\nTiling complete — {total} tiles across {len(tiles_by_page)} pages")
    return tiles_by_page, tile_payloads
