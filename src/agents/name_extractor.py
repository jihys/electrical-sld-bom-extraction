"""Panel name extraction agent.

Extracts panel names from tiled page images using LLM,
then deduplicates and validates against DI OCR candidates.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .llm_caller import call_llm, parse_json
from .prompts import EXTRACT_NAMES_SCHEMA, DEDUP_NAMES_SCHEMA, build_name_extract_prompt, build_name_dedup_prompt
from ..tools.image_tools import ndarray_to_data_url


# ──────────────────────────────────────────────────────────────────────────────
# Whitespace trimming
# ──────────────────────────────────────────────────────────────────────────────

def _trim_whitespace(img: np.ndarray, threshold: int = 245, padding: int = 10,
                     min_pixels: int = 5) -> np.ndarray:
    """Crop out near-white margins from a tile image.

    A row/column is treated as content only if >= min_pixels pixels are non-white,
    so thin border lines (1-2px) spanning the full tile are ignored.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < threshold
    content_rows = np.where(mask.sum(axis=1) >= min_pixels)[0]
    content_cols = np.where(mask.sum(axis=0) >= min_pixels)[0]
    if len(content_rows) == 0 or len(content_cols) == 0:
        return img
    r1 = max(0, int(content_rows[0]) - padding)
    r2 = min(img.shape[0], int(content_rows[-1]) + padding + 1)
    c1 = max(0, int(content_cols[0]) - padding)
    c2 = min(img.shape[1], int(content_cols[-1]) + padding + 1)
    return img[r1:r2, c1:c2]


# ──────────────────────────────────────────────────────────────────────────────
# Tiling
# ──────────────────────────────────────────────────────────────────────────────

def tile_page(
    image_path: str,
    tile_width: int = 2000,
    overlap: int = 400,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Split a page image into overlapping vertical tiles.

    Returns [(tile_img, (x1, y1, x2, y2)), ...].
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    tiles = []
    x = 0
    while x < w:
        x2 = min(x + tile_width, w)
        tile = img[0:h, x:x2]
        tiles.append((tile, (x, 0, x2, h)))
        if x2 >= w:
            break
        x += tile_width - overlap
    return tiles


# ──────────────────────────────────────────────────────────────────────────────
# Per-tile extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_names_from_tile(
    tile_img: np.ndarray,
    llm_client,
    deployment: str,
    tile_idx: int = 0,
) -> List[str]:
    """Extract panel names from a single tile image."""
    import tempfile, os
    # Trim whitespace margins to focus on content
    tile_img = _trim_whitespace(tile_img)
    # Save tile temporarily for LLM call
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(tmp.name, tile_img)

    prompt = build_name_extract_prompt()
    raw = call_llm(
        llm_client, deployment, prompt,
        image_paths=[tmp.name],
        label=f"name_extract tile{tile_idx}",
    )
    os.unlink(tmp.name)

    result = parse_json(raw)
    print(f"  [DEBUG] tile{tile_idx} raw type={type(raw).__name__} len={len(raw) if raw else 0}")
    print(f"  [DEBUG] tile{tile_idx} parsed type={type(result).__name__} result={str(result)[:200]}")
    if not result:
        return []
    # LLM may return a list directly (e.g. ["name1", "name2"])
    if isinstance(result, list):
        names = [str(n) for n in result if isinstance(n, str)]
        print(f"  [DEBUG] tile{tile_idx} list-mode names={names[:5]}")
        return names
    if not isinstance(result, dict):
        return []
    if not result.get("extractable", True):
        print(f"  [DEBUG] tile{tile_idx} extractable=False")
        return []
    # Handle alternative key names LLM might use
    names = result.get("panel_names") or result.get("panel_bays") or result.get("names") or result.get("panels") or []
    print(f"  [DEBUG] tile{tile_idx} dict-mode names={names[:5]}")
    return names


# ──────────────────────────────────────────────────────────────────────────────
# Dedup + hallucination guard
# ──────────────────────────────────────────────────────────────────────────────

def dedup_panel_names(
    candidates: List[str],
    llm_client,
    deployment: str,
    page_img_path: str,
) -> List[str]:
    """Deduplicate candidates via LLM, then apply hallucination guard + series gap-fill."""
    if not candidates:
        return []

    # Pre-merge exact duplicates preserving order
    merged = list(dict.fromkeys(n.strip() for n in candidates if n.strip()))
    if not merged:
        return []

    prompt = build_name_dedup_prompt(merged)
    raw = call_llm(
        llm_client, deployment, prompt,
        image_paths=[page_img_path],
        label="name_dedup",
    )
    result = parse_json(raw)
    if isinstance(result, list):
        cleaned = [str(n).strip() for n in result if isinstance(n, str) and str(n).strip()]
    elif isinstance(result, dict):
        names = result.get("panel_names") or result.get("panel_bays") or result.get("names") or merged
        cleaned = [str(n).strip() for n in names if str(n).strip()]
    else:
        cleaned = merged

    # ── Hard guard: only keep names traceable to the merged candidate set ─────
    _OCR_PAIRS = frozenset(
        frozenset(pair) for pair in [
            ('0', 'O'), ('0', 'D'), ('0', 'Q'),
            ('1', 'I'), ('1', 'L'), ('1', '7'),
            ('2', 'Z'), ('5', 'S'), ('6', 'G'),
            ('8', 'B'), ('9', 'G'), ('9', 'Q'),
        ]
    )

    def _norm(s: str) -> str:
        s = re.sub(r"[\s\-]", "", s).upper()
        s = re.sub(r"(?<!\d)0+(\d)", r"\1", s)
        return s

    def _light_norm(s: str) -> str:
        return re.sub(r"[\s\-_]", "", s).upper()

    def _ocr_close(a: str, b: str) -> bool:
        if len(a) != len(b):
            return False
        diffs = [(ca, cb) for ca, cb in zip(a, b) if ca != cb]
        if len(diffs) != 1:
            return False
        return frozenset(diffs[0]) in _OCR_PAIRS

    candidate_norms  = {_norm(c) for c in merged}
    light_candidates = {_light_norm(c) for c in merged}

    allowed: List[str] = []
    ocr_fixed: List[str] = []
    blocked: List[str] = []
    for name in cleaned:
        if _norm(name) in candidate_norms:
            allowed.append(name)
        elif any(_ocr_close(_light_norm(name), lc) for lc in light_candidates):
            allowed.append(name)
            ocr_fixed.append(name)
        else:
            blocked.append(name)

    if ocr_fixed:
        print(f"  [dedup-guard] OCR-corrected {len(ocr_fixed)} name(s): {ocr_fixed[:5]}")
    if blocked:
        print(f"  [dedup-guard] Blocked {len(blocked)} hallucinated name(s): "
              f"{blocked[:10]}{'...' if len(blocked) > 10 else ''}")

    # ── Series gap-fill: replace duplicate with missing adjacent number ────────
    return _fill_series_gaps(allowed)


def _fill_series_gaps(names: List[str]) -> List[str]:
    """Replace one duplicate with the missing adjacent number when gap == 1."""
    from collections import Counter

    _SERIES_PAT = re.compile(r'^(.*?)(\d+)([^0-9]*)$')
    groups: dict = {}
    unparsed: List[str] = []
    for n in names:
        m = _SERIES_PAT.match(n)
        if m:
            pre, num_s, suf = m.groups()
            groups.setdefault((pre, suf, len(num_s)), []).append(int(num_s))
        else:
            unparsed.append(n)

    result = list(unparsed)
    for (pre, suf, zpad), nums in groups.items():
        cnt = Counter(nums)
        unique_sorted = sorted(cnt)
        out_nums = list(nums)

        for i in range(len(unique_sorted) - 1):
            a, b = unique_sorted[i], unique_sorted[i + 1]
            if b - a == 2:
                gap = a + 1
                if cnt[a] > 1:
                    cnt[a] -= 1
                    out_nums[out_nums.index(a)] = gap
                    cnt[gap] = cnt.get(gap, 0) + 1
                elif cnt[b] > 1:
                    cnt[b] -= 1
                    idx = len(out_nums) - 1 - out_nums[::-1].index(b)
                    out_nums[idx] = gap
                    cnt[gap] = cnt.get(gap, 0) + 1

        fmt = f"{{:0{zpad}d}}" if zpad > 1 else "{:d}"
        result.extend(pre + fmt.format(n) + suf for n in out_nums)

    return result


def hallucination_guard(
    names: List[str],
    di_lines: List[Dict],
) -> List[str]:
    """Keep only names that match at least one DI OCR line (fuzzy match).

    This prevents the LLM from hallucinating panel names not present in the drawing.
    """
    ocr_texts = [line["content"] for line in di_lines]
    verified = []
    for name in names:
        if _name_in_ocr(name, ocr_texts):
            verified.append(name)
        else:
            print(f"  hallucination_guard: dropped '{name}' — no OCR match")
    return verified


def _name_in_ocr(name: str, ocr_texts: List[str]) -> bool:
    """Check if a panel name appears (fuzzily) in any OCR text."""
    norm_name = _alphanum(name)
    for text in ocr_texts:
        norm_text = _alphanum(text)
        if norm_name in norm_text or norm_text in norm_name:
            return True
        if _fuzzy_match(norm_name, norm_text):
            return True
    return False


def _alphanum(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    return re.sub(r"[^a-z0-9]", "", text)


def _fuzzy_match(a: str, b: str) -> bool:
    """Simple fuzzy: same length, at most 1 char different."""
    if len(a) < 3 or len(a) != len(b):
        return False
    diff = sum(1 for ca, cb in zip(a, b) if ca != cb)
    return diff <= 1
