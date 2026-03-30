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
    """Deduplicate candidates via LLM with full page image for verification."""
    if not candidates:
        return []

    prompt = build_name_dedup_prompt(candidates)
    raw = call_llm(
        llm_client, deployment, prompt,
        image_paths=[page_img_path],
        label="name_dedup",
    )
    result = parse_json(raw)
    if isinstance(result, list):
        return [str(n) for n in result if isinstance(n, str)] or candidates
    if isinstance(result, dict):
        return result.get("panel_names") or result.get("panel_bays") or result.get("names") or candidates
    return candidates


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
