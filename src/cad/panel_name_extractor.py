"""
panel_name_extractor.py
------------------------
LLM-based panel name extraction from tiled SLD (Single Line Diagram) images.

Pipeline:
  1. tile_page()         — smart tiling using SVG + whitespace cues
  2. extract_from_tile() — LLM reads each tile and lists panel names (or
                           returns None when names are intentionally absent)
  3. dedup_for_page()    — LLM deduplicates candidates across overlapping tiles,
                           double-checked against the full page image
  4. save_panel_names()  — writes by_page JSON matching panel_names.json format

Output format (matches existing reference):
  {"by_page": {"1": ["HV 1", "TR 1", ...], "3": [...], ...}}
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# LLM JSON schemas
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "extractable": {
            "type": "boolean",
            "description": (
                "false if panel names are visibly blanked out, intentionally removed, "
                "or this tile contains no panel identifiers at all"
            ),
        },
        "panel_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "All panel / section identifiers found in this image. "
                "Each drawing uses its own naming convention — extract exactly what is written. "
                "Pure letter codes without numbers are valid if they label an enclosed bay."
            ),
        },
        "reasoning": {"type": "string"},
    },
    "required": ["extractable", "panel_names"],
}

_DEDUP_SCHEMA = {
    "type": "object",
    "properties": {
        "panel_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Deduplicated, canonical panel name list for this page. "
                "Merge spelling variants (e.g. missing space or hyphen between prefix and suffix). "
            ),
        },
        "reasoning": {"type": "string"},
    },
    "required": ["panel_names"],
}

_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "panel_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Complete list of all panel bay names visible on this page. "
                "Includes names from the candidate list plus any additional names "
                "you can see in the image that were missed."
            ),
        },
        "added": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names that were NOT in the candidate list but are visible in the image.",
        },
        "removed": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names from the candidate list that are NOT actually panel bay names (false positives).",
        },
        "reasoning": {"type": "string"},
    },
    "required": ["panel_names"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _img_to_data_url(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode('.png', img_bgr)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _parse_json(text: str) -> Optional[dict]:
    """Extract JSON object from LLM output, stripping markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


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


def _call_llm(
    llm_client,
    deployment: str,
    prompt: str,
    image_urls: List[str],
    label: str = "",
) -> str:
    content: List[dict] = [{"type": "input_text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "input_image", "image_url": url, "detail": "high"})
    t0 = time.time()
    tag = f" [{label}]" if label else ""
    print(f"  LLM{tag} calling...", end=" ", flush=True)
    response = llm_client.responses.create(
        model=deployment,
        input=[{"role": "user", "content": content}],
        temperature=0,
    )
    elapsed = round(time.time() - t0, 3)
    print(f"done ({elapsed:.1f}s)", flush=True)
    from .panel_utils import log_llm_call
    log_llm_call(label, elapsed, len(image_urls), reasoning_effort="none", source="panel_name_extractor")
    return response.output_text, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Auto-select reference image
# ─────────────────────────────────────────────────────────────────────────────

_SELECT_REF_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {
            "type": "integer",
            "description": "1-based index of the reference image that best matches the SLD page style.",
        },
        "reasoning": {"type": "string"},
    },
    "required": ["choice"],
}


def select_best_reference_image(
    sample_page_img: np.ndarray,
    candidate_paths: List[Path],
    llm_client,
    deployment: str,
    category: str = "panel_name",
) -> Optional[Path]:
    """Use LLM to pick the best-matching reference image for the given SLD page.

    Args:
        sample_page_img: A sample page image from the uploaded PDF.
        candidate_paths: List of reference image file paths to choose from.
        llm_client: Azure OpenAI client.
        deployment: Model deployment name.
        category: Description of what the reference is for (for the prompt).

    Returns:
        The Path of the best-matching reference, or None on failure.
    """
    if not candidate_paths:
        return None
    if len(candidate_paths) == 1:
        return candidate_paths[0]

    # Build image list: page first, then candidates
    image_urls = [_img_to_data_url(sample_page_img)]
    for cp in candidate_paths:
        img = cv2.imread(str(cp))
        if img is not None:
            image_urls.append(_img_to_data_url(img))

    ref_labels = "\n".join(
        f"  Image {i+2}: {cp.name}" for i, cp in enumerate(candidate_paths)
    )

    prompt = (
        f"You are selecting the best visual reference image for {category} extraction "
        "from an electrical Single Line Diagram (SLD).\n\n"
        "## Image 1: Sample page from the uploaded SLD document\n"
        "This is one page of the document being processed.\n\n"
        "## Reference image candidates:\n"
        f"{ref_labels}\n\n"
        "## Task\n"
        "Compare the visual style of the SLD page (Image 1) with each reference candidate.\n"
        "Pick the reference image whose panel label style (font, box shape, naming convention) "
        "most closely matches the uploaded document.\n\n"
        "Return JSON:\n" + json.dumps(_SELECT_REF_SCHEMA, indent=2)
    )

    raw, elapsed = _call_llm(llm_client, deployment, prompt, image_urls, label="select-ref")
    result = _parse_json(raw)
    if result is None:
        return candidate_paths[0]

    choice = result.get("choice", 1)
    idx = max(0, min(choice - 1, len(candidate_paths) - 1))
    reasoning = result.get("reasoning", "")
    print(f"  [select-ref] Chose {candidate_paths[idx].name}: {reasoning[:120]}")
    return candidate_paths[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_panel_names_from_tile(
    tile_img: np.ndarray,
    llm_client,
    deployment: str,
    example_img_b64: Optional[str] = None,
    example_imgs_b64: Optional[List[str]] = None,
) -> Tuple[Optional[List[str]], float]:
    """Ask LLM to list panel bay labels visible in a single tile image.

    Returns:
        (list[str], elapsed_s) — panel names found + LLM call time
        (None, elapsed_s)      — tile has no panel section boxes
    """
    # Normalise: support both single image (legacy) and list
    _refs: List[str] = []
    if example_imgs_b64:
        _refs = list(example_imgs_b64)
    elif example_img_b64:
        _refs = [example_img_b64]
    n_refs = len(_refs)
    tile_img_idx = n_refs + 1  # 1-based index for the tile image

    example_preamble = ""
    if _refs:
        ref_range = "Image 1" if n_refs == 1 else f"Image 1..{n_refs}"
        example_preamble = (
            f"## Reference images ({ref_range})\n"
            f"{ref_range} {'is a sample crop' if n_refs == 1 else 'are sample crops'} from the SAME document you are analyzing.\n"
            f"{'It shows' if n_refs == 1 else 'They show'} examples of PANEL NAMES from this drawing with annotations.\n"
            f"Study {ref_range} to learn:\n"
            "  • The exact visual style of panel name label boxes in this document\n"
            "    (shape, border style, position, font, 1-line vs 2-line format)\n"
            "  • Which boxes are panel name labels (blue outlines)\n"
            "  • How to combine 2-line labels into one string\n"
            "  • RED X = label box cut off at the edge → omit this name\n"
            "  • RED TEXT = additional rules specific to this drawing\n"
            f"Use the panel name examples in {ref_range} as the reference when\n"
            f"extracting panel names from Image {tile_img_idx} (the actual tile).\n\n"
        )

    step1 = (
        "## Step 1 — Identify the label pattern\n"
        + (f"{ref_range} {'shows' if n_refs == 1 else 'show'} examples of PANEL NAMES from this document.\n"
           f"Use those examples as your reference: find labels in Image {tile_img_idx} that match the same\n"
           f"shape, border, position, and format as the panel name examples in {ref_range}.\n\n"
           if _refs else
           "Find the repeating bordered boxes (rectangle or hexagon) that name each bay.\n"
           "Note shape, position, and line count. Apply the pattern consistently to all bays.\n\n"
           )
    )

    prompt = (
        ("You are analyzing a cropped section of an electrical Single Line Diagram (SLD).\n\n"
         + example_preamble)
        if _refs else
        "You are analyzing a cropped section of an electrical Single Line Diagram (SLD).\n\n"
    ) + (
        "## Task\n"
        "Extract the short alphanumeric codes that NAME each enclosed panel bay in this SLD crop.\n"
        "Each drawing has its own naming convention — read exactly what is written.\n\n"
        "## Most important principle\n"
        "Panel name labels share a CONSISTENT visual pattern throughout the drawing:\n"
        "same shape (rectangle or hexagon), same border style, same position relative to bay\n"
        "boundaries. Identify that repeating pattern first, then extract every instance of it.\n\n"
        + step1
        + "## What counts as a panel bay name\n"
        "A panel bay name identifies a ZONE or SECTION of the switchboard — a bounded area\n"
        "that encloses multiple electrical components (breakers, busbars, CTs, etc.) together.\n"
        "The label box may sit ON or JUST OUTSIDE the boundary line of that zone — it does not\n"
        "need to be inside the bounded area. Look for small labeled boxes (rectangle or hexagon)\n"
        "at the corners or edges of dashed/solid boundary lines.\n"
        "Panel names often follow a pattern combining voltage class, floor, and bay identifier\n"
        "(e.g. 'LV 4F-1': low-voltage panel, 4th floor, bay 1 / 'HV 5F-2A': high-voltage,\n"
        "5th floor, bay 2A / 'TR 5F': transformer panel, 5th floor). Not all drawings use\n"
        "this convention, but such prefixes are part of the panel name and must be included.\n\n"
        "## Step 2 — Extract\n"
        "• Panel bay names are often written across TWO LINES inside the label box.\n"
        "  Combine the top line and bottom line into a single string (with space as separator).\n"
        "  Do NOT output each line separately.\n"
        "  Example: top='LV', bottom='4F-CON' → output 'LV 4F-CON'\n"
        "• Slashes in codes (e.g. 'X/B'): transcribe exactly — do not drop the slash.\n"
        "• Pure letter codes are valid if they label an enclosed zone.\n"
        "• If the same label pattern repeats for multiple bays (e.g. RF 2F-1A, RF 2F-2A, RF 3F-1A …),\n"
        "  list every distinct name — do not collapse repeating patterns into one.\n\n"
        "## Exclude\n"
        "- Room/area names, title block text, plain English phrases with no alphanumeric code\n"
        "- Labels accompanied by 'TO:' text — these are feeder/circuit destination identifiers,\n"
        "  not panel bay names (e.g. a box showing 'MCCE-1-F1' next to 'TO: ATS-1-F1' is a\n"
        "  feeder label, not a bay name)\n"
        "- Cross-reference annotations: text such as 'FROM C.S U-4.2 SUBSTATION (E-H-01B)'\n"
        "  or 'TO: PANEL-X' mentions a panel from ANOTHER part of the drawing. These are\n"
        "  cable routing references, NOT panel bay names for THIS section. Only extract names\n"
        "  that serve as the section/zone label for an enclosed bay area in this tile.\n"
        "- Individual equipment/component type codes on device symbols (breakers, transformers,\n"
        "  current/voltage sensors, protection relays, switches, etc.) — these name a device,\n"
        "  not a bay area. Note: 'UPS', 'BAT' combined with a floor/number identifier\n"
        "  (e.g. 'UPS 3F-2', 'BAT 3F-5') are panel bay names — include them.\n"
        "- Product brand/model numbers, electrical specs (kV, kA, MVA), revision notes\n\n"
        "## Cut-off rule\n"
        "If a label box touches or crosses ANY edge of this image → omit it.\n"
        "Only include names whose box is fully visible. Other tiles will capture cut-off labels.\n\n"
        "## Extractability\n"
        "`extractable: false` only if this tile has NO bay section boxes at all or all labels\n"
        "are blanked out. Bays present but no labels → `extractable: true`, `panel_names: []`.\n\n"
        "Return JSON only:\n" + json.dumps(_EXTRACT_SCHEMA, indent=2)
    )
    image_urls: List[str] = []
    for _rb64 in _refs:
        image_urls.append(f"data:image/png;base64,{_rb64}")
    image_urls.append(_img_to_data_url(tile_img))
    raw, elapsed = _call_llm(llm_client, deployment, prompt, image_urls, label="extract")
    result = _parse_json(raw)
    if result is None:
        return [], elapsed
    if not result.get("extractable", True):
        return None, elapsed
    names = result.get("panel_names", [])
    return [str(n).strip() for n in names if str(n).strip()], elapsed


def dedup_panel_names_for_page(
    candidate_names: List[str],
    full_page_img: np.ndarray,
    llm_client,
    deployment: str,
) -> Tuple[List[str], float]:
    """Clean and deduplicate tile candidate names using LLM (no image).

    Returns (names, elapsed_s).
    candidate_names : all raw names from tile extractions (may include duplicates)
    full_page_img   : unused (kept for API compatibility)
    """
    merged = list(dict.fromkeys(n.strip() for n in candidate_names if n.strip()))
    if not merged:
        return [], 0.0

    candidates_json = json.dumps(merged, indent=2)
    prompt = (
        "You are reviewing electrical panel bay label extraction for one page of an SLD.\n\n"
        "## Candidate panel bay labels (from tile crops of this page)\n"
        f"{candidates_json}\n\n"
        "## Your task: CLEAN and DEDUPLICATE only\n"
        "All output names must appear in the candidates list above (possibly in variant spelling).\n"
        "Adding new names that are not in the candidates list is not needed.\n\n"
        "### Cleaning rules\n"
        "1. Keep ALL candidates. Do not remove any name.\n"
        "2. Fix typos and normalize formatting within candidates:\n"
        "   - Add missing hyphens: 'EHV8' → 'E-HV-08', 'EHV2A' → 'E-HV-02A'\n"
        "   - Normalize zero-padding to the dominant format in each series\n"
        "   - OCR corrections: 'E-HV-0B' → 'E-HV-08' (B≈8), 'E-HV-0l' → 'E-HV-01' (l≈1)\n"
        "   - Merge OCR variants of the same label into one canonical form\n"
        "   Keep BOTH if they could be genuinely distinct panels.\n"
        "2b. Fix duplicate-in-series gaps: replace ONE duplicate with the missing adjacent\n"
        "   number when gap == 1.  Example: [E-HV-08, E-HV-08, E-HV-10] → E-HV-09 replaces\n"
        "   one E-HV-08.  Replaces a duplicate — does not increase total count.\n"
        "Return JSON only:\n" + json.dumps(_DEDUP_SCHEMA, indent=2)
    )
    raw, elapsed = _call_llm(llm_client, deployment, prompt, [], label="dedup")
    result = _parse_json(raw)
    if result is None:
        return merged, elapsed
    names = result.get("panel_names", [])
    cleaned = [str(n).strip() for n in names if str(n).strip()]

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

    candidate_norms  = {_norm(c)       for c in merged}
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
              f"{blocked[:10]}{'…' if len(blocked) > 10 else ''}")

    # ── Series gap-fill: replace duplicate with missing adjacent number ────────
    _SERIES_PAT = re.compile(r'^(.*?)(\d+)([^0-9]*)$')

    def _fill_series_gaps(names: List[str]) -> List[str]:
        from collections import Counter as _Ctr
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
            cnt = _Ctr(nums)
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

    return _fill_series_gaps(allowed), elapsed


def verify_panel_names_with_full_page(
    candidate_names: List[str],
    full_page_img: np.ndarray,
    llm_client,
    deployment: str,
    example_img_b64: Optional[str] = None,
    example_imgs_b64: Optional[List[str]] = None,
) -> Tuple[List[str], float]:
    """Verify and complete panel names using the full-page image.

    Sends the full page image to LLM along with the candidate list from tiled
    extraction.  The LLM checks for missed names and removes false positives.

    Returns (final_names, elapsed_s).
    """
    if not candidate_names:
        return [], 0.0

    # Normalise: support both single image (legacy) and list
    _refs: List[str] = []
    if example_imgs_b64:
        _refs = list(example_imgs_b64)
    elif example_img_b64:
        _refs = [example_img_b64]
    n_refs = len(_refs)
    page_img_idx = n_refs + 1

    candidates_json = json.dumps(candidate_names, indent=2)

    example_preamble = ""
    if _refs:
        ref_range = "Image 1" if n_refs == 1 else f"Image 1..{n_refs}"
        example_preamble = (
            f"## Reference images ({ref_range})\n"
            f"{ref_range} {'shows' if n_refs == 1 else 'show'} examples of panel name label boxes from this document.\n"
            "BLUE outlines = panel name labels.  RED X = cut-off labels to ignore.\n"
            f"Use {ref_range} to learn the visual style, then apply it to Image {page_img_idx}.\n\n"
        )

    prompt = (
        "You are verifying panel bay name extraction for one page of an electrical SLD.\n\n"
        + example_preamble
        + f"## Image {page_img_idx}: Full page image\n"
        "The attached image is the FULL PAGE of the SLD drawing.\n\n"
        "## Candidate panel names (from tiled extraction)\n"
        f"{candidates_json}\n\n"
        "## Your task\n"
        "1. Scan the ENTIRE page image carefully from left edge to right edge.\n"
        "   Find ALL panel bay name labels — short alphanumeric codes in bordered boxes\n"
        "   (rectangles or hexagons) that identify sections/zones of the switchboard.\n"
        "   Pay special attention to SMALLER or AUXILIARY sections near page edges\n"
        "   (e.g., distribution boards, transformer bays, NGR sections) — these are\n"
        "   easily missed but are valid panel bays.\n"
        "2. Compare what you see with the candidate list above.\n"
        "3. ADD any panel names visible in the image that are missing from the candidates.\n"
        "4. REMOVE any candidate names that are NOT actually panel bay names:\n"
        "   - Equipment labels, component codes, hallucinated names\n"
        "   - Cross-reference annotations: text like 'FROM C.S U-4.2 SUBSTATION (E-H-01B)'\n"
        "     or 'TO: PANEL-X' that references a panel from ANOTHER page/section.\n"
        "     These mention a panel name but the panel itself is NOT on this page.\n"
        "     A valid panel name must label a bounded area on THIS page containing\n"
        "     electrical components (breakers, busbars, CTs, etc.).\n\n"
        "## Rules\n"
        "- Panel names with UPPER/LOWER suffixes: if you see 'E-H-02B' with '(UPPER)' annotation,\n"
        "  output as 'E-H-02B (UPPER)'.  Same for (LOWER).\n"
        "- 2-line labels: combine into single string (e.g. top='NGR' bottom='E-TR-01B' → 'NGR-E-TR-01B',\n"
        "  or top='E-TR' bottom='01B' → 'E-TR-01B').\n"
        "- Do NOT include equipment/device labels, feeder destinations ('TO:' items), or specs.\n"
        "- Include every distinct panel name — do not collapse series.\n"
        "- Scan the full page width — panels at the far right edge of the page are often missed.\n\n"
        "Return JSON only:\n" + json.dumps(_VERIFY_SCHEMA, indent=2)
    )

    image_urls: List[str] = []
    for _rb64 in _refs:
        image_urls.append(f"data:image/png;base64,{_rb64}")
    image_urls.append(_img_to_data_url(full_page_img))

    raw, elapsed = _call_llm(llm_client, deployment, prompt, image_urls, label="verify-full-page")
    result = _parse_json(raw)
    if result is None:
        return candidate_names, elapsed

    names = result.get("panel_names", [])
    added = result.get("added", [])
    removed = result.get("removed", [])
    if added:
        print(f"  [verify] Added {len(added)} name(s): {added}")
    if removed:
        print(f"  [verify] Removed {len(removed)} name(s): {removed}")

    cleaned = [str(n).strip() for n in names if str(n).strip()]
    return cleaned, elapsed


def extract_panel_names_for_pages(
    pages: List[Tuple[int, Path]],
    tiles_by_page: Dict[int, List[Tuple[int, int, int, int]]],
    page_images: Dict[int, np.ndarray],
    llm_client,
    deployment: str,
    output_dir: Optional[Path] = None,
    save_tiles: bool = False,
    example_img_b64: Optional[str] = None,
    example_imgs_b64: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Full extraction pipeline: tiles → LLM extract → dedup → by_page dict.

    Args:
        pages           : list of (page_num, png_path)
        tiles_by_page   : {page_num: [(x1,y1,x2,y2), ...]}
        page_images     : {page_num: BGR ndarray of full page}
        llm_client      : AzureOpenAI client
        deployment      : model deployment name
        output_dir      : optional folder to save intermediate tile images + JSON
        save_tiles      : save cropped tile PNGs if output_dir is given
        example_img_b64 : optional base64-encoded PNG (legacy single image).
        example_imgs_b64: optional list of base64-encoded PNGs (multiple references).

    Returns:
        {"by_page": {"1": [...], "3": [...], ...}}  (skipped pages omitted)
    """
    by_page: Dict[str, List[str]] = {}
    timing_records: List[dict] = []

    for page_num, png_path in pages:
        tiles = tiles_by_page.get(page_num, [])
        full_img = page_images.get(page_num)
        if full_img is None:
            print(f"[page {page_num}] full image not loaded — skip")
            continue

        print(f"\n{'='*60}")
        print(f"  Page {page_num} — {len(tiles)} tiles")
        print(f"{'='*60}")

        page_dir = (output_dir / f"page{page_num}") if output_dir else None
        if page_dir:
            page_dir.mkdir(parents=True, exist_ok=True)

        candidates: List[str] = []
        tiles_with_panels = 0   # tiles that had panel sections (extractable=True)

        for t_idx, (x1, y1, x2, y2) in enumerate(tiles):
            tile_img = full_img[y1:y2, x1:x2]
            tile_img = _trim_whitespace(tile_img)
            print(f"  tile {t_idx+1}/{len(tiles)}: [{x1},{y1},{x2},{y2}] "
                  f"({tile_img.shape[1]}×{tile_img.shape[0]}px after trim)")

            if save_tiles and page_dir is not None:
                tile_path = page_dir / f"tile_{t_idx+1:02d}.png"
                cv2.imwrite(str(tile_path), tile_img)

            result, tile_elapsed = extract_panel_names_from_tile(
                tile_img, llm_client, deployment,
                example_img_b64=example_img_b64,
                example_imgs_b64=example_imgs_b64,
            )
            timing_records.append({
                "page_num": page_num, "tile_idx": t_idx + 1,
                "call_type": "extract", "elapsed_s": tile_elapsed,
            })

            if result is None:
                # This tile has no panel section boxes — skip this tile, continue others
                print(f"    → extractable=False (no panel bays in this tile — skipping tile)")
                continue
            else:
                tiles_with_panels += 1
                print(f"    → {len(result)} names: {result}")
                candidates.extend(result)

        if tiles_with_panels == 0:
            print(f"  [page {page_num}] no panel section tiles found — skip")
            continue

        if not candidates:
            print(f"  [page {page_num}] no panel names found in any tile — skip")
            continue

        print(f"\n  Deduplicating {len(candidates)} candidates (incl. overlap duplicates)…")
        final_names, dedup_elapsed = dedup_panel_names_for_page(candidates, full_img, llm_client, deployment)
        timing_records.append({
            "page_num": page_num, "tile_idx": None,
            "call_type": "dedup", "elapsed_s": dedup_elapsed,
        })
        print(f"  → Final: {final_names}")

        by_page[str(page_num)] = final_names

        # Save per-page intermediate JSON
        if page_dir is not None:
            (page_dir / "panel_names_candidates.json").write_text(
                json.dumps({"candidates": candidates, "final": final_names}, ensure_ascii=False, indent=2)
            )

    return {"by_page": by_page, "timing": timing_records}


def save_panel_names(result: dict, out_path: Path) -> None:
    """Write panel names JSON to file (creates parent dirs as needed)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


def extract_whole_image_names(
    crops: list,
    llm_client,
    deployment: str,
    example_img_b64: Optional[str] = None,
    example_imgs_b64: Optional[List[str]] = None,
    filter_pages: Optional[list] = None,
    max_workers: int = 8,
    timeout: int = 600,
) -> dict:
    """Run whole-image (no tile split) panel name extraction for each crop.

    Parameters
    ----------
    crops         : list of (page_num, crop_idx, png_path, bbox)
    llm_client    : Azure OpenAI client
    deployment    : deployment name
    example_img_b64 : optional base64-encoded example image
    filter_pages  : only process these page numbers (None = all)
    max_workers   : thread pool size
    timeout       : per-future timeout in seconds

    Returns
    -------
    dict: {page_num: [name, ...]}
    """
    import cv2
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    crops_by_page: dict = defaultdict(list)
    for page_num, crop_idx, png_path, bbox in crops:
        if filter_pages is not None and page_num not in filter_pages:
            continue
        img = cv2.imread(str(png_path))
        if img is not None:
            crops_by_page[page_num].append((crop_idx, img))

    def _extract(pn, crop_idx, img):
        try:
            names, _elapsed = extract_panel_names_from_tile(
                img, llm_client, deployment,
                example_img_b64=example_img_b64,
                example_imgs_b64=example_imgs_b64,
            )
        except Exception as e:
            print(f"  [Page {pn}] crop {crop_idx} ERROR: {e}")
            return pn, []
        names = names or []
        print(f"  [Page {pn}] crop {crop_idx}: {len(names)} names")
        return pn, names

    result: dict = defaultdict(list)
    tasks = [
        (pn, crop_idx, img)
        for pn in sorted(crops_by_page)
        for crop_idx, img in sorted(crops_by_page[pn])
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract, pn, ci, img): (pn, ci)
                   for pn, ci, img in tasks}
        for future in as_completed(futures, timeout=timeout):
            pn, names = future.result()
            result[pn].extend(names)

    # LLM dedup per page (fixes typos/patterns + removes duplicates)
    deduped = {}
    for pn, names in result.items():
        print(f"\n  [Page {pn}] Deduplicating {len(names)} candidates…")
        deduped[pn], _ = dedup_panel_names_for_page(names, None, llm_client, deployment)
        print(f"  [Page {pn}] → Final: {deduped[pn]}")
    return deduped
