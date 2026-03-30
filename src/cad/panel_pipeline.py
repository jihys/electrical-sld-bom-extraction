"""panel_pipeline.py
-------------------
Core panel-area extraction pipeline.

Public API
----------
run_panel_extraction(
    client, deployment, guide_images,
    crop_items_by_page, panel_names_data, bbox_by_image,
    output_root, notebook_dir,
    grid_size=120, verify_max_tries=3,
) -> dict   # summary {'by_page': {...}}
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from openai import AzureOpenAI

from .panel_images import make_locate_all_input_image, make_locate_input_image, make_verify_all_overlay_image, make_verify_overlay_image
from .panel_prompts import (
    LOCATE_ALL_SCHEMA, LOCATE_SCHEMA, VERIFY_ALL_SCHEMA, VERIFY_SCHEMA,
    build_locate_all_prompt, build_locate_prompt, build_verify_all_prompt, build_verify_prompt,
)
from .panel_utils import call_llm, parse_json, safe_name, sanitize_bbox


# ── bbox helpers ──────────────────────────────────────────────────────────────

def apply_exclude_regions(
    panel_crop: np.ndarray,
    bbox: List[int],
    exclude_regions: List[List[int]],
) -> np.ndarray:
    """White-fill exclude_regions on the panel crop image.

    Parameters
    ----------
    panel_crop : cropped panel image (will be copied, not modified in place)
    bbox : [x1, y1, x2, y2] of the panel crop in the original image coordinates
    exclude_regions : list of [x1, y1, x2, y2] in original image coordinates

    Returns
    -------
    panel_crop with excluded areas filled with white (255, 255, 255)
    """
    if not exclude_regions:
        return panel_crop
    result = panel_crop.copy()
    bx1, by1 = bbox[0], bbox[1]
    for region in exclude_regions:
        # Convert from original image coords to panel-local coords
        rx1 = max(region[0] - bx1, 0)
        ry1 = max(region[1] - by1, 0)
        rx2 = min(region[2] - bx1, result.shape[1])
        ry2 = min(region[3] - by1, result.shape[0])
        if rx2 > rx1 and ry2 > ry1:
            result[ry1:ry2, rx1:rx2] = 255
    return result


def bbox_contains(outer: List[int], inner: List[int]) -> bool:
    """Check whether inner bbox is fully contained within outer bbox."""
    if not outer or not inner:
        return True
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def bbox_area(b: List[int]) -> int:
    return (b[2] - b[0]) * (b[3] - b[1]) if b else 0


def get_local_bbox(bbox_by_image: dict, page_num: int, crop_idx: int, panel_name: str) -> Optional[List[int]]:
    """Return the bbox relative to image_n_m.png from panel_bbox_matches_by_image.json."""
    crop_key = f'image_{page_num}_{crop_idx}.png'
    entries = bbox_by_image.get('by_page', {}).get(str(page_num), {}).get(crop_key, {}).get(panel_name, [])
    for e in entries:
        b = e.get('bbox')
        if b:
            return b
    return None


# ── single panel extraction ───────────────────────────────────────────────────

def process_one_panel(
    client: AzureOpenAI,
    deployment: str,
    guide_images: List[np.ndarray],
    page_num: int,
    crop_idx: int,
    panel_name: str,
    img: np.ndarray,
    all_name_bboxes: Dict[str, Optional[List[int]]],
    page_out_dir: Path,
    page_debug_dir: Path,
    grid_size: int = 120,
    verify_max_tries: int = 3,
    skip_extraction: bool = False,
) -> Optional[dict]:
    """Run the locate → verify loop for a single panel.

    If skip_extraction=True, save the entire image as the panel without calling the LLM.
    (Used when there is only 1 crop and 1 panel.)
    """
    h, w = img.shape[:2]
    matched_bbox = all_name_bboxes.get(panel_name)
    s = safe_name(panel_name)

    print(f'\n  [page{page_num}|crop{crop_idx}|{panel_name}] matched_bbox={matched_bbox}')

    # ── Skip: the entire image is the panel region ───────────────────────────
    if skip_extraction:
        out_path = page_out_dir / f'panel_{s}.png'
        cv2.imwrite(str(out_path), img)
        print(f'  [page{page_num}|{panel_name}] ✓ skip extraction — saved whole image: {out_path.name}')
        return {
            'page_num': page_num,
            'panel_name': panel_name,
            'file': str(out_path),
            'bbox': [0, 0, w, h],
            'crop_idx': crop_idx,
        }

    if matched_bbox is None:
        print(f'    WARNING: matched_bbox is None — skipping')
        return None

    other_name_bboxes = [
        (n, b) for n, b in all_name_bboxes.items()
        if n != panel_name and b is not None
    ]

    locate_img = make_locate_input_image(
        img, panel_name, grid_size,
        target_bbox=matched_bbox,
        other_name_bboxes=other_name_bboxes,
    )
    cv2.imwrite(str(page_debug_dir / f'panel_{s}_locate.png'), locate_img)

    locate_prompt = build_locate_prompt(panel_name, w, h, grid_size, LOCATE_SCHEMA)

    ok = False
    bbox: Optional[List[int]] = None
    corrected_bbox: Optional[List[int]] = None
    last_failed_bbox: Optional[List[int]] = None  # displayed on locate_img on retry
    exclude_regions: List[List[int]] = []

    # ── best_bbox tracking: save the result with the fewest issues even if verify fails ──
    best_bbox: Optional[List[int]] = None
    best_exclude_regions: List[List[int]] = []
    best_score: int = 5  # number of non-ok edges (initial value = exceeds 4)

    # ── Per-axis oscillation detection: track history of LLM-suggested values per axis ──
    AXES = ['x1', 'y1', 'x2', 'y2']
    axis_history: Dict[str, List[int]] = {a: [] for a in AXES}

    for attempt in range(1, verify_max_tries + 1):
        # ── Step 1: determine bbox ────────────────────────────────────────────
        if corrected_bbox is not None:
            bbox = corrected_bbox
            corrected_bbox = None
            print(f'    [page{page_num}|{panel_name}|attempt {attempt}] using corrected_bbox: {bbox}')
        else:
            # On retry, display the previous failed bbox in red on locate_img to provide context to the LLM
            if last_failed_bbox is not None:
                from .panel_images import draw_target_bbox
                retry_locate_img = draw_target_bbox(
                    locate_img, last_failed_bbox,
                    color=(0, 0, 255), label='FAILED',
                )
            else:
                retry_locate_img = locate_img

            loc_raw = call_llm(
                client, deployment, locate_prompt,
                [*guide_images, retry_locate_img],
                label=f'locate {panel_name}' if attempt == 1 else f'relocate {panel_name} (attempt {attempt})',
            )
            loc = parse_json(loc_raw)
            if not loc or not loc.get('found', False):
                print(f'    [page{page_num}|{panel_name}|attempt {attempt}] locate: not found')
                break
            bbox = sanitize_bbox(loc.get('bbox'), w, h)
            if not bbox:
                print(f'    [page{page_num}|{panel_name}|attempt {attempt}] locate: no valid bbox in {loc.get("bbox")}')
                break
            exclude_regions = loc.get('exclude_regions', [])

        # ── Step 2: basic bbox validation ────────────────────────────────────
        area_ratio = bbox_area(bbox) / (w * h) * 100
        print(f'    [page{page_num}|{panel_name}|attempt {attempt}] area={area_ratio:.1f}%  bbox={bbox}')

        # Relaxed containment: NAME center within bbox (expanded by NAME dims)
        _mcx = (matched_bbox[0] + matched_bbox[2]) / 2
        _mcy = (matched_bbox[1] + matched_bbox[3]) / 2
        _mw = matched_bbox[2] - matched_bbox[0]
        _mh = matched_bbox[3] - matched_bbox[1]
        _ex = [bbox[0] - _mw, bbox[1] - _mh, bbox[2] + _mw, bbox[3] + _mh]
        if not (_ex[0] <= _mcx <= _ex[2] and _ex[1] <= _mcy <= _ex[3]):
            print(f'    [page{page_num}|{panel_name}|attempt {attempt}] REJECT: NAME center outside bbox+margin')
            bbox = None
            continue

        if area_ratio > 80:
            print(f'    [page{page_num}|{panel_name}|attempt {attempt}] REJECT: area too large ({area_ratio:.1f}%)')
            bbox = None
            continue

        # ── Step 3: crop + verify ─────────────────────────────────────────────
        panel_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        verify_overlay = make_verify_overlay_image(
            img, panel_name, bbox, grid_size,
            name_bbox=matched_bbox,
            other_name_bboxes=other_name_bboxes,
        )
        cv2.imwrite(str(page_debug_dir / f'panel_{s}_v{attempt}.png'), verify_overlay)
        cv2.imwrite(str(page_debug_dir / f'panel_{s}_v{attempt}.crop.png'), panel_crop)

        verify_prompt = build_verify_prompt(VERIFY_SCHEMA, bbox, w, h, exclude_regions=exclude_regions)
        ver_raw = call_llm(
            client, deployment, verify_prompt,
            [*guide_images, verify_overlay, panel_crop],
            label=f'verify {panel_name} ({attempt}/{verify_max_tries})',
            reasoning_effort="none",
        )
        ver = parse_json(ver_raw)
        valid = ver.get('valid') if ver else None
        print(f'    [page{page_num}|{panel_name}|attempt {attempt}] verify: valid={valid}')

        if ver and valid:
            ok = True
            break

        # verify failed → update best_bbox and print edges
        last_failed_bbox = bbox

        if ver:
            edges = ver.get('edges', {})

            # best_bbox: save the bbox with the fewest non-ok edges
            non_ok_count = sum(1 for a in AXES if edges.get(a, {}).get('status', 'ok') != 'ok')
            if non_ok_count < best_score:
                best_score = non_ok_count
                best_bbox = bbox
                best_exclude_regions = ver.get('exclude_regions', exclude_regions)
                print(f'    [page{page_num}|{panel_name}|attempt {attempt}] best_bbox updated (score={non_ok_count}): {bbox}')

            for edge_name in AXES:
                edge = edges.get(edge_name, {})
                status = edge.get('status', '?')
                if status != 'ok':
                    issue = edge.get('issue', '')
                    delta = edge.get('delta', 0)
                    corrected_val = edge.get('corrected', '?')
                    print(f'      {edge_name}: {status} delta={delta} → {corrected_val}  {issue}')

            cb = sanitize_bbox(ver.get('corrected_bbox'), w, h)
            # Relaxed check: NAME center within corrected bbox (expanded by NAME dims)
            if cb and matched_bbox:
                _mcx2 = (matched_bbox[0] + matched_bbox[2]) / 2
                _mcy2 = (matched_bbox[1] + matched_bbox[3]) / 2
                _mw2 = matched_bbox[2] - matched_bbox[0]
                _mh2 = matched_bbox[3] - matched_bbox[1]
                _ex2 = [cb[0] - _mw2, cb[1] - _mh2, cb[2] + _mw2, cb[3] + _mh2]
                _cb_ok = _ex2[0] <= _mcx2 <= _ex2[2] and _ex2[1] <= _mcy2 <= _ex2[3]
            else:
                _cb_ok = bool(cb)
            if _cb_ok:
                # Update exclude_regions from verify response
                exclude_regions = ver.get('exclude_regions', exclude_regions)
                # ── Per-axis oscillation detection: freeze the current value if the same axis alternates increase↔decrease ──
                cb_list = list(cb)
                current_vals = list(bbox)
                for i, axis_name in enumerate(AXES):
                    suggested = cb_list[i]
                    axis_history[axis_name].append(suggested)
                    hist = axis_history[axis_name]
                    if len(hist) >= 3:
                        a, b, c = hist[-3], hist[-2], hist[-1]
                        if (a < b and b > c) or (a > b and b < c):
                            cb_list[i] = current_vals[i]
                            print(f'      {axis_name}: oscillation detected (…{a}→{b}→{c}) → freeze at {current_vals[i]}')
                corrected_bbox = sanitize_bbox(cb_list, w, h)
                if corrected_bbox:
                    print(f'    [page{page_num}|{panel_name}|attempt {attempt}] corrected_bbox: {corrected_bbox}')
                else:
                    corrected_bbox = None
                    print(f'    [page{page_num}|{panel_name}|attempt {attempt}] no usable corrected_bbox → relocate with FAILED bbox shown')
            else:
                print(f'    [page{page_num}|{panel_name}|attempt {attempt}] no usable corrected_bbox → relocate with FAILED bbox shown')

    # Exited without valid=True → attempt to save using best_bbox
    if not ok:
        if best_bbox is not None:
            panel_crop = img[best_bbox[1]:best_bbox[3], best_bbox[0]:best_bbox[2]]
            bbox = best_bbox
            exclude_regions = best_exclude_regions
            ok = True
            print(f'  [page{page_num}|{panel_name}] using best_bbox (score={best_score}): {bbox}')
        else:
            print(f'  [page{page_num}|{panel_name}] FAILED — not saved')
            return None

    # Apply white fill on exclude_regions
    if exclude_regions:
        panel_crop = apply_exclude_regions(panel_crop, bbox, exclude_regions)
        print(f'  [page{page_num}|{panel_name}] applied {len(exclude_regions)} exclude_regions (white fill)')

    out_path = page_out_dir / f'panel_{s}.png'
    cv2.imwrite(str(out_path), panel_crop)
    print(f'  [page{page_num}|{panel_name}] ✓ saved: {out_path.name}')

    return {
        'page_num': page_num,
        'panel_name': panel_name,
        'file': str(out_path),
        'bbox': bbox,
        'exclude_regions': exclude_regions,
        'crop_idx': crop_idx,
    }


# ── main entry point ──────────────────────────────────────────────────────────

def run_panel_extraction(
    client: AzureOpenAI,
    deployment: str,
    guide_images: List[np.ndarray],
    crop_items_by_page: Dict[int, List[dict]],
    panel_names_data: dict,
    bbox_by_image: dict,
    output_root: Path,
    notebook_dir: Path,
    grid_size: int = 120,
    verify_max_tries: int = 3,
    max_workers: int = 8,
) -> dict:
    """Run the full panel extraction pipeline.

    Returns
    -------
    summary : {'by_page': {str(page_num): [result_dict, ...]}}
              result_dict keys: panel_name, files, bboxes, crop_idx
    """
    debug_dir = output_root / '_debug_inputs'
    debug_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = []
    for page_num, crop_items in sorted(crop_items_by_page.items()):
        names = panel_names_data['by_page'][str(page_num)]
        page_out_dir = output_root / f'page{page_num}'
        page_out_dir.mkdir(parents=True, exist_ok=True)
        page_debug_dir = debug_dir / f'page{page_num}'
        page_debug_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n{"="*70}')
        print(f'Page {page_num} — {len(names)} panels, {len(crop_items)} crops')
        print(f'{"="*70}')

        for crop_item in crop_items:
            crop_idx = crop_item['crop_idx']
            img = cv2.imread(str(crop_item['path']))
            if img is None:
                print(f'  [crop {crop_idx}] cannot load {crop_item["path"].name} — skip')
                continue

            h, w = img.shape[:2]
            all_name_bboxes = {
                name: get_local_bbox(bbox_by_image, page_num, crop_idx, name)
                for name in names
            }
            print(f'  crop{crop_idx} ({w}x{h}px) — name bboxes:')
            for name, b in all_name_bboxes.items():
                print(f'    {name}: {b}')

            # If only 1 panel in this crop has a bbox, skip LLM extraction (whole image = panel)
            panels_in_crop = [n for n, b in all_name_bboxes.items() if b is not None]
            skip = len(panels_in_crop) == 1

            for panel_name in names:
                all_tasks.append((
                    client, deployment, guide_images,
                    page_num, crop_idx, panel_name, img, all_name_bboxes,
                    page_out_dir, page_debug_dir, grid_size, verify_max_tries,
                    skip,
                ))

    print(f'\nRunning {len(all_tasks)} panel tasks in parallel...')

    results_by_page: Dict[int, list] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_panel, *task): task for task in all_tasks}
        for future in as_completed(futures):
            result = future.result()
            if result:
                pn = result.pop('page_num')
                result['file'] = str(Path(result['file']).relative_to(notebook_dir))
                results_by_page[pn].append(result)

    summary: dict = {'by_page': {}}
    for page_num in sorted(results_by_page.keys()):
        items = sorted(results_by_page[page_num], key=lambda x: (x['crop_idx'], x['panel_name']))
        summary['by_page'][str(page_num)] = items

    # ── Post-extraction validation ────────────────────────────────────────────
    for page_num, crop_items in sorted(crop_items_by_page.items()):
        names = panel_names_data['by_page'][str(page_num)]
        for crop_item in crop_items:
            crop_idx = crop_item['crop_idx']
            img = cv2.imread(str(crop_item['path']))
            if img is None:
                continue
            h, w = img.shape[:2]
            all_name_bboxes = {
                name: get_local_bbox(bbox_by_image, page_num, crop_idx, name)
                for name in names
            }
            page_items = [r for r in results_by_page.get(page_num, []) if r.get('crop_idx') == crop_idx]
            found_names = {r['panel_name'] for r in page_items}
            missing = [n for n in names if n not in found_names and all_name_bboxes.get(n) is not None]
            if missing:
                print(f'  [page{page_num}|crop{crop_idx}] ⚠ MISSING {len(missing)} panel(s): {missing}')
            else:
                print(f'  [page{page_num}|crop{crop_idx}] ✓ all {len(found_names)} panel(s) located')
            if page_items:
                mask = np.zeros((h, w), dtype=np.uint8)
                for r in page_items:
                    bx = r['bbox']
                    mask[bx[1]:bx[3], bx[0]:bx[2]] = 1
                covered = int(np.sum(mask))
                total = w * h
                cov_pct = covered / total * 100
                print(f'  [page{page_num}|crop{crop_idx}] coverage: {cov_pct:.1f}% ({covered:,}/{total:,}px)  uncovered: {100-cov_pct:.1f}%')
                if 100 - cov_pct > 20:
                    print(f'  [page{page_num}|crop{crop_idx}] ⚠ large uncovered area ({100-cov_pct:.1f}%)')

    summary_path = output_root / 'panel_areas_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nSaved summary: {summary_path}')

    return summary


# ── Batch mode: locate ALL panels in one LLM call ─────────────────────────────

def process_all_panels_batch(
    client: AzureOpenAI,
    deployment: str,
    guide_images: List[np.ndarray],
    page_num: int,
    crop_idx: int,
    names: List[str],
    img: np.ndarray,
    all_name_bboxes: Dict[str, Optional[List[int]]],
    page_out_dir: Path,
    page_debug_dir: Path,
    grid_size: int = 120,
    verify_max_tries: int = 3,
    verify_reasoning_effort: str = 'none',
) -> List[dict]:
    """Locate ALL panels in one LLM call, then verify each crop individually.

    Returns list of result dicts for panels that succeeded.
    """
    h, w = img.shape[:2]
    panels_with_bbox = [n for n in names if all_name_bboxes.get(n) is not None]

    if not panels_with_bbox:
        print(f'  [page{page_num}|crop{crop_idx}] batch: no panels with matched bbox — skip')
        return []

    # ── Step 1: locate ALL panels in one LLM call ─────────────────────────────
    locate_img = make_locate_all_input_image(img, all_name_bboxes, grid_size)
    cv2.imwrite(str(page_debug_dir / f'batch_crop{crop_idx}_locate_all.png'), locate_img)

    locate_prompt = build_locate_all_prompt(panels_with_bbox, w, h, grid_size, LOCATE_ALL_SCHEMA)
    loc_raw = call_llm(
        client, deployment, locate_prompt,
        [*guide_images, locate_img],
        label=f'locate ALL {len(panels_with_bbox)} panels (crop{crop_idx})',
    )
    loc = parse_json(loc_raw)

    if not loc or 'panels' not in loc:
        print(f'  [page{page_num}|crop{crop_idx}] batch: locate-all returned no valid JSON')
        return []

    located_bboxes: Dict[str, Optional[List[int]]] = {}
    located_exclude_regions: Dict[str, List[List[int]]] = {}
    for entry in loc.get('panels', []):
        name = entry.get('name', '').strip()
        raw_bbox = entry.get('bbox')
        if entry.get('found') and raw_bbox:
            b = sanitize_bbox(raw_bbox, w, h)
            matched = all_name_bboxes.get(name)
            # Accept if the NAME bbox center falls inside the located panel bbox
            # (relaxed from full containment to avoid rejecting panels where
            #  the name text sits at the edge of the panel region)
            if b and matched:
                name_cx = (matched[0] + matched[2]) / 2
                name_cy = (matched[1] + matched[3]) / 2
                # Expand panel bbox by NAME dimensions as margin tolerance
                # (panel names often sit at/just outside the panel edge)
                name_w = matched[2] - matched[0]
                name_h = matched[3] - matched[1]
                ex = [b[0] - name_w, b[1] - name_h, b[2] + name_w, b[3] + name_h]
                if ex[0] <= name_cx <= ex[2] and ex[1] <= name_cy <= ex[3]:
                    located_bboxes[name] = b
                    located_exclude_regions[name] = entry.get('exclude_regions', [])
                else:
                    print(f'    [{name}] locate-all: NAME center ({name_cx:.0f},{name_cy:.0f}) outside panel bbox {b} (margin={name_w},{name_h}) — skip')
                    located_bboxes[name] = None
            else:
                print(f'    [{name}] locate-all: bbox invalid or no matched NAME bbox — skip')
                located_bboxes[name] = None
        else:
            print(f'    [{name}] locate-all: not found')
            located_bboxes[name] = None

    # ── Step 2: verify ALL pending panels in batch per round ───────────────────
    AXES = ['x1', 'y1', 'x2', 'y2']

    # Build initial per-panel state
    panel_states: Dict[str, dict] = {}
    for panel_name in names:
        matched_bbox = all_name_bboxes.get(panel_name)
        if matched_bbox is None:
            continue
        initial_bbox = located_bboxes.get(panel_name)
        if initial_bbox is None:
            print(f'  [page{page_num}|crop{crop_idx}|{panel_name}] batch: no valid bbox — skip')
            continue
        print(f'\n  [page{page_num}|crop{crop_idx}|{panel_name}] batch initial_bbox={initial_bbox}')
        panel_states[panel_name] = {
            'bbox': initial_bbox,
            'exclude_regions': located_exclude_regions.get(panel_name, []),
            'best_bbox': None,
            'best_exclude_regions': [],
            'best_score': 5,
            'corrected_bbox': None,
            'axis_history': {a: [] for a in AXES},
            'ok': False,
            'panel_crop': None,
        }

    for attempt in range(1, verify_max_tries + 1):
        pending = [n for n, s in panel_states.items() if not s['ok']]
        if not pending:
            break

        # Apply corrected_bbox corrections for this round
        for name in pending:
            state = panel_states[name]
            if state['corrected_bbox'] is not None:
                state['bbox'] = state['corrected_bbox']
                state['corrected_bbox'] = None
                print(f'    [{name}|attempt {attempt}] using corrected_bbox: {state["bbox"]}')

        # Reject panels with oversized bbox
        still_pending = []
        for name in pending:
            state = panel_states[name]
            bbox = state['bbox']
            area_ratio = bbox_area(bbox) / (w * h) * 100
            print(f'    [{name}|attempt {attempt}] area={area_ratio:.1f}%  bbox={bbox}')
            if area_ratio > 80:
                print(f'    [{name}|attempt {attempt}] REJECT: area too large ({area_ratio:.1f}%)')
                # Mark as done (failed) — will fall back to best_bbox at the end
            else:
                still_pending.append(name)
        pending = still_pending
        if not pending:
            break

        # Build batch verify images
        panel_bboxes_for_overlay = {n: panel_states[n]['bbox'] for n in pending}
        verify_overlay = make_verify_all_overlay_image(
            img, panel_bboxes_for_overlay, all_name_bboxes, grid_size,
        )
        cv2.imwrite(str(page_debug_dir / f'batch_verify_overlay_a{attempt}.png'), verify_overlay)

        panel_crops_ordered = []
        for name in pending:
            bbox = panel_states[name]['bbox']
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            panel_states[name]['panel_crop'] = crop
            s = safe_name(name)
            cv2.imwrite(str(page_debug_dir / f'panel_{s}_v{attempt}.png'), verify_overlay)
            cv2.imwrite(str(page_debug_dir / f'panel_{s}_v{attempt}.crop.png'), crop)
            panel_crops_ordered.append(crop)

        panels_info = [{'name': n, 'bbox': panel_states[n]['bbox'], 'exclude_regions': panel_states[n]['exclude_regions']} for n in pending]
        verify_prompt = build_verify_all_prompt(panels_info, w, h, VERIFY_ALL_SCHEMA)

        ver_raw = call_llm(
            client, deployment, verify_prompt,
            [*guide_images, verify_overlay, *panel_crops_ordered],
            label=f'verify ALL {len(pending)} panels (attempt {attempt}/{verify_max_tries})',
            reasoning_effort=verify_reasoning_effort,
        )
        ver = parse_json(ver_raw)

        if not ver or 'panels' not in ver:
            print(f'    [attempt {attempt}] batch verify: no valid JSON — skip round')
            continue

        for panel_result in ver.get('panels', []):
            name = panel_result.get('name', '').strip()
            if name not in panel_states:
                continue
            state = panel_states[name]
            matched_bbox = all_name_bboxes.get(name)
            bbox = state['bbox']
            valid = panel_result.get('valid')
            print(f'    [{name}|attempt {attempt}] verify: valid={valid}')

            if valid:
                state['ok'] = True
                continue

            edges = panel_result.get('edges', {})
            non_ok_count = sum(1 for a in AXES if edges.get(a, {}).get('status', 'ok') != 'ok')
            if non_ok_count < state['best_score']:
                state['best_score'] = non_ok_count
                state['best_bbox'] = bbox
                state['best_exclude_regions'] = panel_result.get('exclude_regions', state['exclude_regions'])
                print(f'    [{name}|attempt {attempt}] best_bbox updated (score={non_ok_count}): {bbox}')

            for edge_name in AXES:
                edge = edges.get(edge_name, {})
                status = edge.get('status', '?')
                if status != 'ok':
                    delta = edge.get('delta', 0)
                    corrected_val = edge.get('corrected', '?')
                    print(f'      {name} {edge_name}: {status} delta={delta} → {corrected_val}  {edge.get("issue", "")}')

            cb = sanitize_bbox(panel_result.get('corrected_bbox'), w, h)
            # Relaxed check: NAME center within corrected bbox (expanded by NAME dims)
            if cb and matched_bbox:
                _mcx = (matched_bbox[0] + matched_bbox[2]) / 2
                _mcy = (matched_bbox[1] + matched_bbox[3]) / 2
                _mw = matched_bbox[2] - matched_bbox[0]
                _mh = matched_bbox[3] - matched_bbox[1]
                _ex = [cb[0] - _mw, cb[1] - _mh, cb[2] + _mw, cb[3] + _mh]
                _name_near = _ex[0] <= _mcx <= _ex[2] and _ex[1] <= _mcy <= _ex[3]
            else:
                _name_near = bool(cb)
            if _name_near:
                # Update exclude_regions from verify response
                state['exclude_regions'] = panel_result.get('exclude_regions', state['exclude_regions'])
                cb_list = list(cb)
                current_vals = list(bbox)
                for i, axis_name in enumerate(AXES):
                    state['axis_history'][axis_name].append(cb_list[i])
                    hist = state['axis_history'][axis_name]
                    if len(hist) >= 3:
                        a_v, b_v, c_v = hist[-3], hist[-2], hist[-1]
                        if (a_v < b_v and b_v > c_v) or (a_v > b_v and b_v < c_v):
                            cb_list[i] = current_vals[i]
                            print(f'      {name} {axis_name}: oscillation detected → freeze at {current_vals[i]}')
                corrected = sanitize_bbox(cb_list, w, h)
                if corrected:
                    state['corrected_bbox'] = corrected
                    print(f'    [{name}|attempt {attempt}] corrected_bbox: {corrected}')
                else:
                    print(f'    [{name}|attempt {attempt}] no usable corrected_bbox')
            else:
                print(f'    [{name}|attempt {attempt}] no usable corrected_bbox')

    # Collect results
    results = []
    for panel_name, state in panel_states.items():
        bbox = state['bbox']
        panel_crop = state['panel_crop']
        exclude_regions = state['exclude_regions']

        if not state['ok']:
            if state['best_bbox'] is not None:
                bbox = state['best_bbox']
                exclude_regions = state['best_exclude_regions']
                panel_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                print(f'  [page{page_num}|{panel_name}] using best_bbox (score={state["best_score"]}): {bbox}')
            else:
                print(f'  [page{page_num}|{panel_name}] FAILED — not saved')
                continue
        elif panel_crop is None:
            panel_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        # Apply white fill on exclude_regions
        if exclude_regions:
            panel_crop = apply_exclude_regions(panel_crop, bbox, exclude_regions)
            print(f'  [page{page_num}|{panel_name}] applied {len(exclude_regions)} exclude_regions (white fill)')

        s = safe_name(panel_name)
        out_path = page_out_dir / f'panel_{s}.png'
        cv2.imwrite(str(out_path), panel_crop)
        print(f'  [page{page_num}|{panel_name}] ✓ saved: {out_path.name}')
        results.append({
            'page_num': page_num,
            'panel_name': panel_name,
            'file': str(out_path),
            'bbox': bbox,
            'exclude_regions': exclude_regions,
            'crop_idx': crop_idx,
            'crop_w': w,
            'crop_h': h,
        })

    # ── Post-locate validation ──────────────────────────────────────────────────
    # 1) Completeness: check all requested panel names were located
    found_names = {r['panel_name'] for r in results}
    missing_names = [n for n in names if n not in found_names and all_name_bboxes.get(n) is not None]
    if missing_names:
        print(f'\n  [page{page_num}|crop{crop_idx}] ⚠ MISSING {len(missing_names)} panel(s): {missing_names}')
    else:
        print(f'\n  [page{page_num}|crop{crop_idx}] ✓ all {len(found_names)} panel(s) located')

    # 2) Coverage: check what fraction of image area is covered by located bboxes
    if results:
        mask = np.zeros((h, w), dtype=np.uint8)
        for r in results:
            bx = r['bbox']
            mask[bx[1]:bx[3], bx[0]:bx[2]] = 1
        covered_px = int(np.sum(mask))
        total_px = w * h
        coverage_pct = covered_px / total_px * 100
        uncovered_pct = 100 - coverage_pct
        print(f'  [page{page_num}|crop{crop_idx}] coverage: {coverage_pct:.1f}% ({covered_px:,}/{total_px:,}px)  uncovered: {uncovered_pct:.1f}%')
        if uncovered_pct > 20:
            print(f'  [page{page_num}|crop{crop_idx}] ⚠ large uncovered area ({uncovered_pct:.1f}%) — some panels may be missed or bboxes too small')

    return results


def run_panel_extraction_batch(
    client: AzureOpenAI,
    deployment: str,
    guide_images: List[np.ndarray],
    crop_items_by_page: Dict[int, List[dict]],
    panel_names_data: dict,
    bbox_by_image: dict,
    output_root: Path,
    notebook_dir: Path,
    grid_size: int = 120,
    verify_max_tries: int = 3,
    max_workers: int = 4,
    verify_reasoning_effort: str = 'none',
) -> dict:
    """Batch version of run_panel_extraction.

    For each crop image, sends ONE locate call asking the LLM to identify ALL
    panel areas simultaneously, then verifies each crop individually.

    Compared to run_panel_extraction (per-panel), this reduces locate calls
    from N (one per panel) to 1 per crop image.
    """
    debug_dir = output_root / '_debug_inputs'
    debug_dir.mkdir(parents=True, exist_ok=True)

    batch_tasks = []
    for page_num, crop_items in sorted(crop_items_by_page.items()):
        names = panel_names_data['by_page'][str(page_num)]
        page_out_dir = output_root / f'page{page_num}'
        page_out_dir.mkdir(parents=True, exist_ok=True)
        page_debug_dir = debug_dir / f'page{page_num}'
        page_debug_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n{"="*70}')
        print(f'Page {page_num} — {len(names)} panels, {len(crop_items)} crops')
        print(f'{"="*70}')

        for crop_item in crop_items:
            crop_idx = crop_item['crop_idx']
            img = cv2.imread(str(crop_item['path']))
            if img is None:
                print(f'  [crop {crop_idx}] cannot load {crop_item["path"].name} — skip')
                continue

            h, w = img.shape[:2]
            all_name_bboxes = {
                name: get_local_bbox(bbox_by_image, page_num, crop_idx, name)
                for name in names
            }
            print(f'  crop{crop_idx} ({w}x{h}px) — name bboxes:')
            for name, b in all_name_bboxes.items():
                print(f'    {name}: {b}')

            batch_tasks.append((
                client, deployment, guide_images,
                page_num, crop_idx, names, img, all_name_bboxes,
                page_out_dir, page_debug_dir, grid_size, verify_max_tries,
                verify_reasoning_effort,
            ))

    print(f'\nRunning {len(batch_tasks)} crop tasks (locate-all per crop)...')

    # Clear call log before this run
    from .panel_utils import llm_call_log
    llm_call_log.clear()

    results_by_page: Dict[int, list] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_all_panels_batch, *task): task for task in batch_tasks}
        for future in as_completed(futures):
            for result in future.result():
                pn = result.pop('page_num')
                result['file'] = str(Path(result['file']).relative_to(notebook_dir))
                results_by_page[pn].append(result)

    summary: dict = {'by_page': {}}
    for page_num in sorted(results_by_page.keys()):
        items = sorted(results_by_page[page_num], key=lambda x: (x['crop_idx'], x['panel_name']))
        summary['by_page'][str(page_num)] = items

    # Save LLM call timing log under timing/ directory
    summary['llm_call_log'] = list(llm_call_log)
    timing_dir = output_root.parent / 'timing'
    timing_dir.mkdir(parents=True, exist_ok=True)
    llm_log_path = timing_dir / 'nb05_panel_area_llm_calls.json'
    llm_log_path.write_text(json.dumps(llm_call_log, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved LLM call log: {llm_log_path} ({len(llm_call_log)} calls)')

    summary_path = output_root / 'panel_areas_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nSaved summary: {summary_path}')

    return summary
