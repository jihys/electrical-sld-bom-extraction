"""bay_pipeline.py
-----------------
Pipeline for splitting panel crop images into bay units when the image is large enough.

Public API
----------
run_bay_splitting(
    client, deployment, guide_images,
    panel_areas_summary, notebook_dir, output_root,
    grid_size=120, min_width=600, max_workers=8,
) -> dict   # bay_areas_summary {'by_page': {...}}
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

from .panel_images import add_grid_overlay
from .panel_prompts import BAY_SCHEMA, build_bay_prompt
from .panel_utils import call_llm, parse_json, safe_name, sanitize_bbox


def _parse_bboxes(raw: list, w: int, h: int) -> List[List[int]]:
    result = []
    for item in (raw or []):
        b = sanitize_bbox(item, w, h)
        if b:
            result.append(b)
    return result


def process_one_bay_split(
    client: AzureOpenAI,
    deployment: str,
    guide_images: List[np.ndarray],
    panel_name: str,
    panel_img_path: Path,
    out_dir: Path,
    grid_size: int = 120,
    min_width: int = 600,
    crop_w: Optional[int] = None,
    reference_width: int = 2000,
) -> Optional[dict]:
    """Split a single panel image into bay units.

    If the image width is less than the effective min_width, skip (return None).
    When *crop_w* (source crop image width) is provided, the threshold is
    scaled: ``effective = min_width * crop_w / reference_width``.
    If there is only 1 bay, skip (return None).
    If there are 2 or more bays, save per-bay crops and return the result.
    """
    img = cv2.imread(str(panel_img_path))
    if img is None:
        print(f'  [{panel_name}] cannot load {panel_img_path.name} — skip')
        return None

    h, w = img.shape[:2]

    if crop_w and reference_width > 0:
        effective_min_width = int(min_width * crop_w / reference_width)
    else:
        effective_min_width = min_width

    if w < effective_min_width:
        print(f'  [{panel_name}] skip (width={w} < effective_min_width={effective_min_width} [base={min_width}, crop_w={crop_w}])')
        return None

    print(f'\n  [{panel_name}] bay split attempt — image {w}x{h}px')

    overlay = add_grid_overlay(img, grid_size)
    prompt = build_bay_prompt(panel_name, w, h, grid_size, BAY_SCHEMA)

    raw = call_llm(
        client, deployment, prompt,
        [*guide_images, overlay],
        label=f'bay locate {panel_name}',
    )
    result = parse_json(raw)
    if not result:
        print(f'  [{panel_name}] bay locate: parse failed')
        return None

    n_bays = result.get('n_bays', 1)
    bboxes = _parse_bboxes(result.get('bboxes', []), w, h)

    print(f'  [{panel_name}] n_bays={n_bays}, valid bboxes={len(bboxes)}')

    if n_bays <= 1 or len(bboxes) <= 1:
        print(f'  [{panel_name}] single bay — no split')
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    s = safe_name(panel_name)
    bay_files = []
    for i, bbox in enumerate(bboxes):
        crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        out_path = out_dir / f'panel_{s}_bay{i + 1:02d}.png'
        cv2.imwrite(str(out_path), crop)
        bay_files.append({'bbox': bbox, 'file': str(out_path)})

    print(f'  [{panel_name}] ✓ split into {len(bboxes)} bays → {[Path(b["file"]).name for b in bay_files]}')
    return {
        'panel_name': panel_name,
        'source_file': str(panel_img_path),
        'bays': bay_files,
    }


def run_bay_splitting(
    client: AzureOpenAI,
    deployment: str,
    guide_images: List[np.ndarray],
    panel_areas_summary: dict,
    notebook_dir: Path,
    output_root: Path,
    grid_size: int = 120,
    min_width: int = 600,
    reference_width: int = 2000,
    max_workers: int = 8,
) -> dict:
    """Attempt bay splitting for each panel image in panel_areas_summary.

    Returns
    -------
    bay_summary : {'by_page': {str(page_num): [result_dict, ...]}}
                  result_dict keys: panel_name, source_file, bays (list of {bbox, file})
    """
    tasks = []
    for page_key, items in panel_areas_summary.get('by_page', {}).items():
        page_num = int(page_key)
        page_out_dir = output_root / f'page{page_num}'
        for item in items:
            f = item.get('file')
            if not f:
                continue
            panel_img_path = notebook_dir / f
            item_crop_w = item.get('crop_w')
            tasks.append((page_num, item['panel_name'], panel_img_path, page_out_dir, item_crop_w))

    print(f'\nRunning bay splitting for {len(tasks)} panels in parallel...')

    results_by_page: Dict[int, list] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one_bay_split,
                client, deployment, guide_images,
                panel_name, panel_img_path, page_out_dir,
                grid_size, min_width, item_crop_w, reference_width,
            ): (page_num, panel_name)
            for page_num, panel_name, panel_img_path, page_out_dir, item_crop_w in tasks
        }
        for future in as_completed(futures):
            page_num, panel_name = futures[future]
            result = future.result()
            if result:
                result['source_file'] = str(Path(result['source_file']).relative_to(notebook_dir))
                for bay in result['bays']:
                    bay['file'] = str(Path(bay['file']).relative_to(notebook_dir))
                results_by_page[page_num].append(result)

    bay_summary: dict = {'by_page': {}}
    for page_num in sorted(results_by_page.keys()):
        items = sorted(results_by_page[page_num], key=lambda x: x['panel_name'])
        bay_summary['by_page'][str(page_num)] = items

    summary_path = output_root / 'bay_areas_summary.json'
    summary_path.write_text(json.dumps(bay_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nSaved bay summary: {summary_path}')

    return bay_summary
