"""
Batch verification test: verify N new crops in a single LLM call.

Current approach:  🟠existing + 🟢pending + 🟣current  → 1 crop per LLM call (sequential)
New approach:      🟠existing + 🟣all-new-candidates    → N crops in 1 LLM call (batch)

Usage:
    cd electrical-sld-bom-extraction
    python tests/test_batch_verify.py --pages 4 7 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# -- project imports ----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cad.llm_client import (
    make_client,
    responses_call,
    responses_call_with_image,
    safe_parse_json,
    VALIDATION_SCHEMA,
    _build_detection_prompt,
)
from cad.image_utils import (
    GRID_STEP_X,
    GRID_STEP_Y,
    detect_content_bbox,
    draw_grid_with_cell_numbers,
    image_to_data_url,
)
from cad.pipeline import (
    load_existing_crops,
    validate_page_with_gpt,
    _page_image_path,
    _filter_new_polygons,
    polygon_to_bbox,
)
from cad.data_types import ImageCropInfo
from cad.data_types import crop_polygon

# ── Colors (BGR) ────────────────────────────────────────────────────────────────
ORANGE = (22, 115, 249)   # existing confirmed crops
PURPLE = (234, 51, 147)   # new candidate crops


def _hex_to_bgr(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _draw_polygon(img, polygon, color, width=10):
    pts = np.array(
        [[int(polygon[i]), int(polygon[i + 1])] for i in range(0, len(polygon), 2)],
        dtype=np.int32,
    )
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=width)


# ── Batch verification schema ──────────────────────────────────────────────────

BATCH_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "crops": {
            "type": "array",
            "description": "Evaluation result for each PURPLE candidate crop, in order.",
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Crop index (matching the label on the overlay image)",
                    },
                    "is_electrical": {
                        "type": "boolean",
                        "description": (
                            "True if this crop contains an electrical circuit diagram "
                            "(symbols, connection lines, hierarchical flow). "
                            "False if it contains only text tables, notes, legends, title blocks, "
                            "photos, or non-electrical content."
                        ),
                    },
                    "issue": {
                        "type": "string",
                        "description": "Brief description of the issue (empty string if correct)",
                    },
                    "x1": {
                        "type": "object",
                        "description": "Left edge analysis",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["right", "left", "none"],
                                "description": (
                                    "left=expand left (include more content); "
                                    "right=shrink left (trim whitespace); "
                                    "none=correct"
                                ),
                            },
                            "corrected": {"type": "integer", "description": "Corrected x1 pixel"},
                        },
                        "required": ["direction", "corrected"],
                    },
                    "y1": {
                        "type": "object",
                        "description": "Top edge analysis",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["down", "up", "none"],
                                "description": (
                                    "up=expand top (include more content); "
                                    "down=shrink top (trim whitespace); "
                                    "none=correct"
                                ),
                            },
                            "corrected": {"type": "integer", "description": "Corrected y1 pixel"},
                        },
                        "required": ["direction", "corrected"],
                    },
                    "x2": {
                        "type": "object",
                        "description": "Right edge analysis",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["right", "left", "none"],
                                "description": (
                                    "right=expand right (include more content); "
                                    "left=shrink right (trim whitespace); "
                                    "none=correct"
                                ),
                            },
                            "corrected": {"type": "integer", "description": "Corrected x2 pixel"},
                        },
                        "required": ["direction", "corrected"],
                    },
                    "y2": {
                        "type": "object",
                        "description": "Bottom edge analysis",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["down", "up", "none"],
                                "description": (
                                    "down=expand bottom (include more content); "
                                    "up=shrink bottom (trim whitespace); "
                                    "none=correct"
                                ),
                            },
                            "corrected": {"type": "integer", "description": "Corrected y2 pixel"},
                        },
                        "required": ["direction", "corrected"],
                    },
                },
                "required": ["index", "is_electrical", "x1", "y1", "x2", "y2"],
            },
        },
    },
    "required": ["crops"],
}


# ── Batch verification prompt ──────────────────────────────────────────────────

def build_batch_verification_prompt(
    existing_boxes: List[Dict],
    candidate_boxes: List[Dict],
    image_size: Tuple[int, int],
    grid_step_x: int = GRID_STEP_X,
    grid_step_y: int = GRID_STEP_Y,
) -> str:
    """Build a prompt that asks the LLM to evaluate ALL candidate crops at once."""
    w, h = image_size

    prompt = (
        "You are verifying candidate bounding boxes for electrical circuit diagram regions "
        "on an engineering drawing page.\n\n"
        "**📸 IMAGE DESCRIPTION:**\n"
        f"- Full page image ({w}×{h} pixels) with a coordinate grid overlay\n"
        f"- Grid: lines every {grid_step_x}px (x-axis) and {grid_step_y}px (y-axis), "
        f"cells labeled (col, row)\n"
        "- 🟠 **ORANGE boxes**: Already-confirmed existing crop regions (DO NOT evaluate these)\n"
        "- 🟣 **PURPLE boxes with index labels**: NEW candidate regions to evaluate\n\n"
        "**🔌 What is an ELECTRICAL CIRCUIT DIAGRAM?**\n"
        "✅ Contains: visual SYMBOLS (transformers ⚡, breakers □, relays ○, motors M, switches), "
        "CONNECTION LINES (solid/dashed), HIERARCHICAL FLOW (source→transformer→breaker→load)\n"
        "❌ NOT a circuit diagram: pure text tables, revision history, relay function tables, "
        "title blocks, notes/legends (text only), photos, non-electrical drawings, "
        "page header strips\n\n"
        "**YOUR TASK:**\n"
        f"Evaluate each of the {len(candidate_boxes)} PURPLE candidate boxes.\n"
        "For each candidate:\n"
        "1. **is_electrical**: Does it contain an electrical circuit diagram? (true/false)\n"
        "2. **Edge analysis**: For each edge (x1, y1, x2, y2), determine if it needs adjustment:\n"
        "   - Check if circuit content is CUT OFF at any edge → expand that edge\n"
        "   - Check if there is EXCESS whitespace → shrink that edge\n"
        "   - If edge is correct → direction='none'\n"
        "3. **issue**: Brief description of any problem (empty if correct)\n\n"
        "**RULES:**\n"
        "1. If a candidate is NOT an electrical circuit diagram, set is_electrical=false "
        "and set all edge directions to 'none' (keep corrected values same as current bbox)\n"
        "2. If a candidate IS a valid circuit diagram with correct edges, set is_electrical=true "
        "and all directions to 'none'\n"
        "3. If edges need adjustment, provide the corrected pixel coordinates using the grid\n"
        "4. Include COMPLETE circuit content: all connection lines to endpoints, "
        "all components at bottom (motors/terminals)\n"
        "5. Better to include EXTRA space than cut off content\n\n"
    )

    prompt += f"**Existing ORANGE boxes (for reference only):** {json.dumps(existing_boxes)}\n\n"
    prompt += f"**PURPLE candidate boxes to evaluate:** {json.dumps(candidate_boxes)}\n\n"
    prompt += f"**Image size:** {w} × {h} pixels\n\n"
    prompt += f"**JSON output schema:**\n{json.dumps(BATCH_VERIFICATION_SCHEMA, indent=2)}\n"

    return prompt


# ── Batch verify function ──────────────────────────────────────────────────────

def batch_verify_crops(
    client,
    deployment: str,
    base_image: np.ndarray,
    existing_crops: List[ImageCropInfo],
    new_crops: List[ImageCropInfo],
    reasoning_effort: str = "low",
    grid_step_x: int = GRID_STEP_X,
    grid_step_y: int = GRID_STEP_Y,
) -> Tuple[List[ImageCropInfo], Dict]:
    """
    Verify all new crops in a SINGLE LLM call.

    Colors: 🟠 orange = existing confirmed, 🟣 purple = new candidates
    Returns: (verified_crops, stats)
    """
    if not new_crops:
        return [], {"llm_calls": 0, "time_s": 0, "results": []}

    h_img, w_img = base_image.shape[:2]
    image_size = (w_img, h_img)

    # Build overlay: grid + orange existing + purple candidates with index labels
    overlay = draw_grid_with_cell_numbers(
        base_image, step_x=grid_step_x, step_y=grid_step_y
    )

    # Draw existing crops in orange
    for crop in existing_crops:
        if crop.polygon:
            _draw_polygon(overlay, crop.polygon, ORANGE, 10)

    # Draw new candidates in purple with index labels
    for crop in new_crops:
        if crop.polygon:
            _draw_polygon(overlay, crop.polygon, PURPLE, 10)
            # Add index label near top-left of the box
            xs = [int(crop.polygon[i]) for i in range(0, len(crop.polygon), 2)]
            ys = [int(crop.polygon[i]) for i in range(1, len(crop.polygon), 2)]
            label_x, label_y = min(xs) + 10, min(ys) + 40
            cv2.putText(
                overlay, str(crop.index),
                (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, PURPLE, 4, cv2.LINE_AA,
            )

    # Build prompt
    existing_boxes = [
        {"index": c.index, "polygon": c.polygon} for c in existing_crops if c.polygon
    ]
    candidate_boxes = [
        {"index": c.index, "polygon": c.polygon} for c in new_crops if c.polygon
    ]
    prompt = build_batch_verification_prompt(
        existing_boxes, candidate_boxes, image_size,
        grid_step_x=grid_step_x, grid_step_y=grid_step_y,
    )

    # Also send individual crop images for better content assessment
    crop_images = []
    for crop in new_crops:
        if crop.polygon:
            ci = crop_polygon(base_image, crop.polygon)
            if ci is not None:
                crop_images.append(ci)

    # Send: overlay + all crop images
    images_to_send = [overlay] + crop_images

    print(f"  [batch_verify] Sending {len(new_crops)} candidates "
          f"({len(images_to_send)} images total) in 1 LLM call...")

    t0 = time.time()
    text, elapsed = responses_call_with_image(
        client, deployment, prompt, images_to_send,
        reasoning_effort=reasoning_effort,
    )
    elapsed = time.time() - t0
    print(f"  [batch_verify] LLM responded in {elapsed:.1f}s")

    result = safe_parse_json(text)
    crop_results = result.get("crops", [])

    # Process results
    verified = []
    crop_map = {c.index: c for c in new_crops}
    results_log = []

    for cr in crop_results:
        idx = cr.get("index")
        is_electrical = cr.get("is_electrical", False)
        issue = cr.get("issue", "")
        crop = crop_map.get(idx)
        if not crop:
            print(f"    ⚠️ Unknown crop index {idx} in response, skipping")
            continue

        ea = {k: cr.get(k, {}) for k in ["x1", "y1", "x2", "y2"]}
        dir_summary = " | ".join(
            f"{k}: {ea[k].get('direction', '?')} → {ea[k].get('corrected', '?')}"
            for k in ["x1", "y1", "x2", "y2"]
        )

        if not is_electrical:
            print(f"    ✗ Crop {idx}: NOT electrical — {issue}")
            results_log.append({"index": idx, "action": "dropped", "reason": issue})
            continue

        # Check if edges need correction
        all_none = all(
            ea[k].get("direction", "") == "none" for k in ["x1", "y1", "x2", "y2"]
        )

        if all_none:
            print(f"    ✓ Crop {idx}: valid, edges correct")
            verified.append(crop)
            results_log.append({"index": idx, "action": "accepted", "edges": "correct"})
        else:
            # Apply edge corrections
            polygon = crop.polygon
            xs = [polygon[i] for i in range(0, len(polygon), 2)]
            ys = [polygon[i] for i in range(1, len(polygon), 2)]

            x1_c = int(ea["x1"].get("corrected", min(xs)))
            y1_c = int(ea["y1"].get("corrected", min(ys)))
            x2_c = int(ea["x2"].get("corrected", max(xs)))
            y2_c = int(ea["y2"].get("corrected", max(ys)))

            # Clamp to image bounds
            x1_c = max(0, min(x1_c, w_img - 1))
            y1_c = max(0, min(y1_c, h_img - 1))
            x2_c = max(0, min(x2_c, w_img))
            y2_c = max(0, min(y2_c, h_img))

            if x1_c >= x2_c or y1_c >= y2_c:
                print(f"    ✗ Crop {idx}: invalid corrected edges, dropping")
                results_log.append({"index": idx, "action": "dropped", "reason": "invalid edges"})
                continue

            corrected_polygon = [
                float(x1_c), float(y1_c),
                float(x2_c), float(y1_c),
                float(x2_c), float(y2_c),
                float(x1_c), float(y2_c),
            ]
            crop.polygon = corrected_polygon
            crop.bbox = polygon_to_bbox(corrected_polygon, image_size, 0)
            print(f"    ✓ Crop {idx}: valid, edges corrected → [{x1_c},{y1_c},{x2_c},{y2_c}]")
            print(f"      Edges: {dir_summary}")
            verified.append(crop)
            results_log.append({
                "index": idx, "action": "corrected",
                "corrected_bbox": [x1_c, y1_c, x2_c, y2_c],
            })

    stats = {
        "llm_calls": 1,
        "time_s": round(elapsed, 3),
        "n_candidates": len(new_crops),
        "n_verified": len(verified),
        "n_dropped": len(new_crops) - len(verified),
        "results": results_log,
    }

    return verified, stats


# ── Test runner ────────────────────────────────────────────────────────────────

def run_test_page(
    page_num: int,
    output_root: Path,
    crops_root: Path,
    client,
    deployment: str,
):
    """Run detection → batch verification on a single page and report results."""
    print(f"\n{'='*60}")
    print(f"  PAGE {page_num}")
    print(f"{'='*60}")

    # Load page image
    base_img = cv2.imread(str(_page_image_path(output_root, page_num)))
    if base_img is None:
        print(f"  ⚠️ No page image found, skipping")
        return None
    h_img, w_img = base_img.shape[:2]
    image_size = (w_img, h_img)
    print(f"  Image: {w_img}×{h_img}")

    # Load existing DI crops
    existing_crops = load_existing_crops(output_root, crops_root, page_num)
    print(f"  Existing crops: {len(existing_crops)}")
    for c in existing_crops:
        print(f"    crop {c.index}: {c.polygon[:4]}...")

    # Content bbox
    content_bbox = detect_content_bbox(base_img)
    cx1, cy1, cx2, cy2 = content_bbox
    is_full = cx1 == 0 and cy1 == 0 and cx2 == w_img and cy2 == h_img

    # Coverage
    content_area = (cx2 - cx1) * (cy2 - cy1)
    existing_covered = sum(
        (b[2] - b[0]) * (b[3] - b[1])
        for crop in existing_crops
        if crop.polygon and (b := polygon_to_bbox(crop.polygon, image_size, 0))
    )
    coverage_pct = min(100.0, existing_covered / content_area * 100.0) if content_area > 0 else 0

    # Step 1: Detection (same as current pipeline)
    print(f"\n  --- Detection (effort=none) ---")
    t0 = time.time()
    result, det_elapsed = validate_page_with_gpt(
        client, deployment, output_root, page_num, existing_crops, image_size,
        detection_reasoning_effort="none",
        grid_step_x=GRID_STEP_X, grid_step_y=GRID_STEP_Y,
        content_bbox=content_bbox if not is_full else None,
        existing_coverage_pct=coverage_pct,
        use_grid=True,
    )
    status = result.get("status")
    missing = result.get("missing", [])
    drop_list = result.get("drop_non_electrical", [])
    print(f"  Detection: status={status}, {len(missing)} new regions, "
          f"{len(drop_list)} drops, {det_elapsed:.1f}s")

    # Apply drops
    if drop_list:
        drop_set = {item.get("index") for item in drop_list if isinstance(item, dict)}
        for item in drop_list:
            if isinstance(item, dict):
                print(f"    🗑️ Drop existing crop {item.get('index')}: {item.get('reason', '')}")
        existing_crops = [c for c in existing_crops if c.index not in drop_set]

    if status in ("complete", "no-images") or not missing:
        print(f"  No new regions to verify.")
        return {"page": page_num, "detection_time": det_elapsed, "verify_time": 0,
                "n_detected": 0, "n_verified": 0}

    # Parse missing polygons
    raw_polys = []
    for item in missing:
        if isinstance(item, dict):
            raw_polys.append(item.get("polygon", []))
        else:
            raw_polys.append(item)

    # Clip to content bbox
    if not is_full:
        clipped = []
        for p in raw_polys:
            b = polygon_to_bbox(p, image_size, 0)
            if b:
                x1c = max(b[0], cx1)
                y1c = max(b[1], cy1)
                x2c = min(b[2], cx2)
                y2c = min(b[3], cy2)
                if x2c > x1c and y2c > y1c:
                    clipped.append([float(x1c), float(y1c), float(x2c), float(y1c),
                                    float(x2c), float(y2c), float(x1c), float(y2c)])
        raw_polys = clipped

    # Filter overlaps
    filtered = _filter_new_polygons(raw_polys, existing_crops, image_size)
    n_detected = len(filtered)
    print(f"  After filtering: {n_detected} new candidate(s)")

    if not filtered:
        return {"page": page_num, "detection_time": det_elapsed, "verify_time": 0,
                "n_detected": 0, "n_verified": 0}

    # Create ImageCropInfo objects for candidates
    start_idx = max((c.index for c in existing_crops), default=0)
    new_crops = []
    for i, poly in enumerate(filtered, start=1):
        idx = start_idx + i
        bbox = polygon_to_bbox(poly, image_size, 0)
        new_crops.append(ImageCropInfo(
            page_num=page_num,
            index=idx,
            image_path=Path("dummy"),
            coords_path=None,
            polygon=list(poly),
            bbox=bbox,
            is_existing=False,
        ))

    for c in new_crops:
        xs = [c.polygon[j] for j in range(0, len(c.polygon), 2)]
        ys = [c.polygon[j] for j in range(1, len(c.polygon), 2)]
        print(f"    candidate {c.index}: [{int(min(xs))},{int(min(ys))},{int(max(xs))},{int(max(ys))}]")

    # Step 2: Batch verification (NEW approach)
    print(f"\n  --- Batch Verification (effort=low, {n_detected} candidates) ---")
    verified, stats = batch_verify_crops(
        client, deployment, base_img,
        existing_crops, new_crops,
        reasoning_effort="low",
        grid_step_x=GRID_STEP_X, grid_step_y=GRID_STEP_Y,
    )

    print(f"\n  Result: {stats['n_verified']} verified, {stats['n_dropped']} dropped, "
          f"{stats['time_s']}s (1 LLM call)")

    return {
        "page": page_num,
        "detection_time": round(det_elapsed, 3),
        "verify_time": stats["time_s"],
        "total_time": round(det_elapsed + stats["time_s"], 3),
        "n_detected": n_detected,
        "n_verified": stats["n_verified"],
        "n_dropped": stats["n_dropped"],
        "results": stats["results"],
    }


def main():
    parser = argparse.ArgumentParser(description="Test batch crop verification")
    parser.add_argument("--pages", type=int, nargs="+", default=[4, 7, 8],
                        help="Page numbers to test (default: 4 7 8)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_root = project_root / "outputs" / "di_detection"
    crops_root = output_root  # DI crops are stored under di_detection/pageN/images/

    print("Connecting to Azure OpenAI...")
    client, deployment = make_client()
    print(f"  deployment: {deployment}")

    all_results = []
    for page_num in args.pages:
        result = run_test_page(page_num, output_root, crops_root, client, deployment)
        if result:
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    total_det = sum(r["detection_time"] for r in all_results)
    total_ver = sum(r["verify_time"] for r in all_results)
    total_det_n = sum(r["n_detected"] for r in all_results)
    total_ver_n = sum(r["n_verified"] for r in all_results)
    print(f"  Pages tested: {len(all_results)}")
    print(f"  Total detected: {total_det_n} candidates")
    print(f"  Total verified: {total_ver_n} accepted")
    print(f"  Total dropped: {total_det_n - total_ver_n}")
    print(f"  Detection time: {total_det:.1f}s ({len(all_results)} calls)")
    print(f"  Verification time: {total_ver:.1f}s ({len(all_results)} calls, batch)")
    print(f"  Total LLM calls: {len(all_results) * 2} "
          f"(vs old: {len(all_results)} + {total_det_n}×1-3 = "
          f"{len(all_results) + total_det_n}-{len(all_results) + total_det_n*3})")


if __name__ == "__main__":
    main()
