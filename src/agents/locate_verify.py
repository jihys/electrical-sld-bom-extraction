"""Locate + Verify loop agent.

Implements the iterative locate→verify pattern for panel bbox detection:
  1. Compose locate input image → LLM locates panel bbox
  2. Compose verify input image → LLM validates edges
  3. If invalid, apply corrections → loop back
  4. Oscillation detection to prevent infinite loops

Also provides locate_and_verify_batch() which sends ONE locate-all LLM call
for all panels on a page (matching the 05b_panel_area_cropping_batch approach),
reducing locate calls from N panels to 1 per page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2

from .llm_caller import call_llm, parse_json, sanitize_bbox
from .prompts import (
    build_locate_prompt, build_verify_prompt,
    build_locate_all_prompt, build_verify_all_prompt,
)
from ..models.panel import PanelCrop
from ..tools.image_tools import (
    compose_locate_input, compose_verify_input,
    compose_locate_all_input, compose_verify_all_overlay,
)
from ..tools.geometry_tools import bbox_contains, bbox_area


# ──────────────────────────────────────────────────────────────────────────────
# Locate → Verify loop
# ──────────────────────────────────────────────────────────────────────────────

def locate_and_verify_panel(
    llm_client,
    deployment: str,
    page_img_path: str,
    panel_name: str,
    name_bbox: List[int],
    other_panels: Dict[str, List[int]],
    output_dir: str,
    grid_size: int = 120,
    max_tries: int = 10,
) -> Optional[PanelCrop]:
    """Run the locate→verify loop for a single panel.

    Returns a PanelCrop with the verified bbox, or None if all attempts fail.
    """
    img = cv2.imread(page_img_path)
    h, w = img.shape[:2]

    AXES = ["x1", "y1", "x2", "y2"]
    axis_history: Dict[str, List[int]] = {a: [] for a in AXES}

    best_bbox: Optional[List[int]] = None
    best_score: int = 5  # number of non-ok edges (lower is better)

    corrected_bbox: Optional[List[int]] = None
    last_failed_bbox: Optional[List[int]] = None

    for attempt in range(1, max_tries + 1):
        # ── Step 1: Determine bbox ──
        if corrected_bbox is not None:
            bbox = corrected_bbox
            corrected_bbox = None
        else:
            # Compose locate input
            locate_path = f"{output_dir}/locate_{_safe(panel_name)}_v{attempt}.png"
            compose_locate_input(
                page_img_path, name_bbox, panel_name,
                other_panels, locate_path, grid_size,
            )

            prompt = build_locate_prompt(panel_name, w, h, grid_size)
            images = [locate_path]
            if last_failed_bbox is not None:
                # Show failed bbox as red overlay for context
                _draw_failed_bbox(locate_path, last_failed_bbox)

            raw = call_llm(
                llm_client, deployment, prompt,
                image_paths=images,
                label=f"locate {panel_name} v{attempt}",
            )
            loc = parse_json(raw)
            if not loc or not loc.get("found", False):
                print(f"  [{panel_name}] attempt {attempt}: not found")
                continue

            bbox = sanitize_bbox(loc.get("bbox"), w, h)
            if not bbox:
                continue

        # ── Step 2: Basic validation ──
        area_ratio = bbox_area(tuple(bbox)) / (w * h) * 100
        if not bbox_contains(tuple(bbox), tuple(name_bbox)):
            print(f"  [{panel_name}] attempt {attempt}: bbox doesn't contain name bbox")
            continue
        if area_ratio > 80:
            print(f"  [{panel_name}] attempt {attempt}: area too large ({area_ratio:.1f}%)")
            continue

        # ── Step 3: Verify ──
        verify_path = f"{output_dir}/verify_{_safe(panel_name)}_v{attempt}.png"
        compose_verify_input(
            page_img_path, bbox, name_bbox, panel_name,
            other_panels, verify_path, grid_size,
        )

        verify_prompt = build_verify_prompt(bbox, w, h)
        ver_raw = call_llm(
            llm_client, deployment, verify_prompt,
            image_paths=[verify_path],
            label=f"verify {panel_name} v{attempt}",
        )
        ver = parse_json(ver_raw)

        if ver and ver.get("valid"):
            return PanelCrop(
                panel_name=panel_name,
                bbox=tuple(bbox),
                confidence=0.95,
                verified_by="llm_verify",
                status="verified",
                verify_attempts=attempt,
            )

        # ── Step 4: Process corrections ──
        last_failed_bbox = bbox

        if ver:
            edges = ver.get("edges", {})
            non_ok = sum(1 for a in AXES if edges.get(a, {}).get("status", "ok") != "ok")
            if non_ok < best_score:
                best_score = non_ok
                best_bbox = bbox

            # Apply corrections with oscillation detection
            cb = ver.get("corrected_bbox")
            cb = sanitize_bbox(cb, w, h)
            if cb and bbox_contains(tuple(cb), tuple(name_bbox)):
                cb_list = list(cb)
                for i, axis in enumerate(AXES):
                    axis_history[axis].append(cb_list[i])
                    hist = axis_history[axis]
                    if len(hist) >= 3:
                        a, b, c = hist[-3], hist[-2], hist[-1]
                        if (a < b and b > c) or (a > b and b < c):
                            cb_list[i] = bbox[i]  # freeze oscillating axis
                            print(f"    {axis}: oscillation detected → freeze at {bbox[i]}")
                corrected_bbox = sanitize_bbox(cb_list, w, h)

    # All attempts exhausted — use best bbox if available
    if best_bbox is not None:
        return PanelCrop(
            panel_name=panel_name,
            bbox=tuple(best_bbox),
            confidence=max(0.3, 1.0 - best_score * 0.15),
            verified_by="best_effort",
            status="unverified",
            verify_attempts=max_tries,
        )

    return None


def _safe(name: str) -> str:
    """Make a filename-safe version of a panel name."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _draw_failed_bbox(image_path: str, bbox: List[int]):
    """Draw a red 'FAILED' bbox on an image in-place."""
    img = cv2.imread(image_path)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, "FAILED", (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(image_path, img)


# ──────────────────────────────────────────────────────────────────────────────
# Batch locate → verify-all loop  (05b approach)
# ──────────────────────────────────────────────────────────────────────────────

def locate_and_verify_batch(
    llm_client,
    deployment: str,
    page_img_path: str,
    all_name_bboxes: Dict[str, Optional[List[int]]],
    output_dir: str,
    guide_image_paths: Optional[List[str]] = None,
    grid_size: int = 120,
    max_tries: int = 10,
) -> List[PanelCrop]:
    """Locate ALL panels in one LLM call, then verify all pending panels per round.

    This is the batch approach from NB05b: instead of N separate locate calls
    (one per panel), we send ONE locate-all call for the entire page, then verify
    all panels simultaneously each round.

    Parameters
    ----------
    all_name_bboxes : {panel_name: [x1,y1,x2,y2] or None}
        Name-label bboxes in full-page pixel coordinates.
    guide_image_paths : optional list of guide image file paths
        Prepended to LLM image inputs (e.g. panel_box_explanation.png).

    Returns
    -------
    List[PanelCrop] — one entry per successfully located panel.
    """
    img = cv2.imread(page_img_path)
    h, w = img.shape[:2]
    AXES = ["x1", "y1", "x2", "y2"]
    guides = list(guide_image_paths or [])
    has_guide = bool(guides)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_panel_names = list(all_name_bboxes.keys())
    if not all_panel_names:
        print("  [batch] no panels — skip")
        return []

    # ── Step 1: Locate ALL panels in one LLM call ──────────────────────────
    locate_img_path = str(out / "locate_all.png")
    compose_locate_all_input(page_img_path, all_name_bboxes, locate_img_path, grid_size)

    locate_prompt = build_locate_all_prompt(all_panel_names, w, h, grid_size, has_guide)
    loc_raw = call_llm(
        llm_client, deployment, locate_prompt,
        image_paths=guides + [locate_img_path],
        label=f"locate ALL {len(all_panel_names)} panels",
    )
    loc = parse_json(loc_raw)
    if not loc or "panels" not in loc:
        print("  [batch] locate-all: no valid JSON response")
        return []

    located_bboxes: Dict[str, Optional[List[int]]] = {}
    for entry in loc.get("panels", []):
        name = entry.get("name", "").strip()
        raw_bbox = entry.get("bbox")
        if entry.get("found") and raw_bbox:
            b = sanitize_bbox(raw_bbox, w, h)
            matched = all_name_bboxes.get(name)
            # If name bbox exists, validate containment; otherwise accept any valid bbox
            if b and (not matched or bbox_contains(tuple(b), tuple(matched))):
                located_bboxes[name] = b
                print(f"  [{name}] locate-all: bbox={b}")
            else:
                print(f"  [{name}] locate-all: bbox invalid or doesn't contain name bbox — skip")
                located_bboxes[name] = None
        else:
            print(f"  [{name}] locate-all: not found")
            located_bboxes[name] = None

    # ── Build initial per-panel state ──────────────────────────────────────
    panel_states: Dict[str, dict] = {}
    for panel_name in all_panel_names:
        initial_bbox = located_bboxes.get(panel_name)
        if initial_bbox is None:
            continue
        panel_states[panel_name] = {
            "bbox": initial_bbox,
            "best_bbox": None,
            "best_score": 5,
            "corrected_bbox": None,
            "axis_history": {a: [] for a in AXES},
            "ok": False,
            "panel_crop": None,
        }

    # ── Step 2: Verify-all loop ────────────────────────────────────────────
    for attempt in range(1, max_tries + 1):
        pending = [n for n, s in panel_states.items() if not s["ok"]]
        if not pending:
            break

        # Apply pending corrections
        for name in pending:
            state = panel_states[name]
            if state["corrected_bbox"] is not None:
                state["bbox"] = state["corrected_bbox"]
                state["corrected_bbox"] = None
                print(f"  [{name}|attempt {attempt}] using corrected_bbox: {state['bbox']}")

        # Reject oversized panels
        still_pending = []
        for name in pending:
            bbox = panel_states[name]["bbox"]
            area_ratio = bbox_area(tuple(bbox)) / (w * h) * 100
            if area_ratio > 80:
                print(f"  [{name}|attempt {attempt}] REJECT: area too large ({area_ratio:.1f}%)")
            else:
                still_pending.append(name)
        pending = still_pending
        if not pending:
            break

        # Build overlay + individual crops
        panel_bboxes_for_overlay = {n: panel_states[n]["bbox"] for n in pending}
        overlay_path = str(out / f"verify_all_a{attempt}.png")
        compose_verify_all_overlay(
            page_img_path, panel_bboxes_for_overlay, all_name_bboxes,
            overlay_path, grid_size,
        )

        crop_paths = []
        for name in pending:
            bbox = panel_states[name]["bbox"]
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            panel_states[name]["panel_crop"] = crop
            cp = str(out / f"crop_{_safe(name)}_a{attempt}.png")
            cv2.imwrite(cp, crop)
            crop_paths.append(cp)

        panels_info = [{"name": n, "bbox": panel_states[n]["bbox"]} for n in pending]
        verify_prompt = build_verify_all_prompt(panels_info, w, h, has_guide)

        ver_raw = call_llm(
            llm_client, deployment, verify_prompt,
            image_paths=guides + [overlay_path] + crop_paths,
            label=f"verify ALL {len(pending)} panels (attempt {attempt}/{max_tries})",
        )
        ver = parse_json(ver_raw)
        if not ver or "panels" not in ver:
            print(f"  [attempt {attempt}] batch verify: no valid JSON — skip round")
            continue

        for panel_result in ver.get("panels", []):
            name = panel_result.get("name", "").strip()
            if name not in panel_states:
                continue
            state = panel_states[name]
            matched_bbox = all_name_bboxes.get(name)
            bbox = state["bbox"]
            valid = panel_result.get("valid")
            print(f"  [{name}|attempt {attempt}] verify: valid={valid}")

            if valid:
                state["ok"] = True
                continue

            edges = panel_result.get("edges", {})
            non_ok_count = sum(1 for a in AXES if edges.get(a, {}).get("status", "ok") != "ok")
            if non_ok_count < state["best_score"]:
                state["best_score"] = non_ok_count
                state["best_bbox"] = bbox
                print(f"  [{name}|attempt {attempt}] best_bbox updated (score={non_ok_count}): {bbox}")

            cb = sanitize_bbox(panel_result.get("corrected_bbox"), w, h)
            if cb and matched_bbox and bbox_contains(tuple(cb), tuple(matched_bbox)):
                cb_list = list(cb)
                current_vals = list(bbox)
                for i, axis_name in enumerate(AXES):
                    state["axis_history"][axis_name].append(cb_list[i])
                    hist = state["axis_history"][axis_name]
                    if len(hist) >= 3:
                        a_v, b_v, c_v = hist[-3], hist[-2], hist[-1]
                        if (a_v < b_v and b_v > c_v) or (a_v > b_v and b_v < c_v):
                            cb_list[i] = current_vals[i]
                            print(f"    {name} {axis_name}: oscillation detected → freeze at {current_vals[i]}")
                corrected = sanitize_bbox(cb_list, w, h)
                if corrected:
                    state["corrected_bbox"] = corrected
                    print(f"  [{name}|attempt {attempt}] corrected_bbox: {corrected}")
            else:
                print(f"  [{name}|attempt {attempt}] no usable corrected_bbox")

    # ── Collect results ────────────────────────────────────────────────────
    results: List[PanelCrop] = []
    for panel_name, state in panel_states.items():
        bbox = state["bbox"]
        panel_crop = state["panel_crop"]

        if not state["ok"]:
            if state["best_bbox"] is not None:
                bbox = state["best_bbox"]
                panel_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                print(f"  [{panel_name}] using best_bbox (score={state['best_score']}): {bbox}")
            else:
                print(f"  [{panel_name}] FAILED — not saved")
                continue
        elif panel_crop is None:
            panel_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        results.append(PanelCrop(
            panel_name=panel_name,
            bbox=tuple(bbox),
            confidence=0.95 if state["ok"] else max(0.3, 1.0 - state["best_score"] * 0.15),
            verified_by="llm_verify_batch",
            status="verified" if state["ok"] else "unverified",
            verify_attempts=max_tries,
        ))

    return results
