"""
pipeline.py
-----------
Full detection + verification pipeline.
All functions accept explicit path/config parameters (no module-level globals).
Extracted/adapted from 02_missing_image_detection.ipynb.

Key addition vs notebook 02:
- verify_and_refine_new_crops() returns (verified_crops, stats) where
  stats = {"iterations_per_crop": [...], "total_iterations": N}
- run_page_pipeline() is a new single-call entry point that returns all stats.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
from openai import AzureOpenAI, NotFoundError

from .data_types import (
    ImageCropInfo,
    _normalize_polygon,
    _parse_crop_index,
    bbox_from_polygon,
    bbox_intersect,
    polygon_from_bbox,
    collect_polygons,
    crop_polygon,
    draw_overlay,
    polygon_to_bbox,
    read_polygon_txt,
    write_polygon_txt,
)
from .image_utils import (
    GRID_STEP_X,
    GRID_STEP_Y,
    detect_content_bbox,
    draw_grid_with_cell_numbers,
    draw_polygon,
    hex_to_bgr,
)
from .llm_client import (
    DIRECTIONAL_VERIFICATION_SCHEMA,
    _build_detection_prompt,
    _build_verification_prompt_v4,
    check_di_crop,
    responses_call,
    responses_call_with_image,
    safe_parse_json,
)
from .region_utils import _whitefill_polygons


# ── Bbox helpers ───────────────────────────────────────────────────────────────

def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _intersection_area(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> int:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if left >= right or top >= bottom:
        return 0
    return (right - left) * (bottom - top)


def _contains_bbox(
    outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _filter_new_polygons(
    new_polygons: List[Sequence[float]],
    existing_crops: List[ImageCropInfo],
    image_size: Tuple[int, int],
) -> List[List[float]]:
    existing_bboxes: List[Tuple[int, int, int, int]] = []
    for crop in existing_crops:
        bbox = crop.bbox or (polygon_to_bbox(crop.polygon, image_size, 0) if crop.polygon else None)
        if bbox:
            existing_bboxes.append(bbox)

    candidates: List[Tuple[List[float], Tuple[int, int, int, int], int]] = []
    for polygon in new_polygons:
        norm = _normalize_polygon(polygon)
        if not norm:
            continue
        bbox = polygon_to_bbox(norm, image_size, 0)
        if not bbox:
            continue
        # Always rectangularize: convert any polygon shape to axis-aligned bbox rectangle
        x1, y1, x2, y2 = bbox
        rect = [float(x1), float(y1), float(x2), float(y1),
                float(x2), float(y2), float(x1), float(y2)]
        if any(_intersection_area(existing, bbox) > 0 for existing in existing_bboxes):
            continue
        if any(_contains_bbox(existing, bbox) for existing in existing_bboxes):
            continue
        candidates.append((rect, bbox, _bbox_area(bbox)))

    candidates.sort(key=lambda item: item[2], reverse=True)
    kept: List[List[float]] = []
    kept_bboxes: List[Tuple[int, int, int, int]] = []
    for polygon, bbox, _ in candidates:
        if any(_intersection_area(kb, bbox) > 0 for kb in kept_bboxes):
            continue
        kept.append(polygon)
        kept_bboxes.append(bbox)
    return kept


# ── Full-page helpers ──────────────────────────────────────────────────────────

def full_page_polygon(image_size: Tuple[int, int]) -> List[float]:
    width, height = image_size
    return [0.0, 0.0, float(width), 0.0, float(width), float(height), 0.0, float(height)]


def is_full_page_polygon(
    polygon: Sequence[float], image_size: Tuple[int, int], tolerance: int = 0
) -> bool:
    bbox = polygon_to_bbox(polygon, image_size, 0)
    if not bbox:
        return False
    left, top, right, bottom = bbox
    width, height = image_size
    return (
        left <= tolerance
        and top <= tolerance
        and right >= width - tolerance
        and bottom >= height - tolerance
    )


# ── Path helpers ───────────────────────────────────────────────────────────────

def _page_image_path(output_root: Path, page_num: int) -> Path:
    primary = output_root / f"page{page_num}" / f"page{page_num}.png"
    if primary.is_file():
        return primary
    return output_root / f"page{page_num}.png"


def _page_images_dir(output_root: Path, page_num: int) -> Path:
    return output_root / f"page{page_num}" / "images"


def _resolve_llm_input_image_path(output_root: Path, page_num: int) -> Path:
    page_dir = output_root / f"page{page_num}"
    images_dir = page_dir / "images"
    names = [
        f"images_overaly_page{page_num}.png",
        f"images_overaly_page{page_num:02d}.png",
        f"images_overlay_page{page_num}.png",
        f"images_overlay_page{page_num:02d}.png",
    ]
    for name in names:
        for d in [images_dir, page_dir]:
            p = d / name
            if p.exists():
                return p
    return _page_image_path(output_root, page_num)


def _page_output_dir(crops_root: Path, page_num: int) -> Path:
    return crops_root / f"page{page_num}"


def _output_crop_image_path(crops_root: Path, page_num: int, crop_index: int) -> Path:
    return _page_output_dir(crops_root, page_num) / f"image_{page_num}_{crop_index}.png"


def _output_crop_txt_path(crops_root: Path, page_num: int, crop_index: int) -> Path:
    return _page_output_dir(crops_root, page_num) / f"image_{page_num}_{crop_index}.txt"


def _output_overlay_path(crops_root: Path, page_num: int, crop_index: int) -> Path:
    return _page_output_dir(crops_root, page_num) / f"image_overlay_{page_num}_{crop_index}.png"


def _output_overall_overlay_path(crops_root: Path, page_num: int, crop_index: int) -> Path:
    return _page_output_dir(crops_root, page_num) / f"image_overlay_{page_num}_{crop_index}_overall.png"


def _output_iteration_image_path(
    crops_temp_root: Path, page_num: int, crop_index: int, iteration: int, is_final: bool = False
) -> Path:
    suffix = "final" if is_final else f"iter{iteration}"
    return _page_output_dir(crops_temp_root, page_num) / f"image_{page_num}_{crop_index}_{suffix}.png"


def _output_iteration_overlay_path(
    crops_temp_root: Path, page_num: int, crop_index: int, iteration: int
) -> Path:
    return (
        _page_output_dir(crops_temp_root, page_num)
        / f"image_{page_num}_{crop_index}_iter{iteration}_overlay.png"
    )


# ── Crop loading ───────────────────────────────────────────────────────────────

def load_existing_crops(
    output_root: Path, crops_root: Path, page_num: int
) -> List[ImageCropInfo]:
    """Load existing DI crops for this page always from output_root (first_image_detection_updated).

    Always loads fresh from output_root so every run starts from the original DI results.
    """
    crop_paths: List[Path] = []
    images_dir = _page_images_dir(output_root, page_num)
    page_input_dir = output_root / f"page{page_num}"
    for d in [images_dir, page_input_dir]:
        if not d.exists():
            continue
        crop_paths.extend(sorted(d.glob(f"image_{page_num}_*.png")))
        crop_paths.extend(sorted(d.glob(f"image_{page_num:02d}_*.png")))
        crop_paths.extend(sorted(d.glob(f"images_{page_num}_*.png")))
        crop_paths.extend(sorted(d.glob(f"images_{page_num:02d}_*.png")))

    seen: set = set()
    unique_paths = []
    for path in crop_paths:
        if path not in seen:
            # Skip overlay images
            if "overlay" in path.stem.lower():
                continue
            seen.add(path)
            unique_paths.append(path)

    if not unique_paths:
        return []

    items: List[ImageCropInfo] = []
    base_img = cv2.imread(str(_page_image_path(output_root, page_num)))
    h_img, w_img = base_img.shape[:2]

    for crop_path in unique_paths:
        idx = _parse_crop_index(crop_path, page_num)
        if idx is None:
            continue
        coords_path = crop_path.with_suffix(".txt")
        polygon = None
        bbox = None
        if coords_path.exists():
            polygon = read_polygon_txt(coords_path)
        if polygon:
            bbox = polygon_to_bbox(polygon, (w_img, h_img), 0)
        items.append(
            ImageCropInfo(
                page_num=page_num,
                index=idx,
                image_path=crop_path,
                coords_path=coords_path if coords_path.exists() else None,
                polygon=polygon,
                bbox=bbox,
                is_existing=True,
            )
        )
    return items


def load_all_crops(
    output_root: Path, crops_root: Path, page_num: int
) -> List[ImageCropInfo]:
    """Load all crops from crops_root (existing DI + newly detected) after a pipeline run.

    Unlike load_existing_crops (which only reads from output_root/DI folder),
    this reads from crops_root where sync_existing_crops_to_output copies all crops.
    Use this for post-pipeline visualization and post-processing.
    """
    crops_dir = _page_output_dir(crops_root, page_num)
    if not crops_dir.exists():
        return []

    base_img = cv2.imread(str(_page_image_path(output_root, page_num)))
    if base_img is None:
        return []
    h_img, w_img = base_img.shape[:2]

    items: List[ImageCropInfo] = []
    for p in sorted(crops_dir.glob(f"image_{page_num}_*.png")):
        if "overlay" in p.stem.lower():
            continue
        try:
            idx = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        coords_path = p.with_suffix(".txt")
        polygon = read_polygon_txt(coords_path) if coords_path.exists() else None
        bbox = polygon_to_bbox(polygon, (w_img, h_img), 0) if polygon else None
        items.append(ImageCropInfo(
            page_num=page_num, index=idx,
            image_path=p, coords_path=coords_path if coords_path.exists() else None,
            polygon=polygon, bbox=bbox, is_existing=True,
        ))
    return items


def postprocess_crops(
    prod_reports: List[Dict],
    output_root: Path,
    crops_root: Path,
) -> Tuple[int, int]:
    """Clip each crop to its page content_bbox and optionally drop fully-contained crops.

    Returns (total_clipped, total_dropped).
    """
    content_bboxes = {r["page_num"]: tuple(r["content_bbox"])
                      for r in prod_reports if "content_bbox" in r}
    total_clipped = 0
    total_dropped = 0

    for r in prod_reports:
        page_num = r["page_num"]
        content_bbox = content_bboxes.get(page_num)

        crops = load_existing_crops(output_root, crops_root, page_num)
        if not crops:
            continue

        img_path = _page_image_path(output_root, page_num)
        base_img = cv2.imread(str(img_path))
        if base_img is None:
            print(f"  [Page {page_num}] base image not found at {img_path}, skipping")
            continue

        # 1) Clip each crop to content_bbox
        if content_bbox:
            for crop in crops:
                if not crop.polygon:
                    continue
                box = bbox_from_polygon(crop.polygon)
                clipped = bbox_intersect(box, content_bbox)
                if clipped is None:
                    print(f"  [Page {page_num}] Crop {crop.index}: no overlap with content bbox — skipping")
                    continue
                if clipped != box:
                    new_poly = polygon_from_bbox(*clipped)
                    crop.polygon = new_poly
                    x1, y1, x2, y2 = clipped
                    cropped_img = base_img[y1:y2, x1:x2]
                    if cropped_img.size > 0 and crop.image_path:
                        cv2.imwrite(str(crop.image_path), cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    if crop.coords_path:
                        write_polygon_txt(crop.coords_path, new_poly)
                    print(f"  [Page {page_num}] Crop {crop.index}: clipped {box} -> {clipped}")
                    total_clipped += 1

        # 2) Drop fully-contained crops
        valid_crops = [c for c in crops if c.polygon]
        boxes = [bbox_from_polygon(c.polygon) for c in valid_crops]
        to_drop = set()
        for i, (ci, bi) in enumerate(zip(valid_crops, boxes)):
            for j, (cj, bj) in enumerate(zip(valid_crops, boxes)):
                if i == j or i in to_drop or j in to_drop:
                    continue
                # Check if bi is fully contained within bj
                if bj[0] <= bi[0] and bj[1] <= bi[1] and bj[2] >= bi[2] and bj[3] >= bi[3]:
                    to_drop.add(i)
                    print(f"  [Page {page_num}] Crop {ci.index}: fully contained in crop {cj.index} — dropping")
        for i in to_drop:
            c = valid_crops[i]
            if c.image_path and Path(c.image_path).exists():
                Path(c.image_path).unlink()
            if c.coords_path and Path(c.coords_path).exists():
                Path(c.coords_path).unlink()
            total_dropped += 1

    print(f"\nPost-processing complete: {total_clipped} crops clipped, {total_dropped} crops dropped.")
    return total_clipped, total_dropped


# ── Detection ─────────────────────────────────────────────────────────────────

def validate_page_with_gpt(
    client: AzureOpenAI,
    deployment: str,
    output_root: Path,
    page_num: int,
    crops: List[ImageCropInfo],
    image_size: Tuple[int, int],
    detection_reasoning_effort: Optional[str] = None,
    grid_step_x: int = GRID_STEP_X,
    grid_step_y: int = GRID_STEP_Y,
    content_bbox: Optional[Tuple[int, int, int, int]] = None,
    existing_coverage_pct: Optional[float] = None,
    use_grid: bool = True,
) -> Tuple[Dict, float]:
    """Returns (parsed_result, elapsed_seconds)."""
    crop_boxes = [
        {"index": crop.index, "polygon": crop.polygon}
        for crop in crops if crop.polygon
    ]
    prompt = _build_detection_prompt(
        crop_boxes, image_size,
        content_bbox=content_bbox,
        existing_coverage_pct=existing_coverage_pct,
    )
    # Build an orange overlay: draw orange boxes for all existing confirmed crops.
    # This lets the LLM see what's already captured without hiding the underlying content.
    # The updated DI overlay (post crop-check fix) is used as the base so the
    # adjusted crop boundaries are reflected in the orange boxes.
    base_img_path = _page_image_path(output_root, page_num)
    page_img = cv2.imread(str(base_img_path))
    if page_img is None:
        raise FileNotFoundError(f"Page image not found: {base_img_path}")

    if crops:
        page_img = page_img.copy()
        for crop in crops:
            if crop.polygon:
                draw_polygon(page_img, crop.polygon, color=hex_to_bgr("#f97316"), width=10)

    try:
        text, elapsed = responses_call(
            client, deployment, prompt, page_img,
            reasoning_effort=detection_reasoning_effort,
            grid_step_x=grid_step_x, grid_step_y=grid_step_y,
            use_grid=use_grid,
        )
    except NotFoundError as exc:
        raise RuntimeError(
            f"Azure OpenAI returned 404. Check deployment name and endpoint. "
            f"Deployment={deployment}"
        ) from exc
    return safe_parse_json(text), elapsed


# ── Verification ───────────────────────────────────────────────────────────────

def verify_crop_with_llm(
    client: AzureOpenAI,
    deployment: str,
    base_image,
    polygon: Sequence[float],
    existing_crops: List[ImageCropInfo],
    pending_crops: List[ImageCropInfo],
    iteration: int,
    history: Optional[List[Dict]] = None,
    verification_reasoning_effort: Optional[str] = None,
    grid_step_x: int = GRID_STEP_X,
    grid_step_y: int = GRID_STEP_Y,
    crop_image=None,
    use_grid: bool = True,
    clipped_edges_hint: Optional[List[str]] = None,
) -> Tuple[Optional[List[float]], str, float]:
    """
    v4 Directional verification.
    Returns (corrected_polygon, issue, elapsed_seconds). If correct, returns (None, "", elapsed).
    crop_image: crop of the current polygon — sent alongside the overlay so the LLM
                can assess content type from the actual region, not just the full-page view.
    """
    if history is None:
        history = []

    crop_boxes = [
        {"index": crop.index, "polygon": crop.polygon}
        for crop in existing_crops if crop.polygon
    ]

    # Build overlay: orange=existing, green=pending, purple=current candidate
    h_img, w_img = base_image.shape[:2]
    if use_grid:
        grid_base = draw_grid_with_cell_numbers(base_image, step_x=grid_step_x, step_y=grid_step_y)
    else:
        grid_base = base_image.copy()
    overlay = grid_base.copy()

    for crop in existing_crops:
        if crop.polygon:
            draw_polygon(overlay, crop.polygon, color=hex_to_bgr("#f97316"), width=10)  # orange
    for crop in pending_crops:
        if crop.polygon:
            draw_polygon(overlay, crop.polygon, color=hex_to_bgr("#84cc16"), width=10)  # green
    draw_polygon(overlay, polygon, color=hex_to_bgr("#9333ea"), width=10)  # purple

    has_pending = len(pending_crops) > 0
    image_size = (w_img, h_img)
    n_existing_ref = len(existing_crops)
    n_pending_ref = len(pending_crops)

    prompt = _build_verification_prompt_v4(
        crop_boxes, polygon, image_size, has_pending, history,
        grid_step_x=grid_step_x, grid_step_y=grid_step_y,
        use_grid=use_grid,
        crop_mode=False,
        n_existing=n_existing_ref,
        n_pending=n_pending_ref,
        clipped_edges_hint=clipped_edges_hint,
    )
    print(f"    Verification v4 iter {iteration + 1}... "
          f"(existing: {n_existing_ref}, pending: {n_pending_ref}, "
          f"history: {len(history)}, effort: {verification_reasoning_effort})")

    # Send overlay + raw crop of the current polygon.
    # The prompt describes two images: overlay (1st) + crop (2nd).
    # Crop from base_image (no whitefill) so the LLM sees actual content.
    if crop_image is None:
        crop_image = crop_polygon(base_image, polygon)
    images_to_send = [overlay, crop_image] if crop_image is not None else [overlay]
    elapsed = 0.0
    try:
        text, elapsed = responses_call_with_image(
            client, deployment, prompt, images_to_send,
            reasoning_effort=verification_reasoning_effort,
        )
        result = safe_parse_json(text)

        if result.get("is_correct"):
            print("      ✓ Verified as correct (all edges OK)")
            return (None, "", elapsed)

        issue = result.get("issue", "Unknown issue")
        ea = {k: result.get(k, {}) for k in ["x1", "y1", "x2", "y2"]}
        dir_summary = " | ".join(
            f"{k}: {ea[k].get('direction', '?')} → {ea[k].get('corrected', '?')}"
            for k in ["x1", "y1", "x2", "y2"]
        )
        print(f"      ✗ Issue: {issue}")
        print(f"        Edges: {dir_summary}")

        all_none = all(ea[k].get("direction", "") == "none" for k in ["x1", "y1", "x2", "y2"])
        if all_none:
            print("        → No edge correction (non-geometric issue)")
            return (None, issue, elapsed)

        xs_cur = [polygon[i] for i in range(0, len(polygon), 2)]
        ys_cur = [polygon[i] for i in range(1, len(polygon), 2)]
        x1_c = int(ea["x1"].get("corrected", min(xs_cur)))
        y1_c = int(ea["y1"].get("corrected", min(ys_cur)))
        x2_c = int(ea["x2"].get("corrected", max(xs_cur)))
        y2_c = int(ea["y2"].get("corrected", max(ys_cur)))

        x1_c = max(0, min(x1_c, w_img - 1))
        y1_c = max(0, min(y1_c, h_img - 1))
        x2_c = max(0, min(x2_c, w_img))
        y2_c = max(0, min(y2_c, h_img))

        if x1_c >= x2_c or y1_c >= y2_c:
            print(f"        ! Invalid corrected edges: x1={x1_c} x2={x2_c} y1={y1_c} y2={y2_c}")
            fallback = result.get("corrected_polygon")
            if fallback:
                normalized = _normalize_polygon(fallback)
                if normalized:
                    # Convert to bounding rectangle to avoid non-rectangular polygons
                    fb_xs = normalized[0::2]
                    fb_ys = normalized[1::2]
                    fb_rect = [
                        float(min(fb_xs)), float(min(fb_ys)),
                        float(max(fb_xs)), float(min(fb_ys)),
                        float(max(fb_xs)), float(max(fb_ys)),
                        float(min(fb_xs)), float(max(fb_ys)),
                    ]
                    print("        → Using fallback corrected_polygon (rectangularized)")
                    return (fb_rect, issue, elapsed)
            return (None, issue, elapsed)

        corrected_polygon = [
            float(x1_c), float(y1_c),
            float(x2_c), float(y1_c),
            float(x2_c), float(y2_c),
            float(x1_c), float(y2_c),
        ]
        enhanced_issue = f"{issue} [edges: {dir_summary}]"
        print(f"        → Corrected: [{x1_c}, {y1_c}, {x2_c}, {y2_c}]")
        return (corrected_polygon, enhanced_issue, elapsed)

    except Exception as exc:
        print(f"      ! Verification error: {exc}")
        return (None, f"Error: {exc}", elapsed)


# ── Output helpers ─────────────────────────────────────────────────────────────

def sync_existing_crops_to_output(
    output_root: Path, crops_root: Path, page_num: int, crops: List[ImageCropInfo]
) -> None:
    output_dir = _page_output_dir(crops_root, page_num)
    output_dir.mkdir(parents=True, exist_ok=True)
    for crop in crops:
        dest_image = _output_crop_image_path(crops_root, page_num, crop.index)
        if crop.image_path.resolve() != dest_image.resolve():
            shutil.copy2(crop.image_path, dest_image)
        crop.image_path = dest_image
        dest_coords = _output_crop_txt_path(crops_root, page_num, crop.index)
        if crop.coords_path and crop.coords_path.exists():
            if crop.coords_path.resolve() != dest_coords.resolve():
                shutil.copy2(crop.coords_path, dest_coords)
        elif crop.polygon:
            write_polygon_txt(dest_coords, crop.polygon)
        crop.coords_path = dest_coords


def append_new_crops(
    output_root: Path,
    crops_root: Path,
    page_num: int,
    polygons: List[Sequence[float]],
    start_idx: int,
) -> List[ImageCropInfo]:
    output_dir = _page_output_dir(crops_root, page_num)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: List[ImageCropInfo] = []
    base_image = cv2.imread(str(_page_image_path(output_root, page_num)))
    h, w = base_image.shape[:2]
    for offset, polygon in enumerate(polygons, start=1):
        cropped = crop_polygon(base_image, polygon)
        if cropped is None:
            continue
        idx = start_idx + offset
        dest_path = _output_crop_image_path(crops_root, page_num, idx)
        cv2.imwrite(str(dest_path), cropped, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        coords_path = _output_crop_txt_path(crops_root, page_num, idx)
        write_polygon_txt(coords_path, polygon)
        bbox = polygon_to_bbox(polygon, (w, h), 0)
        created.append(
            ImageCropInfo(
                page_num=page_num,
                index=idx,
                image_path=dest_path,
                coords_path=coords_path,
                polygon=list(polygon),
                bbox=bbox,
                is_existing=False,
            )
        )
    return created


def move_verified_crops_to_final(
    crops_root: Path,
    crops_temp_root: Path,
    page_num: int,
    verified_crops: List[ImageCropInfo],
) -> None:
    final_dir = _page_output_dir(crops_root, page_num)
    final_dir.mkdir(parents=True, exist_ok=True)
    for crop in verified_crops:
        if crop.is_existing:
            continue
        temp_image = _output_crop_image_path(crops_temp_root, page_num, crop.index)
        temp_coords = _output_crop_txt_path(crops_temp_root, page_num, crop.index)
        final_image = _output_crop_image_path(crops_root, page_num, crop.index)
        final_coords = _output_crop_txt_path(crops_root, page_num, crop.index)
        if temp_image.exists():
            shutil.move(str(temp_image), str(final_image))
            crop.image_path = final_image
        if temp_coords.exists():
            shutil.move(str(temp_coords), str(final_coords))
            crop.coords_path = final_coords


def write_overlays(
    output_root: Path, crops_root: Path, page_num: int, crops: List[ImageCropInfo], page_dpi: int = 300
) -> Optional[Path]:
    """Generate and save overlay images for final crops.

    Args:
        page_dpi: DPI used when rendering the page (for informational/logging purposes).
                  Ensures DPI-aware coordinate handling in overlay generation.
    """
    output_dir = _page_output_dir(crops_root, page_num)
    output_dir.mkdir(parents=True, exist_ok=True)
    polygons = collect_polygons(crops)
    if not polygons:
        return None
    overall_index = max([crop.index for crop in crops], default=len(polygons))
    base_image = cv2.imread(str(_page_image_path(output_root, page_num)))
    print(f"  [write_overlays] Page {page_num}: DPI={page_dpi}, image_shape={base_image.shape if base_image is not None else 'MISSING'}")
    overall = draw_overlay(base_image, polygons)
    overall_path = _output_overall_overlay_path(crops_root, page_num, overall_index)
    cv2.imwrite(str(overall_path), overall, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    for crop in crops:
        if not crop.polygon:
            continue
        overlay = draw_overlay(base_image, [crop.polygon])
        overlay_path = _output_overlay_path(crops_root, page_num, crop.index)
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return overall_path


# ── Verification loop ──────────────────────────────────────────────────────────

def verify_and_refine_new_crops(
    client: AzureOpenAI,
    deployment: str,
    output_root: Path,
    crops_root: Path,
    crops_temp_root: Path,
    page_num: int,
    new_crops: List[ImageCropInfo],
    existing_crops: List[ImageCropInfo],
    max_iterations: int = 3,
    verification_reasoning_effort: Optional[str] = None,
    grid_step_x: int = GRID_STEP_X,
    grid_step_y: int = GRID_STEP_Y,
    use_grid: bool = True,
) -> Tuple[List[ImageCropInfo], Dict]:
    """
    Verify and refine new crops with LLM (max N iterations each).

    Returns:
        (verified_crops, stats)
        stats = {
            "iterations_per_crop": [n1, n2, ...],  # iterations used per crop
            "total_iterations": N,
        }
    """
    if not new_crops:
        return new_crops, {"iterations_per_crop": [], "total_iterations": 0, "llm_calls": []}

    print(f"  Verifying {len(new_crops)} new crops (max {max_iterations} iters, "
          f"effort={verification_reasoning_effort})...")

    base_image = cv2.imread(str(_page_image_path(output_root, page_num)))
    h_img, w_img = base_image.shape[:2]

    # Build a white-filled base used only for saving crops to disk.
    # Confirmed existing regions are blanked so saved crops don't carry duplicate content.
    save_base = base_image.copy()
    for crop in existing_crops:
        if crop.polygon:
            xs = [int(crop.polygon[j]) for j in range(0, len(crop.polygon), 2)]
            ys = [int(crop.polygon[j]) for j in range(1, len(crop.polygon), 2)]
            save_base[max(0, min(ys)):min(h_img, max(ys)),
                      max(0, min(xs)):min(w_img, max(xs))] = 255

    verified_crops: List[ImageCropInfo] = []
    iterations_per_crop: List[int] = []
    llm_calls: List[Dict] = []

    for crop_idx, crop in enumerate(new_crops):
        if crop.is_existing:
            verified_crops.append(crop)
            existing_crops.append(crop)
            iterations_per_crop.append(0)
            continue

        print(f"  [{crop_idx + 1}/{len(new_crops)}] Verifying crop index {crop.index}...")
        current_polygon = crop.polygon
        pending_crops = new_crops[crop_idx + 1:]
        history: List[Dict] = []
        iterations_used = 0
        rejected = False

        for iteration in range(max_iterations):
            iterations_used = iteration + 1

            # Overlay for debug: orange=existing, green=pending, purple=current candidate
            if use_grid:
                overlay_debug = draw_grid_with_cell_numbers(
                    base_image, step_x=grid_step_x, step_y=grid_step_y
                )
            else:
                overlay_debug = base_image.copy()
            for ec in existing_crops:
                if ec.polygon:
                    draw_polygon(overlay_debug, ec.polygon, hex_to_bgr("#f97316"), 10)  # orange
            for pc in pending_crops:
                if pc.polygon:
                    draw_polygon(overlay_debug, pc.polygon, hex_to_bgr("#84cc16"), 10)  # green
            draw_polygon(overlay_debug, current_polygon, hex_to_bgr("#9333ea"), 10)  # purple
            overlay_path = _output_iteration_overlay_path(
                crops_temp_root, page_num, crop.index, iteration
            )
            cv2.imwrite(str(overlay_path), overlay_debug, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            corrected, issue, vrf_elapsed = verify_crop_with_llm(
                client, deployment, base_image, current_polygon,
                existing_crops, pending_crops, iteration, history,
                verification_reasoning_effort=verification_reasoning_effort,
                grid_step_x=grid_step_x, grid_step_y=grid_step_y,
                use_grid=use_grid,
            )
            llm_calls.append({
                "task": "verification",
                "crop_index": crop.index,
                "iteration": iteration + 1,
                "reasoning_effort": verification_reasoning_effort,
                "time_s": round(vrf_elapsed, 3),
            })

            if corrected is None and not issue:
                # Verified correct
                break

            if issue:
                history.append({"polygon": list(current_polygon), "issue": issue})

            if corrected is None:
                if issue:
                    rejected = True
                    print("      ! Rejected (non-geometric issue), dropping crop")
                else:
                    print("      ! No correction provided, stopping iterations")
                break

            current_polygon = corrected

            iter_path = _output_iteration_image_path(
                crops_temp_root, page_num, crop.index, iteration + 1
            )
            cropped = crop_polygon(save_base, current_polygon)
            if cropped is not None:
                cv2.imwrite(str(iter_path), cropped, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            # Update temp files
            temp_path = _output_crop_image_path(crops_temp_root, page_num, crop.index)
            temp_coords = _output_crop_txt_path(crops_temp_root, page_num, crop.index)
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            if cropped is not None:
                cv2.imwrite(str(temp_path), cropped, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            write_polygon_txt(temp_coords, current_polygon)

        if rejected:
            iterations_per_crop.append(iterations_used)
            print(f"      ✗ Dropped in {iterations_used} iteration(s) (not a valid region)")
            continue

        final_path = _output_iteration_image_path(
            crops_temp_root, page_num, crop.index, 0, is_final=True
        )
        final_cropped = crop_polygon(save_base, current_polygon)
        if final_cropped is not None:
            cv2.imwrite(str(final_path), final_cropped, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        crop.polygon = current_polygon
        crop.bbox = polygon_to_bbox(current_polygon, (w_img, h_img), 0)
        verified_crops.append(crop)
        existing_crops.append(crop)
        iterations_per_crop.append(iterations_used)
        print(f"      ✓ Done in {iterations_used} iteration(s)")

    stats = {
        "iterations_per_crop": iterations_per_crop,
        "total_iterations": sum(iterations_per_crop),
        "llm_calls": llm_calls,
    }
    return verified_crops, stats


# ── Post-processing: content crop ──────────────────────────────────────────────

def _apply_content_crop_to_page(crops_root: Path, page_num: int) -> List[str]:
    """Crop outer frame & title block from all image_*.png in a page directory.

    Also applies the same crop to overlay images and updates .txt polygon files.
    Returns list of changed file names.
    """
    page_dir = _page_output_dir(crops_root, page_num)
    if not page_dir.exists():
        return []

    changed: List[str] = []
    for img_path in sorted(page_dir.glob("image_*.png")):
        if "overlay" in img_path.name:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        x1, y1, x2, y2 = detect_content_bbox(img)

        if x1 == 0 and y1 == 0 and x2 == w and y2 == h:
            continue  # no frame detected

        print(f"  [crop] {img_path.name}  {w}x{h} → ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}")
        cv2.imwrite(str(img_path), img[y1:y2, x1:x2], [cv2.IMWRITE_PNG_COMPRESSION, 1])

        # Apply same crop to overlay images
        stem = img_path.stem  # e.g. 'image_2_1'
        suffix = stem[6:]     # e.g. '2_1'
        for pattern in [f"image_overlay_{suffix}.png", f"image_overlay_{suffix}_overall.png"]:
            for ov_path in page_dir.glob(pattern):
                ov = cv2.imread(str(ov_path))
                if ov is not None and ov.shape[:2] == (h, w):
                    cv2.imwrite(str(ov_path), ov[y1:y2, x1:x2], [cv2.IMWRITE_PNG_COMPRESSION, 1])

        # Update .txt polygon — coordinates must stay in full-page pixel space.
        # x1,y1,x2,y2 are crop-local; recover the original full-page origin from
        # the existing txt so the stored polygon remains in full-page coordinates.
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            orig_polygon = read_polygon_txt(txt_path)
            ox = int(min(orig_polygon[0::2]))
            oy = int(min(orig_polygon[1::2]))
            write_polygon_txt(txt_path, [
                float(ox + x1), float(oy + y1),
                float(ox + x2), float(oy + y1),
                float(ox + x2), float(oy + y2),
                float(ox + x1), float(oy + y2),
            ])

        changed.append(img_path.name)

    return changed


# ── DI crop check (clipping + excess area) ────────────────────────────────────

def _regenerate_di_overlay(output_root: Path, page_num: int, crops: List[ImageCropInfo]) -> None:
    """Redraw the DI overlay image at output_root with current crop polygons.

    The DI overlay (images_overaly_page{N}.png) is the full page with orange boxes.
    After correcting a DI crop boundary, regenerate it so downstream steps see the
    updated orange box positions.
    """
    page_dir = output_root / f"page{page_num}"
    images_dir = page_dir / "images"
    # Try to find the existing overlay to overwrite it
    overlay_names = [
        f"images_overaly_page{page_num}.png",
        f"images_overaly_page{page_num:02d}.png",
        f"images_overlay_page{page_num}.png",
        f"images_overlay_page{page_num:02d}.png",
    ]
    overlay_path: Optional[Path] = None
    for name in overlay_names:
        for d in [images_dir, page_dir]:
            p = d / name
            if p.exists():
                overlay_path = p
                break
        if overlay_path:
            break
    if overlay_path is None:
        # Create a new one using the first name convention
        images_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = images_dir / f"images_overaly_page{page_num}.png"

    base_img = cv2.imread(str(_page_image_path(output_root, page_num)))
    if base_img is None:
        return
    overlay_img = base_img.copy()
    for crop in crops:
        if crop.polygon:
            draw_polygon(overlay_img, crop.polygon, hex_to_bgr("#f97316"), 10)
    cv2.imwrite(str(overlay_path), overlay_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def verify_di_crops(
    client: AzureOpenAI,
    deployment: str,
    output_root: Path,
    page_num: int,
    crops: List[ImageCropInfo],
    reasoning_effort: Optional[str] = None,
    grid_step_x: int = GRID_STEP_X,
    grid_step_y: int = GRID_STEP_Y,
    use_grid: bool = True,
    max_fix_iterations: int = 3,
) -> Tuple[List[Dict], float]:
    """Crop check: verify boundary placement for each DI-detected crop and fix iteratively.

    Sends both the full-page DI overlay image AND the individual crop image to the LLM.
    Checks for clipping (content cut off) and excess area (unnecessary margins).
    If an issue is found, loops: apply fix → re-check → fix again if still problematic,
    up to max_fix_iterations times per crop.

    Returns:
        (results, total_time_s)
        results: list of per-crop dicts:
        {
            "index": int,
            "image_path": str,
            "is_clipped": bool,          # True if still clipped after all fix attempts
            "clipped_edges": List[str],
            "was_fixed": bool,
            "fix_iterations": int,        # number of fix attempts made
            "note": str,
            "check_time_s": float,
            "fix_total_time_s": float,    # cumulative fix + re-check time
        }
    """
    import time as _time

    # Load (or build) the full-page DI overlay image once
    overlay_image_path = _resolve_llm_input_image_path(output_root, page_num)
    page_overlay_img = cv2.imread(str(overlay_image_path))
    base_img = cv2.imread(str(_page_image_path(output_root, page_num)))
    if base_img is None:
        return [], 0.0
    if page_overlay_img is None:
        page_overlay_img = base_img.copy()
        for crop in crops:
            if crop.polygon:
                draw_polygon(page_overlay_img, crop.polygon, hex_to_bgr("#f97316"), 10)

    h_img, w_img = base_img.shape[:2]
    results = []
    total_time = 0.0

    for i, crop in enumerate(crops):
        if crop.image_path is None or not crop.image_path.exists():
            continue
        crop_img = cv2.imread(str(crop.image_path))
        if crop_img is None:
            continue

        def _current_box(polygon):
            """Return (x1, y1, x2, y2) from polygon."""
            if not polygon:
                return None
            xs = [int(polygon[j]) for j in range(0, len(polygon), 2)]
            ys = [int(polygon[j]) for j in range(1, len(polygon), 2)]
            return min(xs), min(ys), max(xs), max(ys)

        def _box_to_polygon(box):
            """Convert (x1, y1, x2, y2) to 4-corner polygon [x1,y1,x2,y1,x2,y2,x1,y2]."""
            x1, y1, x2, y2 = box
            return [x1, y1, x2, y1, x2, y2, x1, y2]

        # ── Save debug images for this LLM call ──────────────────────────────
        _dbg_dir = output_root / f"page{page_num}" / "images" / "debug_crop_check"
        _dbg_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(_dbg_dir / f"crop{crop.index:02d}_iter00_overlay.png"), page_overlay_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(_dbg_dir / f"crop{crop.index:02d}_iter00_crop.png"), crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        # ── Initial clipping check ────────────────────────────────────────────
        t0 = _time.time()
        try:
            is_clipped, clipped_edges, excess_edges, suggested_box, note = check_di_crop(
                client, deployment, crop_img, page_overlay_img,
                current_box=_current_box(crop.polygon),
                image_size=(w_img, h_img),
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            print(f"  [WARN] Crop check failed for crop {crop.index}: {exc}")
            is_clipped, clipped_edges, excess_edges, suggested_box, note = False, [], [], None, str(exc)
        check_time = round(_time.time() - t0, 3)
        total_time += check_time

        flag = "⚠️ CLIPPED" if is_clipped else ("✂️ EXCESS" if excess_edges else "✅ OK")
        edges_str = ", ".join(clipped_edges) if clipped_edges else "none"
        excess_str = ", ".join(excess_edges) if excess_edges else "none"
        print(f"  [Page {page_num}] Crop {crop.index}: {flag}  clipped={edges_str}  excess={excess_str}  ({check_time}s)  {note}")

        fix_total_time = 0.0
        was_fixed = False
        fix_iterations = 0

        # ── Fix loop: apply suggested_box → re-check, repeat until clean ─────
        other_crops = [c for c in crops if c is not crop]
        while (is_clipped or excess_edges) and crop.polygon and fix_iterations < max_fix_iterations:
            if not suggested_box:
                print(f"    [iter {fix_iterations + 1}] No suggested_box from LLM — cannot fix")
                break

            t1 = _time.time()
            fix_iterations += 1

            # Enforce: only expand clipped edges — keep non-clipped edges at current values.
            # This prevents the LLM from moving the crop to a completely wrong location.
            curr_box = _current_box(crop.polygon)
            if not curr_box:
                break
            curr_x1, curr_y1, curr_x2, curr_y2 = curr_box

            sb_x1 = max(0,     int(suggested_box.get("x1", curr_x1)))
            sb_y1 = max(0,     int(suggested_box.get("y1", curr_y1)))
            sb_x2 = min(w_img, int(suggested_box.get("x2", curr_x2)))
            sb_y2 = min(h_img, int(suggested_box.get("y2", curr_y2)))

            # Edges with no issue stay exactly at current value
            if "left"   not in clipped_edges and "left"   not in excess_edges: sb_x1 = curr_x1
            if "right"  not in clipped_edges and "right"  not in excess_edges: sb_x2 = curr_x2
            if "top"    not in clipped_edges and "top"    not in excess_edges: sb_y1 = curr_y1
            if "bottom" not in clipped_edges and "bottom" not in excess_edges: sb_y2 = curr_y2

            # Clipped edges may only expand (not shrink)
            if "left"   in clipped_edges: sb_x1 = min(sb_x1, curr_x1)
            if "right"  in clipped_edges: sb_x2 = max(sb_x2, curr_x2)
            if "top"    in clipped_edges: sb_y1 = min(sb_y1, curr_y1)
            if "bottom" in clipped_edges: sb_y2 = max(sb_y2, curr_y2)

            # Excess edges may only shrink (not expand)
            if "left"   in excess_edges: sb_x1 = max(sb_x1, curr_x1)
            if "right"  in excess_edges: sb_x2 = min(sb_x2, curr_x2)
            if "top"    in excess_edges: sb_y1 = max(sb_y1, curr_y1)
            if "bottom" in excess_edges: sb_y2 = min(sb_y2, curr_y2)

            if sb_x2 <= sb_x1 or sb_y2 <= sb_y1:
                print(f"    [iter {fix_iterations}] Invalid box after edge clamping — skipping")
                break
            for oc in other_crops:
                if not oc.polygon:
                    continue
                oxs = [int(oc.polygon[j]) for j in range(0, len(oc.polygon), 2)]
                oys = [int(oc.polygon[j]) for j in range(1, len(oc.polygon), 2)]
                ox1, oy1, ox2, oy2 = min(oxs), min(oys), max(oxs), max(oys)
                # Would the suggested box fully contain the other crop?
                would_contain = (sb_x1 <= ox1 and sb_y1 <= oy1 and sb_x2 >= ox2 and sb_y2 >= oy2)
                if not would_contain:
                    continue
                # Clip only the edges that were expanded, to prevent full containment
                if sb_x2 > curr_x2:   # right edge expanded — stop before other crop's right edge
                    sb_x2 = min(sb_x2, ox2 - 1)
                if sb_x1 < curr_x1:   # left edge expanded
                    sb_x1 = max(sb_x1, ox1 + 1)
                if sb_y2 > curr_y2:   # bottom edge expanded
                    sb_y2 = min(sb_y2, oy2 - 1)
                if sb_y1 < curr_y1:   # top edge expanded
                    sb_y1 = max(sb_y1, oy1 + 1)

            corrected = _box_to_polygon((sb_x1, sb_y1, sb_x2, sb_y2))

            # If box didn't actually change after enforcement, stop — further iterations won't help
            if (sb_x1, sb_y1, sb_x2, sb_y2) == curr_box:
                print(f"    [iter {fix_iterations}] Box unchanged after enforcement — stopping early")
                break

            print(f"    [iter {fix_iterations}] Applying fix: edges={edges_str}  box=({sb_x1},{sb_y1},{sb_x2},{sb_y2})")

            # Apply correction
            crop.polygon = corrected
            crop.bbox = polygon_to_bbox(corrected, (w_img, h_img), 0)
            # DI crops are all Pass 1 — they don't overlap each other by design.
            # Save directly from the raw page image with no whitefill.
            new_crop_img = crop_polygon(base_img, corrected)
            if new_crop_img is not None and crop.image_path:
                cv2.imwrite(str(crop.image_path), new_crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                crop_img = new_crop_img
            if crop.coords_path:
                write_polygon_txt(crop.coords_path, corrected)
            was_fixed = True
            _regenerate_di_overlay(output_root, page_num, crops)
            _reloaded = cv2.imread(str(_resolve_llm_input_image_path(output_root, page_num)))
            if _reloaded is not None:
                page_overlay_img = _reloaded

            elapsed = round(_time.time() - t1, 3)
            fix_total_time += elapsed

            # Save debug images for re-check
            cv2.imwrite(str(_dbg_dir / f"crop{crop.index:02d}_iter{fix_iterations:02d}_overlay.png"), page_overlay_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(str(_dbg_dir / f"crop{crop.index:02d}_iter{fix_iterations:02d}_crop.png"), crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            # Re-check clipping with updated crop + overlay
            t2 = _time.time()
            try:
                is_clipped, clipped_edges, excess_edges, suggested_box, note = check_di_crop(
                    client, deployment, crop_img, page_overlay_img,
                    current_box=_current_box(crop.polygon),
                    image_size=(w_img, h_img),
                    reasoning_effort=reasoning_effort,
                )
            except Exception as exc:
                print(f"    [WARN] Crop re-check failed: {exc}")
                break
            check_elapsed = round(_time.time() - t2, 3)
            fix_total_time += check_elapsed
            total_time += check_elapsed
            edges_str = ", ".join(clipped_edges) if clipped_edges else "none"
            flag = "⚠️ still clipped" if is_clipped else "✅ fixed"
            print(f"    [iter {fix_iterations}] Re-check: {flag}  edges={edges_str}  ({check_elapsed}s)  {note}")

        results.append({
            "index": crop.index,
            "image_path": str(crop.image_path),
            "is_clipped": is_clipped,
            "clipped_edges": clipped_edges,
            "was_fixed": was_fixed,
            "fix_iterations": fix_iterations,
            "note": note,
            "check_time_s": check_time,
            "fix_total_time_s": round(fix_total_time, 3),
        })

    return results, round(total_time, 3)


# ── Main pipeline entry point ──────────────────────────────────────────────────

def run_page_pipeline(
    page_num: int,
    *,
    client: AzureOpenAI,
    deployment: str,
    output_root: Path,
    crops_root: Path,
    crops_temp_root: Path,
    detection_reasoning_effort: Optional[str] = None,
    verification_reasoning_effort: Optional[str] = None,
    max_iterations: int = 3,
    grid_step_x: int = GRID_STEP_X,
    grid_step_y: int = GRID_STEP_Y,
    use_grid: bool = True,
    page_dpi_map: Optional[Dict[int, int]] = None,
    run_crop_check: bool = False,
) -> Dict:
    """
    Run the full detection + verification pipeline for one page.

    Parameters:
        page_dpi_map : dict, optional
            Mapping of page_number (1-based) → DPI used when rendering that page.
            Used for accurate overlay generation ensuring DPI-aware coordinate handling.

    Returns:
        {
            "page_num": int,
            "n_existing": int,
            "n_detected": int,
            "n_verified": int,
            "iterations_per_crop": List[int],
            "total_iterations": int,
            "overlay_path": Optional[Path],
            "verification_reasoning_effort": Optional[str],
            "crop_check_results": List[Dict],     # per-crop crop check results
            "crop_check_time_s": float,           # total time for crop check step
        }
    """
    import shutil as _shutil
    import time as _time

    _page_start = _time.time()

    # Log DPI info for this page (if available)
    page_dpi = (page_dpi_map or {}).get(page_num, 300)  # default to 300 DPI if not provided
    print(f"[Page {page_num}] Using DPI: {page_dpi}")

    # Clean temp directory for this page
    temp_dir = _page_output_dir(crops_temp_root, page_num)
    if temp_dir.exists():
        _shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Load existing crops
    crops = load_existing_crops(output_root, crops_root, page_num)
    n_existing = len(crops)
    print(f"\n[Page {page_num}] Existing crops: {n_existing}")

    # Step 0: Crop check — verify boundary placement (clipping + excess area)
    crop_check_results: List[Dict] = []
    crop_check_time_s: float = 0.0
    if run_crop_check and crops:
        print(f"[Page {page_num}] Running crop check...")
        crop_check_results, crop_check_time_s = verify_di_crops(
            client, deployment, output_root, page_num, crops,
            reasoning_effort=verification_reasoning_effort,
            grid_step_x=grid_step_x, grid_step_y=grid_step_y,
            use_grid=use_grid,
            max_fix_iterations=max_iterations,
        )
        n_clipped = sum(1 for r in crop_check_results if r["is_clipped"])
        n_fixed = sum(1 for r in crop_check_results if r.get("was_fixed"))
        if n_clipped:
            print(f"[Page {page_num}] ⚠️ {n_clipped}/{len(crop_check_results)} crops adjusted. ({crop_check_time_s}s)")

    # Get image dimensions
    base_img = cv2.imread(str(_page_image_path(output_root, page_num)))
    if base_img is None:
        raise FileNotFoundError(f"Page image not found for page {page_num}")
    h_img, w_img = base_img.shape[:2]
    image_size = (w_img, h_img)

    # Compute content bbox (deterministic: removes outer frame + title block)
    content_bbox = detect_content_bbox(base_img)
    cx1, cy1, cx2, cy2 = content_bbox
    is_full_image = (cx1 == 0 and cy1 == 0 and cx2 == w_img and cy2 == h_img)
    if not is_full_image:
        print(f"[Page {page_num}] Content bbox: ({cx1},{cy1})-({cx2},{cy2}) "
              f"= {cx2-cx1}x{cy2-cy1}px (frame+title removed)")
    content_poly = [float(cx1), float(cy1), float(cx2), float(cy1),
                    float(cx2), float(cy2), float(cx1), float(cy2)]

    # Compute existing crop coverage vs content area (to warn LLM if coverage is low)
    content_area = (cx2 - cx1) * (cy2 - cy1) if not is_full_image else w_img * h_img
    existing_covered = sum(
        _bbox_area(b)
        for crop in crops
        if crop.polygon and (b := polygon_to_bbox(crop.polygon, image_size, 0)) is not None
    )
    existing_coverage_pct: Optional[float] = None
    if crops and content_area > 0:
        existing_coverage_pct = min(100.0, existing_covered / content_area * 100.0)
        print(f"[Page {page_num}] Existing crop coverage: {existing_coverage_pct:.0f}% of content area")

    # LLM Detection
    print(f"[Page {page_num}] Running detection (effort={detection_reasoning_effort})...")
    result, detection_elapsed = validate_page_with_gpt(
        client, deployment, output_root, page_num, crops, image_size,
        detection_reasoning_effort=detection_reasoning_effort,
        grid_step_x=grid_step_x, grid_step_y=grid_step_y,
        content_bbox=content_bbox if not is_full_image else None,
        existing_coverage_pct=existing_coverage_pct,
        use_grid=use_grid,
    )
    status = result.get("status")
    missing_polygons = result.get("missing", [])
    fp_raw = result.get("full_page", False)
    full_page = bool(fp_raw)

    # Normalize missing list (each item may be dict with "polygon" or a raw list)
    raw_polys = []
    for item in missing_polygons:
        if isinstance(item, dict):
            raw_polys.append(item.get("polygon", []))
        else:
            raw_polys.append(item)

    if status == "complete":
        raw_polys = []
    elif status == "no-images":
        raw_polys = []
    elif full_page and not crops:
        raw_polys = [full_page_polygon(image_size)]

    # Clip all polygons to content bbox — removes outer frame from any detection
    if raw_polys and not is_full_image:
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

    # Coverage checks: snap to content bbox when appropriate
    if raw_polys and not crops:
        content_area = (cx2 - cx1) * (cy2 - cy1)
        total_new_area = sum(
            _bbox_area(b)
            for p in raw_polys
            if (b := polygon_to_bbox(p, image_size, 0)) is not None
        )
        coverage = total_new_area / content_area if content_area > 0 else 1.0

        if coverage > 0.6:
            # Large detection (>60% of content bbox) → snap to content bbox
            raw_polys = [content_poly]
        elif not is_full_image and len(raw_polys) == 1:
            # Single partial detection — expand if it sits inside the content area
            det_bbox = polygon_to_bbox(raw_polys[0], image_size, 0)
            if det_bbox:
                det_area = _bbox_area(det_bbox)
                overlap_ratio = _intersection_area(det_bbox, (cx1, cy1, cx2, cy2)) / max(det_area, 1)
                if overlap_ratio > 0.7 and coverage < 0.55:
                    print(f"[Page {page_num}] Partial detection: {coverage*100:.0f}% of content, "
                          f"{overlap_ratio*100:.0f}% inside content bbox → expanding to full content bbox")
                    raw_polys = [content_poly]

    # Filter overlapping/duplicate polygons
    missing_polygons_filtered = _filter_new_polygons(raw_polys, crops, image_size)
    n_detected = len(missing_polygons_filtered)
    print(f"[Page {page_num}] Detected {n_detected} new region(s) (status={status})")

    # Sync existing crops to output
    sync_existing_crops_to_output(output_root, crops_root, page_num, crops)

    # Create new crops in temp
    start_idx = max((crop.index for crop in crops), default=0)
    new_crops = append_new_crops(
        output_root, crops_temp_root, page_num, missing_polygons_filtered, start_idx
    )

    # Verify and refine
    verified_crops, stats = verify_and_refine_new_crops(
        client, deployment, output_root, crops_root, crops_temp_root,
        page_num, new_crops, crops,
        max_iterations=max_iterations,
        verification_reasoning_effort=verification_reasoning_effort,
        grid_step_x=grid_step_x, grid_step_y=grid_step_y,
        use_grid=use_grid,
    )
    n_verified = len([c for c in verified_crops if not c.is_existing])

    # Move verified to final
    move_verified_crops_to_final(crops_root, crops_temp_root, page_num, verified_crops)

    # Write overlays
    overlay_path = write_overlays(output_root, crops_root, page_num, crops + verified_crops, page_dpi=page_dpi)

    # ── Regenerate final DI overlay showing only confirmed final crops ─────
    # Replaces the intermediate overlay (which showed all iterations) with a
    # clean image of only the final selected regions (DI + new verified crops).
    _regenerate_di_overlay(output_root, page_num, crops + verified_crops)

    # ── Post-processing: crop outer frame & title block from saved images ──
    _apply_content_crop_to_page(crops_root, page_num)

    total_time_s = round(_time.time() - _page_start, 3)

    # Build ordered LLM call log: crop_check first, then detection, then verification
    di_check_calls: List[Dict] = []
    for r in crop_check_results:
        if r["check_time_s"] > 0:
            di_check_calls.append({
                "task": "crop_check",
                "crop_index": r["index"],
                "reasoning_effort": verification_reasoning_effort,
                "time_s": r["check_time_s"],
            })
        if r.get("fix_total_time_s", 0) > 0:
            di_check_calls.append({
                "task": "crop_fix",
                "crop_index": r["index"],
                "fix_iterations": r.get("fix_iterations", 0),
                "reasoning_effort": verification_reasoning_effort,
                "time_s": r["fix_total_time_s"],
            })

    llm_calls: List[Dict] = di_check_calls + [
        {
            "task": "detection",
            "reasoning_effort": detection_reasoning_effort,
            "time_s": round(detection_elapsed, 3),
        }
    ] + stats["llm_calls"]

    return {
        "page_num": page_num,
        "n_existing": n_existing,
        "n_detected": n_detected,
        "n_verified": n_verified,
        "iterations_per_crop": stats["iterations_per_crop"],
        "total_iterations": stats["total_iterations"],
        "overlay_path": overlay_path,
        "verification_reasoning_effort": verification_reasoning_effort,
        "total_time_s": total_time_s,
        "llm_calls": llm_calls,
        "crop_check_results": crop_check_results,
        "crop_check_time_s": crop_check_time_s,
        "content_bbox": list(content_bbox),  # [x1, y1, x2, y2] px
    }
