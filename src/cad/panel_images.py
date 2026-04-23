from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from .grid_utils import draw_ruler


def add_grid_overlay(img_bgr: np.ndarray, grid_size: int = 120) -> np.ndarray:
    # Show ruler ticks only along the image edges — does not obscure CAD content with interior lines
    return draw_ruler(img_bgr, step_x=grid_size, step_y=grid_size)


def draw_target_bbox(img_bgr: np.ndarray, bbox: List[int], color=(255, 0, 0), label: str = "TARGET") -> np.ndarray:
    out = img_bgr.copy()
    h_img, w_img = out.shape[:2]
    x1, y1, x2, y2 = bbox
    # Scale line thickness and font size relative to image dimensions
    ref_dim = max(w_img, h_img)
    line_thickness = max(2, int(ref_dim / 1000))
    cv2.rectangle(out, (x1, y1), (x2, y2), color, line_thickness)
    # Strip prefixes (NAME:, OTHER:, PANEL:) and show the name in a larger font
    display = label.split(":", 1)[-1] if ":" in label else label
    # Adaptive font scale: keep label roughly the same height as the bbox
    bbox_h = max(y2 - y1, 1)
    bbox_w = max(x2 - x1, 1)
    # Target: rendered text height ≈ 80% of bbox height
    # cv2 FONT_HERSHEY_SIMPLEX at scale=1.0 produces ~22px tall glyphs
    # Cap at 2.5 so large panel bboxes don't produce enormous text
    font_scale = min(2.5, max(0.5, (bbox_h * 0.8) / 22.0))
    # Cap so the label width doesn't exceed 2x bbox width
    thickness = max(1, int(font_scale * 1.5))
    (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if tw > bbox_w * 2:
        font_scale *= (bbox_w * 2) / tw
        font_scale = max(0.4, font_scale)
        thickness = max(1, int(font_scale * 1.5))
        (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = max(4, int(font_scale * 4))
    (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # Place label left of bbox end if it would overflow the image right edge
    text_x = x1
    if text_x + tw + pad > w_img:
        text_x = max(0, x2 - tw - pad)
    text_y = max(th + pad, y1 - pad)
    # Background rectangle for readability
    cv2.rectangle(out, (text_x, text_y - th - pad // 2), (text_x + tw + pad, text_y + pad // 2), color, -1)
    cv2.putText(
        out, display, (text_x + pad // 2, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )
    return out


def draw_target_banner(img_bgr: np.ndarray, panel_name: str) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    bw = min(max(200, w - 20), 1200)
    cv2.rectangle(out, (10, 10), (10 + bw, 58), (0, 0, 220), -1)
    cv2.putText(out, f"TARGET: {panel_name}", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_locate_input_image(
    img_bgr: np.ndarray,
    panel_name: str,
    grid_size: int = 120,
    target_bbox: Optional[List[int]] = None,
    other_name_bboxes: Optional[List[tuple]] = None,  # [(other_name, bbox), ...]
) -> np.ndarray:
    out = add_grid_overlay(img_bgr, grid_size)
    # Other panel name regions: green (displayed the same way as in verify)
    if other_name_bboxes:
        for other_name, other_bbox in other_name_bboxes:
            if other_bbox is not None:
                out = draw_target_bbox(out, other_bbox, color=(0, 200, 0), label=f"OTHER:{other_name}")
    # Blue: target panel name location
    if target_bbox is not None:
        out = draw_target_bbox(out, target_bbox, color=(255, 0, 0), label=f"NAME:{panel_name}")
    return out


def make_verify_overlay_image(
    img_bgr: np.ndarray,
    panel_name: str,
    bbox: List[int],
    grid_size: int = 120,
    name_bbox: Optional[List[int]] = None,
    other_name_bboxes: Optional[List[tuple]] = None,  # [(other_name, bbox), ...]
) -> np.ndarray:
    out = add_grid_overlay(img_bgr, grid_size)
    # Other panel name regions: draw in green first
    if other_name_bboxes:
        for other_name, other_bbox in other_name_bboxes:
            if other_bbox is not None:
                out = draw_target_bbox(out, other_bbox, color=(0, 200, 0), label=f"OTHER:{other_name}")
    # Orange: full panel region
    out = draw_target_bbox(out, bbox, color=(0, 165, 255), label="PANEL")
    # Blue: target panel name location
    if name_bbox is not None:
        out = draw_target_bbox(out, name_bbox, color=(255, 0, 0), label=f"NAME:{panel_name}")
    return out


def make_locate_all_input_image(
    img_bgr: np.ndarray,
    all_name_bboxes: dict,  # {panel_name: bbox_or_None}
    grid_size: int = 120,
) -> np.ndarray:
    """Create locate input image with ALL panel name bboxes shown (for batch locate call)."""
    out = add_grid_overlay(img_bgr, grid_size)
    for name, bbox in all_name_bboxes.items():
        if bbox is not None:
            out = draw_target_bbox(out, bbox, color=(255, 0, 0), label=f"NAME:{name}")
    return out


def make_verify_all_overlay_image(
    img_bgr: np.ndarray,
    panel_bboxes: dict,  # {panel_name: bbox} — current bbox per panel being verified
    all_name_bboxes: dict,  # {panel_name: bbox_or_None} — name text locations
    grid_size: int = 120,
) -> np.ndarray:
    """Create a single overlay image for batch verify: shows ALL panels' current PANEL boxes
    (orange, labeled PANEL:name) and all NAME boxes (blue, labeled NAME:name).
    """
    out = add_grid_overlay(img_bgr, grid_size)
    # Draw PANEL boxes for each panel being verified
    for name, bbox in panel_bboxes.items():
        if bbox is not None:
            out = draw_target_bbox(out, bbox, color=(0, 165, 255), label=f"PANEL:{name}")
    # Draw NAME boxes for all panels
    for name, bbox in all_name_bboxes.items():
        if bbox is not None:
            out = draw_target_bbox(out, bbox, color=(255, 0, 0), label=f"NAME:{name}")
    return out


def _find_file_case_insensitive(base: Path, name: str) -> Optional[Path]:
    p = base / name
    if p.exists():
        return p
    low = name.lower()
    for x in base.glob("*"):
        if x.name.lower() == low:
            return x
    return None


def load_guide_images(notebook_dir: Path, guide_image_names: List[str]) -> List[np.ndarray]:
    search_dirs = [notebook_dir, notebook_dir / "test_images", notebook_dir.parent, notebook_dir.parent / "data"]
    out: List[np.ndarray] = []
    for nm in guide_image_names:
        found = None
        for d in search_dirs:
            p = _find_file_case_insensitive(d, nm)
            if p is not None:
                found = p
                break
        if found is None:
            print(f"[WARN] guide image not found: {nm}")
            continue
        img = cv2.imread(str(found))
        if img is None:
            print(f"[WARN] failed to read guide image: {found}")
            continue
        out.append(img)
        print(f"[GUIDE] loaded: {found}")
    return out