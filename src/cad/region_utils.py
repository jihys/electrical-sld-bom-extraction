"""Region extraction, cropping, and persistence utilities."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _save_svg_overlay(
    bg_image_path: Path,
    image_size: Tuple[int, int],
    regions: List[Tuple[List[float], str, int]],
    output_path: Path,
) -> None:
    """Save an SVG overlay that references the page PNG as background and draws
    polygon outlines as crisp vector elements.

    Parameters
    ----------
    bg_image_path : Path
        Background page image (PNG path). Referenced via relative path so the
        SVG file is small.
    image_size : (w, h)
        Pixel dimensions of the background image.
    regions : list of (px_coords, hex_color, stroke_width)
        px_coords is a flat list [x0,y0,x1,y1,...] in pixel coordinates.
    output_path : Path
        Destination .svg file.
    """
    w, h = image_size
    try:
        rel_bg = os.path.relpath(bg_image_path, output_path.parent)
    except ValueError:
        rel_bg = str(bg_image_path)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        f'  <image href="{rel_bg}" x="0" y="0" width="{w}" height="{h}"/>',
    ]
    for coords, color, sw in regions:
        pts = " ".join(f"{coords[i]:.1f},{coords[i+1]:.1f}" for i in range(0, len(coords), 2))
        lines.append(f'  <polygon points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}"/>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _load_page_image(png_path: Path, dpi: int = 300) -> Optional[np.ndarray]:
    """Load page image from the pre-rendered PNG.

    Always reads the existing PNG directly so that all coordinate operations
    work in the same pixel space as the stored polygon/bbox files.
    The ``dpi`` parameter is accepted for API compatibility but is not used.
    """
    return cv2.imread(str(png_path))


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RegionRecord:
    page: int
    kind: str
    index: int
    polygon: Sequence[float]
    dest_path: Path
    text: Optional[str] = None


@dataclass
class PageExtractionSummary:
    page: int
    page_image: Path
    tables: List[Path] = field(default_factory=list)
    images: List[Path] = field(default_factory=list)
    texts: List[Path] = field(default_factory=list)


# ── Directory helpers ─────────────────────────────────────────────────────────

def ensure_page_directories(page_number: int, output_root: Path) -> Dict[str, Path]:
    """Create and return the standard subdirectory layout for a page."""
    base_dir = output_root / f"page{page_number}"
    dirs = {
        "base":   base_dir,
        "table":  base_dir / "table",
        "images": base_dir / "images",
        "text":   base_dir / "text",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


# ── Color and polygon helpers ─────────────────────────────────────────────────

def hex_to_bgr(color: str) -> Tuple[int, int, int]:
    """Convert a hex color string or named color to an OpenCV BGR tuple."""
    named = {
        "gray":     (128, 128, 128),
        "darkgray": (64,  64,  64),
        "white":    (255, 255, 255),
        "black":    (0,   0,   0),
    }
    if color.lower() in named:
        return named[color.lower()]
    h = color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _scale_polygon(polygon: Sequence[float], image_size: Tuple[int, int]) -> List[float]:
    width, height = image_size
    coords = list(polygon)
    if not coords:
        return []
    if max(coords) <= 1.5:  # normalized coordinates
        scaled = []
        for idx, value in enumerate(coords):
            scaled.append(value * (width if idx % 2 == 0 else height))
        return scaled
    return coords


def polygon_to_bbox(
    polygon: Sequence[float],
    image_size: Tuple[int, int],
    padding: int = 6,
) -> Optional[Tuple[int, int, int, int]]:
    coords = _scale_polygon(polygon, image_size)
    if not coords:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    left   = max(int(min(xs)) - padding, 0)
    top    = max(int(min(ys)) - padding, 0)
    right  = min(int(max(xs)) + padding, image_size[0])
    bottom = min(int(max(ys)) + padding, image_size[1])
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def _clamp_polygon_to_bbox(
    polygon: Sequence[float],
    image_size: Tuple[int, int],
    content_bbox: Tuple[int, int, int, int],
) -> List[float]:
    """Scale polygon to pixels and clamp every vertex to stay within content_bbox."""
    coords = _scale_polygon(polygon, image_size)
    if not coords:
        return []
    cx1, cy1, cx2, cy2 = content_bbox
    clamped = []
    for i, v in enumerate(coords):
        if i % 2 == 0:
            clamped.append(float(max(cx1, min(cx2, v))))
        else:
            clamped.append(float(max(cy1, min(cy2, v))))
    return clamped


def crop_polygon(
    image: np.ndarray,
    polygon: Sequence[float],
    padding: int = 6,
) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    bbox = polygon_to_bbox(polygon, (w, h), padding)
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2].copy()


def polygon_to_lines(polygon: Sequence[float], image_size: Tuple[int, int]) -> List[str]:
    coords = _scale_polygon(polygon, image_size)
    if not coords:
        return []
    return [f"{coords[i]:.2f},{coords[i+1]:.2f}" for i in range(0, len(coords), 2)]


def write_polygon_txt(dest_path: Path, polygon: Sequence[float], image_size: Tuple[int, int]) -> None:
    lines = polygon_to_lines(polygon, image_size)
    if lines:
        dest_path.write_text("\n".join(lines))


def draw_polygon(
    image: np.ndarray,
    polygon: Sequence[float],
    image_size: Tuple[int, int],
    color: Tuple[int, int, int],
    width: int = 2,
) -> None:
    """Draw a polygon on the image in-place."""
    coords = _scale_polygon(polygon, image_size)
    if not coords:
        return
    pts = np.array(
        [[int(coords[i]), int(coords[i + 1])] for i in range(0, len(coords), 2)],
        dtype=np.int32,
    )
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=width)


def pick_polygon(bounding_regions) -> Optional[Sequence[float]]:
    if not bounding_regions:
        return None
    region = bounding_regions[0]
    return region.polygon if hasattr(region, "polygon") else None


def _center_inside_bbox(
    polygon: Sequence[float],
    image_size: Tuple[int, int],
    content_bbox: Tuple[int, int, int, int],
) -> bool:
    """Return True if the polygon's center lies inside content_bbox.

    Used to exclude text/table/figure regions outside the outer frame
    (e.g. title block text, margin annotations, frame-edge labels).
    """
    bb = polygon_to_bbox(polygon, image_size, padding=0)
    if not bb:
        return True  # can't determine → don't filter
    cx = (bb[0] + bb[2]) / 2
    cy = (bb[1] + bb[3]) / 2
    x1, y1, x2, y2 = content_bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _is_wrap_around_polygon(
    polygon: Sequence[float],
    image_size: Tuple[int, int],
    max_span_x: float = 0.6,
    max_span_y: float = 0.3,
) -> bool:
    """Return True if the polygon wraps around or is self-intersecting.

    Detects three cases:
    1. Polygon vertices lie significantly outside the image boundary.
    2a. Bounding span extremely wide in x alone (>0.8 page width) — this
        catches SLD row-label pairs where DI groups labels on both sides of
        the diagram into one paragraph, producing a flat parallelogram that
        spans the full width and draws diagonal lines across the page.
    2b. The bounding span is excessively large in both x and y simultaneously
        (indicating a wrap-around annotation).
    3. For 4-point polygons: self-intersecting quadrilateral (vertices in wrong
       order, creating diagonal lines when drawn with cv2.polylines).
    """
    coords = _scale_polygon(polygon, image_size)
    if len(coords) < 6:
        return False

    w, h = image_size
    pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

    # 1. Any vertex significantly outside the image boundary
    for px, py in pts:
        if px < -w * 0.1 or px > w * 1.1 or py < -h * 0.1 or py > h * 1.1:
            return True

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span_x = (max(xs) - min(xs)) / w
    span_y = (max(ys) - min(ys)) / h

    # 2a. Extremely wide span in x alone — row-label wrap-around
    if span_x > 0.8:
        return True

    # 2b. Bounding box spans excessively in both dimensions
    if span_x > max_span_x and span_y > max_span_y:
        return True

    # 3. Self-intersecting quadrilateral (cross products alternate sign)
    if len(pts) == 4:
        signs = []
        n = 4
        for i in range(n):
            p1, p2, p3 = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
            if abs(cross) > 1:
                signs.append(1 if cross > 0 else -1)
        if len(signs) >= 2 and len(set(signs)) > 1:
            return True

    return False


def _overlaps_table_regions(
    polygon: Sequence[float],
    image_size: Tuple[int, int],
    table_bboxes: List[Tuple[int, int, int, int]],
) -> bool:
    """Return True if the polygon's center falls within any table bounding box."""
    bb = polygon_to_bbox(polygon, image_size, padding=0)
    if not bb:
        return False
    cx = (bb[0] + bb[2]) / 2
    cy = (bb[1] + bb[3]) / 2
    for tx1, ty1, tx2, ty2 in table_bboxes:
        if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
            return True
    return False


# ── Persistence functions ─────────────────────────────────────────────────────

def persist_table_regions(
    page_image_paths: List[Path],
    analysis_results: Dict,
    page_summaries: Dict,
    output_root: Path,
    content_bboxes: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
) -> List[RegionRecord]:
    """Crop table regions and save to outputs/pages/<page>/table/.

    Parameters
    ----------
    content_bboxes : optional dict mapping page_num → (x1,y1,x2,y2).
        Regions whose center falls outside this bbox are skipped
        (e.g. title-block tables, frame-edge elements).
    """
    records: List[RegionRecord] = []
    for page_num, image_path in enumerate(page_image_paths, start=1):
        result = analysis_results.get(page_num)
        if result is None:
            continue
        tables = getattr(result, "tables", []) or []
        if not tables:
            continue
        dirs = ensure_page_directories(page_num, output_root)
        page_summary = page_summaries[page_num]
        base_image = _load_page_image(image_path)
        h, w = base_image.shape[:2]
        overlay = base_image.copy()
        cb = (content_bboxes or {}).get(page_num)
        for table_idx, table in enumerate(tables, start=1):
            polygon = pick_polygon(getattr(table, "bounding_regions", None))
            if not polygon:
                continue
            if cb and not _center_inside_bbox(polygon, (w, h), cb):
                continue  # outside outer frame — skip
            cropped = crop_polygon(base_image, polygon)
            if cropped is None:
                continue
            dest_path = dirs["table"] / f"tables_{page_num}_{table_idx:02d}.png"
            cv2.imwrite(str(dest_path), cropped)
            write_polygon_txt(dest_path.with_suffix(".txt"), polygon, (w, h))
            draw_polygon(overlay, polygon, (w, h), color=hex_to_bgr("#1f6feb"), width=2)
            records.append(RegionRecord(page=page_num, kind="table", index=table_idx, polygon=polygon, dest_path=dest_path))
            page_summary.tables.append(dest_path)
        cv2.imwrite(str(dirs["table"] / f"table_overlay_page{page_num:02d}.png"), overlay)
    return records


def _whitefill_polygons(
    image: np.ndarray,
    polygons: List[Sequence[float]],
    shrink_px: int = 0,
    reference_polygon: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return a copy of image with each polygon's bounding box white-filled.

    Parameters
    ----------
    shrink_px : shrink the fill region inward on the facing edge by this many
        pixels so that border text is preserved. Use 0 for exact fill (DI input).
    reference_polygon : the crop currently being saved. When provided the shrink
        is applied ONLY on the single edge of each whitefill polygon that faces
        this reference, and ONLY when the two bounding boxes actually overlap.
        If None the shrink is applied uniformly on all four sides.
    """
    h, w = image.shape[:2]
    filled = image.copy()

    # Pre-compute reference bounding box (pixel coords)
    ref_bounds: Optional[tuple] = None
    if reference_polygon is not None and shrink_px > 0:
        rc = list(reference_polygon)
        if rc and max(rc) <= 1.5:
            rc = [v * (w if i % 2 == 0 else h) for i, v in enumerate(rc)]
        ref_bounds = (min(rc[0::2]), min(rc[1::2]), max(rc[0::2]), max(rc[1::2]))

    for polygon in polygons:
        coords = list(polygon)
        if coords and max(coords) <= 1.5:  # normalized → scale to pixels
            coords = [v * (w if i % 2 == 0 else h) for i, v in enumerate(coords)]
        xs, ys = coords[0::2], coords[1::2]
        bx1, by1 = int(min(xs)), int(min(ys))
        bx2, by2 = int(max(xs)), int(max(ys))

        # Compute per-side shrink: sl=left, st=top, sr=right, sb=bottom
        sl = st = sr = sb = 0
        if shrink_px > 0:
            if ref_bounds is not None:
                rx1, ry1, rx2, ry2 = ref_bounds
                # Only apply margin when the two boxes actually overlap
                overlap_x = max(0, min(bx2, rx2) - max(bx1, rx1))
                overlap_y = max(0, min(by2, ry2) - max(by1, ry1))
                if overlap_x > 0 or overlap_y > 0:
                    # Determine primary facing direction by center offset
                    dcx = (rx1 + rx2) / 2 - (bx1 + bx2) / 2
                    dcy = (ry1 + ry2) / 2 - (by1 + by2) / 2
                    if abs(dcy) >= abs(dcx):
                        # Vertical relationship: shrink only the edge facing reference
                        if dcy > 0:   # reference is below → shrink this polygon's bottom
                            sb = shrink_px
                        else:         # reference is above → shrink this polygon's top
                            st = shrink_px
                    else:
                        # Horizontal relationship
                        if dcx > 0:   # reference is to the right → shrink this polygon's right
                            sr = shrink_px
                        else:         # reference is to the left → shrink this polygon's left
                            sl = shrink_px
            else:
                # No reference: uniform shrink on all four sides
                sl = st = sr = sb = shrink_px

        x1 = max(0, bx1 + sl)
        y1 = max(0, by1 + st)
        x2 = min(w, bx2 - sr)
        y2 = min(h, by2 - sb)
        if x2 > x1 and y2 > y1:
            filled[y1:y2, x1:x2] = 255
    return filled


def whitefill_figures(page_img: np.ndarray, figures) -> Tuple[np.ndarray, int]:
    """Return a copy of page_img with all figure bounding boxes white-filled.

    Used to prepare Pass 2 input for Document Intelligence: white-fills the
    regions already found in Pass 1 so DI can discover additional figures.

    Returns (filled_image, n_filled).
    """
    h, w = page_img.shape[:2]
    filled = page_img.copy()
    n_filled = 0
    for figure in (figures or []):
        polygon = pick_polygon(getattr(figure, "bounding_regions", None))
        if not polygon:
            continue
        coords = list(polygon)
        if max(coords) <= 1.5:
            coords = [v * (w if i % 2 == 0 else h) for i, v in enumerate(coords)]
        xs, ys = coords[0::2], coords[1::2]
        x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
        x2, y2 = min(w, int(max(xs))), min(h, int(max(ys)))
        if x2 > x1 and y2 > y1:
            filled[y1:y2, x1:x2] = 255
            n_filled += 1
    return filled, n_filled


def persist_figure_regions(
    page_image_paths: List[Path],
    analysis_results: Dict,
    page_summaries: Dict,
    output_root: Path,
    content_bboxes: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
    pass1_count_per_page: Optional[Dict[int, int]] = None,
    page_dpi_map: Optional[Dict[int, int]] = None,
) -> List[RegionRecord]:
    """Crop figure regions and save to outputs/pages/<page>/images/.

    Pass 1 figures are saved with no whitefill (DI never returns overlapping regions
    within the same pass). Pass 2 figures are saved with Pass 1 regions white-filled
    (15px inward on the facing edge) so Pass 2 crops don't contain duplicate content.

    Parameters
    ----------
    pass1_count_per_page : dict mapping page_num → number of figures found in Pass 1.
        Figures with fig_idx <= pass1_count are Pass 1 (saved as-is, no whitefill).
        Figures with fig_idx > pass1_count are Pass 2 (Pass 1 regions whitefilled).
    page_dpi_map : optional dict mapping page_num → DPI used when rendering that page.
        Used to render the source PDF at the correct per-page DPI for consistent crop
        resolution. Falls back to 300 DPI when not provided.
    """
    records: List[RegionRecord] = []
    for page_num, image_path in enumerate(page_image_paths, start=1):
        result = analysis_results.get(page_num)
        if result is None:
            continue
        figures = getattr(result, "figures", []) or []
        if not figures:
            continue
        dirs = ensure_page_directories(page_num, output_root)
        page_summary = page_summaries[page_num]
        page_dpi = (page_dpi_map or {}).get(page_num, 300)
        base_image = _load_page_image(image_path, dpi=page_dpi)
        h, w = base_image.shape[:2]
        overlay = base_image.copy()
        cb = (content_bboxes or {}).get(page_num)

        # How many figures came from Pass 1 for this page
        pass1_count = (pass1_count_per_page or {}).get(page_num, len(figures))

        # Collect all polygons upfront (Pass 1 subset used for whitefilling Pass 2 saves)
        fig_polygons = [
            pick_polygon(getattr(fig, "bounding_regions", None)) for fig in figures
        ]
        pass1_polygons = [
            fig_polygons[j] for j in range(min(pass1_count, len(fig_polygons)))
            if fig_polygons[j] is not None
        ]

        for fig_idx, (figure, polygon) in enumerate(zip(figures, fig_polygons), start=1):
            if not polygon:
                continue
            if cb and not _center_inside_bbox(polygon, (w, h), cb):
                continue  # outside outer frame — skip

            # Clamp polygon vertices to content bbox (removes outer-frame bleed)
            effective_polygon = _clamp_polygon_to_bbox(polygon, (w, h), cb) if cb else polygon

            if fig_idx <= pass1_count:
                # Pass 1: DI never overlaps within the same pass → save as-is
                save_base = base_image
            else:
                # Pass 2: whitefill Pass 1 regions from this crop (30px inward, facing edge only)
                save_base = _whitefill_polygons(base_image, pass1_polygons, shrink_px=30, reference_polygon=effective_polygon) if pass1_polygons else base_image

            cropped = crop_polygon(save_base, effective_polygon)
            if cropped is None:
                continue
            dest_path = dirs["images"] / f"images_{page_num}_{fig_idx:02d}.png"
            cv2.imwrite(str(dest_path), cropped)
            write_polygon_txt(dest_path.with_suffix(".txt"), effective_polygon, (w, h))
            draw_polygon(overlay, effective_polygon, (w, h), color=hex_to_bgr("#f97316"), width=10)
            records.append(RegionRecord(page=page_num, kind="figure", index=fig_idx, polygon=effective_polygon, dest_path=dest_path))
            page_summary.images.append(dest_path)
        cv2.imwrite(str(dirs["images"] / f"images_overlay_page{page_num:02d}.png"), overlay)
    return records


def persist_text_regions(
    page_image_paths: List[Path],
    analysis_results: Dict,
    page_summaries: Dict,
    output_root: Path,
    content_bboxes: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
) -> List[RegionRecord]:
    """Crop paragraph regions and save to two locations per page.

    text/raw/
        Raw output — every DI paragraph with a valid polygon is saved here
        (wrap-around polygons excluded since their bbox is unreliable).
        Files: images_<page>_<idx>.png / .txt,  text_<page>_<idx>.txt
        Overlay: text_overlay_page<NN>.png  (colour-coded by filter status)
          green  (#22c55e) – clean (passes all filters)
          orange (#f97316) – outside content_bbox
          blue   (#1f6feb) – inside a table region
          red    (#ef4444) – wrap-around polygon (drawn only, not saved)

    text/clean/
        Filtered final output — only paragraphs that pass all three filters:
          1. inside content_bbox  (frame/title-block text removed)
          2. not wrap-around      (diagonal / self-intersecting polygons removed)
          3. not inside a table   (table text handled separately)
        Files: images_<page>_<idx>.png / .txt,  text_<page>_<idx>.txt
        Overlay: text_overlay_page<NN>_clean.png  (all green)

    Returns RegionRecord list for the clean set only.
    """
    records: List[RegionRecord] = []
    for page_num, image_path in enumerate(page_image_paths, start=1):
        result = analysis_results.get(page_num)
        if result is None:
            continue
        paragraphs = getattr(result, "paragraphs", []) or []
        if not paragraphs:
            continue
        dirs = ensure_page_directories(page_num, output_root)
        page_summary = page_summaries[page_num]
        base_image = _load_page_image(image_path)
        h, w = base_image.shape[:2]
        raw_overlay = base_image.copy()    # colour-coded: all non-wrap-around regions
        clean_overlay = base_image.copy()  # green only: clean regions
        cb = (content_bboxes or {}).get(page_num)

        # text/raw/ and text/clean/ subdirectories
        raw_dir = dirs["text"] / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        clean_dir = dirs["text"] / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)

        # Build table bboxes for this page
        tables = getattr(result, "tables", []) or []
        table_bboxes_page: List[Tuple[int, int, int, int]] = []
        for table in tables:
            tpoly = pick_polygon(getattr(table, "bounding_regions", None))
            if tpoly:
                if cb and not _center_inside_bbox(tpoly, (w, h), cb):
                    continue
                tbb = polygon_to_bbox(tpoly, (w, h), padding=0)
                if tbb:
                    table_bboxes_page.append(tbb)

        raw_idx = 0
        clean_idx = 0
        raw_svg_regions: List[Tuple[List[float], str, int]] = []
        clean_svg_regions: List[Tuple[List[float], str, int]] = []

        for paragraph in paragraphs:
            regions = getattr(paragraph, "bounding_regions", None) or []
            for region in regions:
                polygon = region.polygon if hasattr(region, "polygon") else None
                if not polygon:
                    continue

                coords = _scale_polygon(polygon, (w, h))

                # Wrap-around: add to raw SVG only, skip save (bbox unreliable)
                if _is_wrap_around_polygon(polygon, (w, h)):
                    raw_svg_regions.append((coords, "#ef4444", 2))
                    continue

                # Determine filter status
                if cb and not _center_inside_bbox(polygon, (w, h), cb):
                    hex_color = "#f97316"   # orange — outside frame
                    keep = False
                elif table_bboxes_page and _overlaps_table_regions(polygon, (w, h), table_bboxes_page):
                    hex_color = "#1f6feb"   # blue — inside table
                    keep = False
                else:
                    hex_color = "#22c55e"   # green — clean
                    keep = True

                # ── Raw save: coord + content .txt → text/raw/ ───────────────
                raw_idx += 1
                coord_path = raw_dir / f"text_{page_num}_{raw_idx:03d}.txt"
                write_polygon_txt(coord_path, polygon, (w, h))
                if getattr(paragraph, "content", None):
                    (raw_dir / f"text_content_{page_num}_{raw_idx:03d}.txt").write_text(paragraph.content.strip())
                raw_svg_regions.append((coords, hex_color, 2))

                # ── Clean save: coord + content .txt → text/clean/ ───────────
                if keep:
                    clean_idx += 1
                    clean_coord_path = clean_dir / f"text_{page_num}_{clean_idx:03d}.txt"
                    write_polygon_txt(clean_coord_path, polygon, (w, h))
                    if getattr(paragraph, "content", None):
                        (clean_dir / f"text_content_{page_num}_{clean_idx:03d}.txt").write_text(paragraph.content.strip())
                    clean_svg_regions.append((coords, "#22c55e", 2))
                    records.append(RegionRecord(
                        page=page_num, kind="text", index=clean_idx,
                        polygon=polygon, dest_path=clean_coord_path,
                        text=getattr(paragraph, "content", ""),
                    ))
                    page_summary.texts.append(clean_coord_path)

        # SVG overlays — vector polygon outlines over referenced page PNG
        _save_svg_overlay(image_path, (w, h), raw_svg_regions,
                          raw_dir / f"text_overlay_page{page_num:02d}.svg")
        _save_svg_overlay(image_path, (w, h), clean_svg_regions,
                          clean_dir / f"text_overlay_page{page_num:02d}_clean.svg")
    return records


def save_full_text_overlay(
    page_image_paths: List[Path],
    analysis_results: Dict,
    output_root: Path,
    content_bboxes: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
) -> None:
    """Overlay paragraph bounding boxes and save as text_overlay_pageXX_all.png.

    Applies the same filtering as persist_text_regions:
    content_bbox exclusion, wrap-around removal, and table region exclusion.
    """
    for page_num, image_path in enumerate(page_image_paths, start=1):
        result = analysis_results.get(page_num)
        if result is None:
            continue
        paragraphs = getattr(result, "paragraphs", []) or []
        if not paragraphs:
            continue
        dirs = ensure_page_directories(page_num, output_root)
        base_image = _load_page_image(image_path)
        h, w = base_image.shape[:2]
        overlay = base_image.copy()
        cb = (content_bboxes or {}).get(page_num)

        # Build table bboxes for this page
        tables = getattr(result, "tables", []) or []
        table_bboxes_page: List[Tuple[int, int, int, int]] = []
        for table in tables:
            tpoly = pick_polygon(getattr(table, "bounding_regions", None))
            if tpoly:
                if cb and not _center_inside_bbox(tpoly, (w, h), cb):
                    continue
                tbb = polygon_to_bbox(tpoly, (w, h), padding=0)
                if tbb:
                    table_bboxes_page.append(tbb)

        for paragraph in paragraphs:
            for region in (getattr(paragraph, "bounding_regions", None) or []):
                polygon = region.polygon if hasattr(region, "polygon") else None
                if not polygon:
                    continue
                if cb and not _center_inside_bbox(polygon, (w, h), cb):
                    continue  # outside outer frame — skip
                if _is_wrap_around_polygon(polygon, (w, h)):
                    continue  # wrap-around / self-intersecting — skip
                if table_bboxes_page and _overlaps_table_regions(polygon, (w, h), table_bboxes_page):
                    continue  # inside table — skip
                draw_polygon(overlay, polygon, (w, h), color=hex_to_bgr("#ef4444"), width=2)
        cv2.imwrite(str(dirs["text"] / f"text_overlay_page{page_num:02d}_all.png"), overlay)


def save_raw_text_debug(
    page_image_paths: List[Path],
    analysis_results: Dict,
    output_root: Path,
    content_bboxes: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
) -> None:
    """Save pre-filter debug data for every page.

    For each page two files are written to outputs/pages/<page>/text/:

    text/raw/text_raw_pageXX.json
        Every DI paragraph with full coordinate tracing:
          "text"          – paragraph text content
          "filter"        – reason this paragraph would be filtered:
                            "ok" | "outside_frame" | "wrap_around" | "in_table"
          "polygon_raw"   – exact values returned by DI (before any scaling).
                            Useful to see if DI used normalised (0-1), inch, or
                            pixel coordinates, and the vertex order DI chose.
          "coord_mode"    – "normalized" if max(polygon_raw) <= 1.5,
                            "absolute" otherwise (passed through as-is).
          "polygon_px"    – pixel coordinates after _scale_polygon() —
                            what actually gets drawn / cropped.
          "bbox_px"       – [x1,y1,x2,y2] bounding box in pixels.
          "polygon_is_convex" – True if the 4-pt quadrilateral is convex
                            (non-convex = self-intersecting = diagonal lines).

    text/raw/text_overlay_pageXX_raw.png
        Overlay image with colour-coded bounding boxes:
          green  (#22c55e) – kept
          orange (#f97316) – outside_frame
          red    (#ef4444) – wrap_around
          blue   (#1f6feb) – in_table
    """
    _REASON_COLOR = {
        "ok":             "#22c55e",
        "outside_frame":  "#f97316",
        "wrap_around":    "#ef4444",
        "in_table":       "#1f6feb",
    }

    for page_num, image_path in enumerate(page_image_paths, start=1):
        result = analysis_results.get(page_num)
        if result is None:
            continue
        paragraphs = getattr(result, "paragraphs", []) or []
        if not paragraphs:
            continue
        dirs = ensure_page_directories(page_num, output_root)
        raw_dir = dirs["text"] / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        base_image = _load_page_image(image_path)
        h, w = base_image.shape[:2]
        overlay = base_image.copy()
        cb = (content_bboxes or {}).get(page_num)

        # Build table bboxes for this page
        tables = getattr(result, "tables", []) or []
        table_bboxes_page: List[Tuple[int, int, int, int]] = []
        for table in tables:
            tpoly = pick_polygon(getattr(table, "bounding_regions", None))
            if tpoly:
                if cb and not _center_inside_bbox(tpoly, (w, h), cb):
                    continue
                tbb = polygon_to_bbox(tpoly, (w, h), padding=0)
                if tbb:
                    table_bboxes_page.append(tbb)

        records_json = []
        for paragraph in paragraphs:
            text_content = getattr(paragraph, "content", "") or ""
            for region in (getattr(paragraph, "bounding_regions", None) or []):
                polygon = region.polygon if hasattr(region, "polygon") else None
                if not polygon:
                    continue

                # Determine filter reason
                if cb and not _center_inside_bbox(polygon, (w, h), cb):
                    reason = "outside_frame"
                elif _is_wrap_around_polygon(polygon, (w, h)):
                    reason = "wrap_around"
                elif table_bboxes_page and _overlaps_table_regions(polygon, (w, h), table_bboxes_page):
                    reason = "in_table"
                else:
                    reason = "ok"

                # Raw values exactly as DI returned them
                raw_vals = list(polygon)
                coord_mode = "normalized" if (raw_vals and max(raw_vals) <= 1.5) else "absolute"

                # Scaled pixel coordinates (what gets drawn/cropped)
                coords = _scale_polygon(polygon, (w, h))
                pts_px = [[round(coords[i], 1), round(coords[i + 1], 1)]
                          for i in range(0, len(coords), 2)]
                bbox = polygon_to_bbox(polygon, (w, h), padding=0)

                # Convexity check for 4-pt polygons
                is_convex = None
                if len(pts_px) == 4:
                    signs = []
                    for i in range(4):
                        p1, p2, p3 = pts_px[i], pts_px[(i+1)%4], pts_px[(i+2)%4]
                        cross = (p2[0]-p1[0])*(p3[1]-p2[1]) - (p2[1]-p1[1])*(p3[0]-p2[0])
                        if abs(cross) > 1:
                            signs.append(1 if cross > 0 else -1)
                    is_convex = (len(set(signs)) == 1) if signs else True

                records_json.append({
                    "text": text_content.strip(),
                    "filter": reason,
                    "polygon_raw": raw_vals,
                    "coord_mode": coord_mode,
                    "polygon_px": pts_px,
                    "bbox_px": list(bbox) if bbox else None,
                    "polygon_is_convex": is_convex,
                })

                draw_polygon(overlay, polygon, (w, h),
                             color=hex_to_bgr(_REASON_COLOR[reason]), width=2)

        # Write JSON → text/raw/
        json_path = raw_dir / f"text_raw_page{page_num:02d}.json"
        json_path.write_text(json.dumps(records_json, ensure_ascii=False, indent=2))

        # Write raw overlay → text/raw/
        cv2.imwrite(str(raw_dir / f"text_overlay_page{page_num:02d}_raw.png"), overlay)

        kept = sum(1 for r in records_json if r["filter"] == "ok")
        print(
            f"  page {page_num:2d}: {len(records_json):4d} total paragraphs  "
            f"kept={kept}  "
            f"outside_frame={sum(1 for r in records_json if r['filter'] == 'outside_frame')}  "
            f"wrap_around={sum(1 for r in records_json if r['filter'] == 'wrap_around')}  "
            f"in_table={sum(1 for r in records_json if r['filter'] == 'in_table')}"
        )


def save_di_raw_results(
    page_image_paths: List[Path],
    analysis_results: Dict,
    output_root: Path,
) -> None:
    """Serialize raw DI AnalyzeResult objects to JSON, one file per page.

    Files are written to outputs/pages/<page>/di_result_page<NN>.json.
    Uses the SDK's as_dict() method when available; falls back to a
    best-effort manual serialisation if needed.
    """
    for page_num, image_path in enumerate(page_image_paths, start=1):
        result = analysis_results.get(page_num)
        if result is None:
            continue
        dirs = ensure_page_directories(page_num, output_root)
        out_path = dirs["base"] / f"di_result_page{page_num:02d}.json"

        # azure-ai-documentintelligence models expose as_dict()
        try:
            data = result.as_dict()
        except AttributeError:
            # Fallback: try model_dump (Pydantic-style) or __dict__
            try:
                data = result.model_dump()
            except AttributeError:
                data = vars(result)

        def _default(obj):
            """JSON serialiser for non-primitive SDK objects."""
            try:
                return obj.as_dict()
            except AttributeError:
                pass
            try:
                return obj.model_dump()
            except AttributeError:
                pass
            return str(obj)

        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=_default)
        )
        print(f"  page {page_num:2d}: DI raw result → {out_path.relative_to(output_root)}")
