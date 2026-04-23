"""PDF → PNG / SVG conversion tools.

Wraps PyMuPDF to split PDFs into per-page images and vector data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from pydantic import Field

from ..models.page import PageInfo


# ──────────────────────────────────────────────────────────────────────────────
# DPI computation
# ──────────────────────────────────────────────────────────────────────────────

def _image_dpi_for_page(page: fitz.Page) -> Optional[float]:
    best: Optional[float] = None
    for img in page.get_image_info():
        w_px, h_px = img.get("width", 0), img.get("height", 0)
        bbox = img.get("bbox")
        if not (w_px and h_px and bbox):
            continue
        w_pt = bbox[2] - bbox[0]
        h_pt = bbox[3] - bbox[1]
        if w_pt <= 0 or h_pt <= 0:
            continue
        dpi = max(w_px / (w_pt / 72.0), h_px / (h_pt / 72.0))
        if best is None or dpi > best:
            best = dpi
    return best


def compute_dpi_for_pdf(
    pdf_path: Annotated[str, Field(description="Path to the PDF file")],
    target_min_px: int = 10,
    min_dpi: int = 72,
    max_dpi: int = 720,
    round_to: int = 72,
    no_text_dpi: int = 288,
) -> Tuple[int, Dict[int, int]]:
    """Compute per-page and global DPI based on minimum font size.

    Returns (global_dpi, {page_num: dpi}).
    """
    def _round_dpi(raw: float) -> int:
        clamped = max(min_dpi, min(max_dpi, raw))
        return int(math.ceil(clamped / round_to) * round_to)

    def _font_dpi(font_pt: float) -> int:
        if font_pt < 2.5:
            return 600
        if font_pt < 2.7:
            return 576
        if font_pt < 2.9:
            return 504
        if font_pt < 3.1:
            return 432
        if font_pt < 3.3:
            return 360
        if font_pt < 3.5:
            return 288
        return _round_dpi(target_min_px * 72.0 / font_pt)

    doc = fitz.open(str(pdf_path))
    per_page: Dict[int, int] = {}
    global_min = float("inf")

    for page_idx, page in enumerate(doc, start=1):
        page_min = float("inf")
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0.0)
                    if size >= 2.0:
                        page_min = min(page_min, size)

        if page_min != float("inf"):
            per_page[page_idx] = _font_dpi(page_min)
            global_min = min(global_min, page_min)
        else:
            img_dpi = _image_dpi_for_page(page)
            if img_dpi:
                per_page[page_idx] = _round_dpi(img_dpi)
            elif len(page.get_drawings()) > 30000:
                per_page[page_idx] = 12 * 72  # 864 DPI for high-complexity vector pages
            else:
                per_page[page_idx] = no_text_dpi

    doc.close()
    global_dpi = _font_dpi(global_min) if global_min != float("inf") else no_text_dpi
    return global_dpi, per_page


# ──────────────────────────────────────────────────────────────────────────────
# PDF splitting
# ──────────────────────────────────────────────────────────────────────────────

def split_pdf_to_pages(
    pdf_path: Annotated[str, Field(description="Path to the input PDF")],
    output_dir: Annotated[str, Field(description="Directory to save page files")],
    dpi: int = 300,
    per_page_dpi: Optional[Dict[int, int]] = None,
) -> List[PageInfo]:
    """Split a PDF into per-page PNG + SVG files.

    Returns a list of PageInfo with paths and dimensions.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    pages: List[PageInfo] = []

    # Disable anti-aliasing for sharper lines in CAD/engineering drawings
    fitz.TOOLS.set_aa_level(0)

    for page_idx, page in enumerate(doc, start=1):
        page_dpi = (per_page_dpi or {}).get(page_idx, dpi)
        zoom = page_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=matrix)
        png_path = out / f"page{page_idx}.png"
        pix.save(str(png_path))

        svg_text = page.get_svg_image(matrix=fitz.Identity)
        svg_path = out / f"page{page_idx}.svg"
        svg_path.write_text(svg_text, encoding="utf-8")

        # Extract single-page PDF for DI (preserves vector line data)
        single_pdf_path = out / f"page{page_idx}.pdf"
        single_doc = fitz.open()
        single_doc.insert_pdf(doc, from_page=page_idx - 1, to_page=page_idx - 1)
        single_doc.save(str(single_pdf_path))
        single_doc.close()

        pages.append(PageInfo(
            page_num=page_idx,
            png_path=str(png_path),
            svg_path=str(svg_path),
            pdf_page_path=str(single_pdf_path),
            dpi=page_dpi,
            width=pix.width,
            height=pix.height,
        ))

    # Restore default anti-aliasing level
    fitz.TOOLS.set_aa_level(8)
    doc.close()
    return pages
