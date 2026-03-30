"""Azure Document Intelligence tools."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

from pydantic import Field

from ..models.region import PageExtractionSummary, RegionRecord


# ── Client factory ────────────────────────────────────────────────────────────

def create_di_client(endpoint: str, key: Optional[str] = None):
    """Create an Azure Document Intelligence client.

    Uses Managed Identity (matching the notebook setup).
    """
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import DocumentAnalysisFeature
    from azure.identity import ManagedIdentityCredential

    credential = ManagedIdentityCredential()
    return DocumentIntelligenceClient(endpoint=endpoint, credential=credential)


# ── Single-page analysis ─────────────────────────────────────────────────────

def analyze_page(
    di_client,
    model_id: str,
    image_path: Annotated[str, Field(description="Path to the image to analyze")],
    max_retries: int = 8,
) -> Dict:
    """Run DI on a single image. Returns raw lines and figure regions.

    Retries automatically on HTTP 429 (rate limit) with exponential backoff.
    """
    from azure.ai.documentintelligence.models import DocumentAnalysisFeature
    from azure.core.exceptions import HttpResponseError

    features = [
        DocumentAnalysisFeature.OCR_HIGH_RESOLUTION,
        DocumentAnalysisFeature.KEY_VALUE_PAIRS,
        DocumentAnalysisFeature.STYLE_FONT,
    ]

    path = Path(image_path)
    ext = path.suffix.lower()
    content_type_map = {
        ".pdf": "application/pdf", ".png": "image/png",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")

    for attempt in range(max_retries + 1):
        try:
            with path.open("rb") as f:
                poller = di_client.begin_analyze_document(
                    model_id=model_id, body=f, content_type=content_type, features=features,
                )
            result = poller.result()
            break
        except HttpResponseError as exc:
            if exc.status_code == 429 and attempt < max_retries:
                wait = min(2 ** attempt, 60)
                time.sleep(wait)
                continue
            raise

    lines = []
    for page in (result.pages or []):
        for line in (page.lines or []):
            poly = [float(v) for v in line.polygon]
            xs, ys = poly[0::2], poly[1::2]
            lines.append({
                "content": line.content,
                "polygon": poly,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
            })

    figures = []
    for fig in (result.figures or []):
        for br in (getattr(fig, "bounding_regions", None) or []):
            poly = [float(v) for v in br.polygon]
            xs, ys = poly[0::2], poly[1::2]
            figures.append({
                "polygon": poly,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "page_number": br.page_number,
            })

    return {"lines": lines, "figures": figures}


# ── Two-pass detection ────────────────────────────────────────────────────────

def two_pass_detection(
    di_client,
    model_id: str,
    image_path: Annotated[str, Field(description="Path to page PNG")],
    output_dir: Annotated[str, Field(description="Directory for intermediate files")],
    page_num: int = 1,
) -> PageExtractionSummary:
    """Run 2-pass DI detection: original → white-fill → re-detect missed regions.

    Returns a deduplicated PageExtractionSummary.
    """
    from ..tools.image_tools import white_fill_regions

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Pass 1: original image
    pass1 = analyze_page(di_client, model_id, image_path)
    pass1_bboxes = [[int(v) for v in f["bbox"]] for f in pass1["figures"]]

    # Pass 2: white-fill pass1 regions, re-detect
    if pass1_bboxes:
        whitefill_path = str(out / f"page{page_num}_whitefill.png")
        white_fill_regions(image_path, pass1_bboxes, whitefill_path)
        pass2 = analyze_page(di_client, model_id, whitefill_path)
    else:
        pass2 = {"figures": [], "lines": []}

    # Merge and dedup
    all_figures = pass1["figures"] + pass2["figures"]
    deduped = _dedup_figure_bboxes(all_figures)

    figure_regions = [
        RegionRecord(
            kind="figure",
            polygon=fig["polygon"],
            bbox=tuple(int(v) for v in fig["bbox"]),
            page_num=page_num,
            source="di",
        )
        for fig in deduped
    ]

    return PageExtractionSummary(
        page_num=page_num,
        figure_regions=figure_regions,
    )


def _dedup_figure_bboxes(figures: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """Remove duplicate figures by IoU overlap."""
    if not figures:
        return []

    from ..tools.geometry_tools import compute_iou

    sorted_figs = sorted(figures, key=lambda f: _bbox_area(f["bbox"]), reverse=True)
    kept = []
    for fig in sorted_figs:
        bbox = fig["bbox"]
        is_dup = False
        for k in kept:
            if compute_iou(tuple(bbox), tuple(k["bbox"])) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(fig)
    return kept


def _bbox_area(bbox) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
