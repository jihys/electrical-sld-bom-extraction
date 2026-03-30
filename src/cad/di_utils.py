"""Azure Document Intelligence client utilities."""

from pathlib import Path
from typing import List, Optional

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from azure.core.credentials import AzureKeyCredential

ANALYSIS_FEATURES: List[DocumentAnalysisFeature] = [
    DocumentAnalysisFeature.OCR_HIGH_RESOLUTION,
    DocumentAnalysisFeature.KEY_VALUE_PAIRS,
    DocumentAnalysisFeature.STYLE_FONT,
]


def create_di_client(endpoint: str, key: Optional[str] = None) -> DocumentIntelligenceClient:
    """Create and return an Azure Document Intelligence client.

    If ``key`` is provided, uses AzureKeyCredential.
    Otherwise falls back to DefaultAzureCredential (AAD token auth).
    """
    if key:
        credential = AzureKeyCredential(key)
    else:
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()
    return DocumentIntelligenceClient(endpoint=endpoint, credential=credential)


def analyze_page_with_document_intelligence(
    client: DocumentIntelligenceClient,
    model_id: str,
    image_path: Path,
    extract_tables: bool = False,
    extract_text: bool = False,
):
    """Submit a page image or PDF to Document Intelligence and return the parsed result.

    By default only figure regions are retained in the result.
    Set extract_tables=True or extract_text=True to keep those regions as well.
    """
    _ext = image_path.suffix.lower()
    _content_type_map = {
        ".pdf":  "application/pdf",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    _content_type = _content_type_map.get(_ext, "application/octet-stream")
    with image_path.open("rb") as data_stream:
        poller = client.begin_analyze_document(
            model_id=model_id,
            body=data_stream,
            content_type=_content_type,
            features=ANALYSIS_FEATURES,
        )
    result = poller.result()
    if not extract_tables:
        result.tables = []
    if not extract_text:
        result.paragraphs = []

    # PDF input returns coordinates in inches; normalize to 0-1 so downstream
    # code (which uses max(coords) <= 1.5 to detect normalized coords) works
    # consistently regardless of input type.
    if _ext == ".pdf":
        _normalize_pdf_polygons(result)

    return result


def _normalize_pdf_polygons(result) -> None:
    """Convert all bounding_region polygons from inches to normalized (0-1) coords.

    Azure DI returns inch-based coordinates for PDF inputs. All other logic in
    this project assumes normalized 0-1 coordinates, so we convert in-place.
    """
    # Build page_index → (width_inch, height_inch) lookup
    page_dims = {}
    for page in (result.pages or []):
        pw = getattr(page, "width", None)
        ph = getattr(page, "height", None)
        pi = getattr(page, "page_number", None)
        if pw and ph and pi:
            page_dims[pi] = (pw, ph)

    if not page_dims:
        return

    def _norm_region(regions):
        if not regions:
            return
        for region in regions:
            pn = getattr(region, "page_number", 1)
            pw, ph = page_dims.get(pn, (None, None))
            if pw is None or ph is None:
                continue
            poly = getattr(region, "polygon", None)
            if not poly:
                continue
            if max(poly) <= 1.5:
                continue  # already normalized
            region.polygon = [
                v / (pw if i % 2 == 0 else ph) for i, v in enumerate(poly)
            ]

    for fig in (result.figures or []):
        _norm_region(getattr(fig, "bounding_regions", None))
        for elem in (getattr(fig, "elements", None) or []):
            _norm_region(getattr(elem, "bounding_regions", None))

    for tbl in (result.tables or []):
        _norm_region(getattr(tbl, "bounding_regions", None))
        for cell in (getattr(tbl, "cells", None) or []):
            _norm_region(getattr(cell, "bounding_regions", None))

    for para in (result.paragraphs or []):
        _norm_region(getattr(para, "bounding_regions", None))

    for word in (getattr(result, "words", None) or []):
        _norm_region(getattr(word, "bounding_regions", None))
