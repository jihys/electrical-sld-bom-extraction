"""Typed messages flowing between workflow executors.

Each message class matches the fields used by the corresponding
executor in workflow/executors.py.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from .page import PageInfo
from .region import PageExtractionSummary, RegionRecord
from .panel import BboxMatch, PanelCrop


# ── Pipeline messages (Executor → Executor) ──────────────────────────────────

class PdfInput(BaseModel):
    """Initial input: PDF file path + processing options."""
    pdf_path: str
    target_pages: Optional[List[int]] = None  # None = all pages
    dpi_override: Optional[int] = None


class PagesReady(BaseModel):
    """Output of SegmentExecutor."""
    pages: List[PageInfo]


class RegionsDetected(BaseModel):
    """Output of DetectExecutor — carries pages + DI results forward."""
    pages: List[PageInfo]
    summaries: List[PageExtractionSummary]
    di_lines_by_page: Dict[int, list]  # {page_num: [{"content":..., "bbox":...}, ...]}


class NamesExtracted(BaseModel):
    """Output of NameExtExecutor — carries full context forward."""
    pages: List[PageInfo]
    summaries: List[PageExtractionSummary]
    di_lines_by_page: Dict[int, list]
    names_by_page: Dict[int, List[str]]


class MatchesReady(BaseModel):
    """Output of MatchExecutor."""
    pages: List[PageInfo]
    names_by_page: Dict[int, List[str]]
    di_lines_by_page: Dict[int, list]
    matches_by_page: Dict[int, Dict[str, Optional[BboxMatch]]]


# ── Locate-Verify loop messages (internal, not used as inter-executor msgs) ──

class PanelLocateRequest(BaseModel):
    """Request to locate a panel's bbox."""
    page_num: int
    panel_name: str
    matched_bbox: Optional[Tuple[int, int, int, int]] = None
    page_img_path: str = ""
    crop_img_path: str = ""
    other_panels: Dict[str, Tuple[int, int, int, int]] = {}
    attempt: int = 0


class PanelVerifyRequest(BaseModel):
    """Request to verify a panel's bbox edges."""
    page_num: int
    panel_name: str
    current_bbox: Tuple[int, int, int, int]
    matched_bbox: Optional[Tuple[int, int, int, int]] = None
    page_img_path: str = ""
    attempt: int = 1
    axis_history: Dict[str, List[int]] = {}
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    best_score: int = 5


# ── Post-loop messages ───────────────────────────────────────────────────────

class PanelsCropped(BaseModel):
    """Output of LocateVerifyExecutor / ValidationExecutor."""
    pages: List[PageInfo]
    crops: List[PanelCrop]
    names_by_page: Dict[int, List[str]] = {}


class HitlReviewRequest(BaseModel):
    """Batch data sent to the human for bbox review via ctx.request_info()."""
    crops: List[PanelCrop]
    pages: List[PageInfo]
    reasons: Dict[str, str] = {}  # {panel_name: reason_string}


class BaySplitResult(BaseModel):
    """Individual bay within a panel."""
    bay_index: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    crop_path: Optional[str] = None


class FinalResult(BaseModel):
    """Terminal output of the entire pipeline."""
    summary: Dict[str, Any] = {}  # full JSON summary with panels + bay info
