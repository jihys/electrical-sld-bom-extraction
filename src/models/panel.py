"""Panel-level data models."""

from typing import List, Optional, Tuple

from pydantic import BaseModel


class BboxMatch(BaseModel):
    """A matched bbox for a panel name from DI text."""
    panel_name: str = ""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) page-coords
    lines: List[str] = []
    source_lines: List[str] = []
    method: str = ""  # e.g. "rule-single", "rule-alphanum", "llm"
    confidence: float = 1.0


class PanelInfo(BaseModel):
    """Extracted panel name with its text-region location."""
    name: str
    page_num: int
    matched_bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 1.0


class PanelCrop(BaseModel):
    """Final cropped panel result."""
    panel_name: str
    page_num: int = 0
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) on page image
    crop_path: Optional[str] = None
    confidence: float = 0.0
    verified_by: str = "agent"  # "agent" or "human"
    status: str = "pending"  # "pending", "verified", "failed"
    verify_attempts: int = 0
