"""Page-level data models."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class ContentBbox(BaseModel):
    """Content bounding box within a page (excludes title block / frame)."""
    x1: int
    y1: int
    x2: int
    y2: int


class PageInfo(BaseModel):
    """Metadata for a single rendered PDF page."""
    page_num: int
    png_path: str  # absolute path as string for serialization
    svg_path: Optional[str] = None
    dpi: int = 300
    width: int = 0
    height: int = 0
    content_bbox: Optional[ContentBbox] = None
