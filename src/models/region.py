"""Region detection data models."""

from typing import List, Optional, Tuple

from pydantic import BaseModel


class RegionRecord(BaseModel):
    """A detected region from DI or LLM supplementation."""
    kind: str  # "figure", "table", "text"
    polygon: List[float]  # flat [x1,y1,x2,y1,x2,y2,x1,y2]
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    page_num: int
    confidence: Optional[float] = None
    source: str = "di"  # "di" or "llm"


class PageExtractionSummary(BaseModel):
    """Summary of DI extraction results for a single page."""
    page_num: int
    figure_regions: List[RegionRecord] = []
    table_regions: List[RegionRecord] = []
    text_regions: List[RegionRecord] = []

    @property
    def n_figures(self) -> int:
        return len(self.figure_regions)

    @property
    def n_tables(self) -> int:
        return len(self.table_regions)

    @property
    def all_regions(self) -> List[RegionRecord]:
        return self.figure_regions + self.table_regions + self.text_regions
