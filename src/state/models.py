"""Data models for pipeline state persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    """Overall pipeline run status."""
    CREATED = "created"
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(str, Enum):
    """Individual step status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"  # rolled back


class StepState(BaseModel):
    """Persistent state for a single pipeline step."""
    id: str = ""                          # Cosmos doc id: {run_id}_step{step_num}
    run_id: str
    step_num: int                         # 1-5
    step_name: str = ""                   # e.g. "step1", "step2a", "step2b"
    status: StepStatus = StepStatus.NOT_STARTED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_secs: Optional[float] = None
    error: Optional[str] = None

    # References to blob-stored artifacts
    blob_prefix: Optional[str] = None     # e.g. "{run_id}/step1/"
    artifact_paths: list[str] = Field(default_factory=list)  # image blob keys

    # Lightweight result data (metadata, not large blobs)
    result_data: dict[str, Any] = Field(default_factory=dict)

    def doc_id(self) -> str:
        return f"{self.run_id}_step{self.step_num}"


class RunState(BaseModel):
    """Persistent state for an entire pipeline run."""
    id: str = ""                          # Cosmos doc id = run_id
    run_id: str
    pdf_hash: str                         # SHA-256 of input PDF
    pdf_filename: str
    status: RunStatus = RunStatus.CREATED
    current_step: int = 0                 # highest completed step
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_pages: int = 0
    selected_pages: list[int] = Field(default_factory=list)

    # Step timing summary
    step_timings: dict[str, float] = Field(default_factory=dict)

    # Flags (mirrors session_state booleans)
    regions_confirmed: bool = False
    names_confirmed: bool = False
    crops_confirmed: bool = False
    bom_confirmed: bool = False


class CachedResult(BaseModel):
    """File-level cache entry for reusing results across runs."""
    id: str = ""                          # Cosmos doc id
    pdf_hash: str
    pdf_filename: str
    step_num: int
    blob_prefix: str                      # source blob prefix to copy from
    artifact_paths: list[str] = Field(default_factory=list)
    result_data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl: int = 2592000                    # 30 days (Cosmos TTL in seconds)
