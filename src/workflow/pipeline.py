"""Workflow pipeline definition using Microsoft Agent Framework WorkflowBuilder."""

from __future__ import annotations

from pathlib import Path

from agent_framework import FileCheckpointStorage, WorkflowBuilder

from ..config import Settings
from .executors import (
    BaySplitExecutor,
    DetectExecutor,
    LocateVerifyExecutor,
    MatchExecutor,
    NameExtExecutor,
    SegmentExecutor,
    ValidationExecutor,
)


def build_pipeline(settings: Settings) -> WorkflowBuilder:
    """Construct the full panel extraction workflow graph.

    Graph topology:
      Segment → Detect → NameExt → Match → LocateVerify → Validation → BaySplit
                                                               ↕
                                                         HITL (request_info)
    """
    # Instantiate executors
    segment_ex = SegmentExecutor(settings)
    detect_ex = DetectExecutor(settings)
    name_ext_ex = NameExtExecutor(settings)
    match_ex = MatchExecutor(settings)
    locate_verify_ex = LocateVerifyExecutor(settings)
    validation_ex = ValidationExecutor(settings)
    bay_split_ex = BaySplitExecutor(settings)

    # Checkpoint storage
    checkpoint_dir = Path(settings.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_storage = FileCheckpointStorage(storage_path=str(checkpoint_dir))

    # Build linear pipeline with HITL at validation stage
    builder = (
        WorkflowBuilder(
            start_executor=segment_ex,
            checkpoint_storage=checkpoint_storage,
        )
        .add_edge(segment_ex, detect_ex)
        .add_edge(detect_ex, name_ext_ex)
        .add_edge(name_ext_ex, match_ex)
        .add_edge(match_ex, locate_verify_ex)
        .add_edge(locate_verify_ex, validation_ex)
        .add_edge(validation_ex, bay_split_ex)
    )

    return builder
