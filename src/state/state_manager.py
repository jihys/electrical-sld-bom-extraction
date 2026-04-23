"""Cosmos DB state manager for pipeline runs and step states."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from azure.cosmos import CosmosClient, ContainerProxy, PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from azure.identity import DefaultAzureCredential

from src.state.models import (
    CachedResult,
    RunState,
    RunStatus,
    StepState,
    StepStatus,
)

logger = logging.getLogger(__name__)


class StateManager:
    """Cosmos DB backed persistent state for pipeline runs.

    Containers:
        pipeline-runs  – one doc per run   (partition: /run_id)
        step-states    – one doc per step   (partition: /run_id)
        file-cache     – per-hash per-step  (partition: /pdf_hash)
    """

    def __init__(
        self,
        cosmos_endpoint: str,
        database_name: str = "sld-bom",
        credential: Optional[object] = None,
        cosmos_key: Optional[str] = None,
    ):
        if cosmos_key:
            self._client = CosmosClient(url=cosmos_endpoint, credential=cosmos_key)
        else:
            cred = credential or DefaultAzureCredential()
            self._client = CosmosClient(url=cosmos_endpoint, credential=cred)
        db = self._client.get_database_client(database_name)
        self._runs: ContainerProxy = db.get_container_client("pipeline-runs")
        self._steps: ContainerProxy = db.get_container_client("step-states")
        self._cache: ContainerProxy = db.get_container_client("file-cache")

    # ── Run CRUD ───────────────────────────────────────────────────────────

    def create_run(
        self,
        run_id: str,
        pdf_hash: str,
        pdf_filename: str,
        selected_pages: list[int] | None = None,
    ) -> RunState:
        """Create a new pipeline run record."""
        run = RunState(
            id=run_id,
            run_id=run_id,
            pdf_hash=pdf_hash,
            pdf_filename=pdf_filename,
            selected_pages=selected_pages or [],
        )
        self._runs.create_item(body=run.model_dump(mode="json"))
        logger.info("Created run %s for %s", run_id, pdf_filename)
        return run

    def get_run(self, run_id: str) -> Optional[RunState]:
        """Get a run by ID. Returns None if not found."""
        try:
            doc = self._runs.read_item(item=run_id, partition_key=run_id)
            return RunState(**doc)
        except CosmosResourceNotFoundError:
            return None

    def update_run(self, run: RunState) -> RunState:
        """Upsert the full run document."""
        run.updated_at = datetime.now(timezone.utc)
        self._runs.upsert_item(body=run.model_dump(mode="json"))
        return run

    def update_run_status(self, run_id: str, status: RunStatus, **kwargs: Any) -> RunState:
        """Update run status and optional extra fields."""
        run = self.get_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found")
        run.status = status
        for k, v in kwargs.items():
            if hasattr(run, k):
                setattr(run, k, v)
        return self.update_run(run)

    def list_runs(self, limit: int = 20) -> list[RunState]:
        """List recent runs, newest first."""
        query = "SELECT * FROM c ORDER BY c.created_at DESC OFFSET 0 LIMIT @limit"
        items = self._runs.query_items(
            query=query,
            parameters=[{"name": "@limit", "value": limit}],
            enable_cross_partition_query=True,
        )
        return [RunState(**doc) for doc in items]

    def find_runs_by_hash(self, pdf_hash: str) -> list[RunState]:
        """Find all runs for a given PDF hash."""
        query = "SELECT * FROM c WHERE c.pdf_hash = @hash ORDER BY c.created_at DESC"
        items = self._runs.query_items(
            query=query,
            parameters=[{"name": "@hash", "value": pdf_hash}],
            enable_cross_partition_query=True,
        )
        return [RunState(**doc) for doc in items]

    # ── Step CRUD ──────────────────────────────────────────────────────────

    def save_step(self, step: StepState) -> StepState:
        """Upsert a step state document."""
        step.id = step.doc_id()
        self._steps.upsert_item(body=step.model_dump(mode="json"))
        logger.debug("Saved step %s/%d status=%s", step.run_id, step.step_num, step.status)
        return step

    def get_step(self, run_id: str, step_num: int) -> Optional[StepState]:
        """Get a specific step state."""
        doc_id = f"{run_id}_step{step_num}"
        try:
            doc = self._steps.read_item(item=doc_id, partition_key=run_id)
            return StepState(**doc)
        except CosmosResourceNotFoundError:
            return None

    def get_all_steps(self, run_id: str) -> list[StepState]:
        """Get all steps for a run, ordered by step_num."""
        query = "SELECT * FROM c WHERE c.run_id = @run_id ORDER BY c.step_num"
        items = self._steps.query_items(
            query=query,
            parameters=[{"name": "@run_id", "value": run_id}],
            partition_key=run_id,
        )
        return [StepState(**doc) for doc in items]

    def start_step(
        self,
        run_id: str,
        step_num: int,
        step_name: str = "",
    ) -> StepState:
        """Mark a step as running."""
        step = self.get_step(run_id, step_num) or StepState(
            run_id=run_id, step_num=step_num
        )
        step.step_name = step_name or f"step{step_num}"
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now(timezone.utc)
        step.completed_at = None
        step.error = None
        return self.save_step(step)

    def complete_step(
        self,
        run_id: str,
        step_num: int,
        *,
        result_data: dict[str, Any] | None = None,
        artifact_paths: list[str] | None = None,
        blob_prefix: str | None = None,
    ) -> StepState:
        """Mark a step as completed with results."""
        step = self.get_step(run_id, step_num)
        if not step:
            raise ValueError(f"Step {run_id}/step{step_num} not found")
        step.status = StepStatus.COMPLETED
        step.completed_at = datetime.now(timezone.utc)
        if step.started_at:
            step.elapsed_secs = (step.completed_at - step.started_at).total_seconds()
        if result_data:
            step.result_data = result_data
        if artifact_paths:
            step.artifact_paths = artifact_paths
        if blob_prefix:
            step.blob_prefix = blob_prefix
        self.save_step(step)

        # Update run's current_step
        run = self.get_run(run_id)
        if run and step_num > run.current_step:
            run.current_step = step_num
            if step.elapsed_secs:
                run.step_timings[step.step_name or f"step{step_num}"] = step.elapsed_secs
            self.update_run(run)

        return step

    def fail_step(self, run_id: str, step_num: int, error: str) -> StepState:
        """Mark a step as failed."""
        step = self.get_step(run_id, step_num)
        if not step:
            step = StepState(run_id=run_id, step_num=step_num)
        step.status = StepStatus.FAILED
        step.completed_at = datetime.now(timezone.utc)
        step.error = error
        if step.started_at:
            step.elapsed_secs = (step.completed_at - step.started_at).total_seconds()
        return self.save_step(step)

    # ── Rollback ───────────────────────────────────────────────────────────

    def rollback_to_step(self, run_id: str, target_step: int) -> list[StepState]:
        """Invalidate all steps after target_step. Returns invalidated steps."""
        all_steps = self.get_all_steps(run_id)
        invalidated: list[StepState] = []

        for step in all_steps:
            if step.step_num > target_step:
                step.status = StepStatus.INVALIDATED
                self.save_step(step)
                invalidated.append(step)

        # Update run
        run = self.get_run(run_id)
        if run:
            run.current_step = target_step
            run.status = RunStatus.RUNNING
            # Clear confirmation flags after target
            if target_step < 2:
                run.regions_confirmed = False
            if target_step < 3:
                run.names_confirmed = False
            if target_step < 4:
                run.crops_confirmed = False
                run.bom_confirmed = False
            self.update_run(run)

        logger.info(
            "Rolled back run %s to step %d, invalidated %d steps",
            run_id, target_step, len(invalidated),
        )
        return invalidated

    # ── File Cache ─────────────────────────────────────────────────────────

    def save_cache(
        self,
        pdf_hash: str,
        pdf_filename: str,
        step_num: int,
        blob_prefix: str,
        artifact_paths: list[str] | None = None,
        result_data: dict[str, Any] | None = None,
    ) -> CachedResult:
        """Save a cache entry for a pdf_hash + step."""
        cache = CachedResult(
            id=f"{pdf_hash}_step{step_num}",
            pdf_hash=pdf_hash,
            pdf_filename=pdf_filename,
            step_num=step_num,
            blob_prefix=blob_prefix,
            artifact_paths=artifact_paths or [],
            result_data=result_data or {},
        )
        self._cache.upsert_item(body=cache.model_dump(mode="json"))
        logger.info("Cached result hash=%s step=%d", pdf_hash[:8], step_num)
        return cache

    def find_cache(self, pdf_hash: str, step_num: int) -> Optional[CachedResult]:
        """Look up cached result for a pdf_hash + step."""
        doc_id = f"{pdf_hash}_step{step_num}"
        try:
            doc = self._cache.read_item(item=doc_id, partition_key=pdf_hash)
            return CachedResult(**doc)
        except CosmosResourceNotFoundError:
            return None

    def find_all_cached_steps(self, pdf_hash: str) -> list[CachedResult]:
        """Find all cached steps for a given PDF hash."""
        query = "SELECT * FROM c WHERE c.pdf_hash = @hash ORDER BY c.step_num"
        items = self._cache.query_items(
            query=query,
            parameters=[{"name": "@hash", "value": pdf_hash}],
            partition_key=pdf_hash,
        )
        return [CachedResult(**doc) for doc in items]
