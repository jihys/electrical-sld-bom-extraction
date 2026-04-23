"""Persistence helpers — bridge between Streamlit checkpoints and Azure storage.

This module wraps StateManager + BlobManager to provide simple sync functions
that the Streamlit app can call alongside existing _save_checkpoint() calls.

Usage in streamlit_app.py:
    from src.state.persistence import get_persistence, persist_step, restore_from_cloud

All functions are no-op when persistent state is disabled (enable_persistent_state=False).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PersistenceLayer:
    """Thin wrapper combining StateManager + BlobManager for Streamlit use.

    All methods are synchronous (Streamlit is sync) and safe to call
    even when Azure resources are not configured (graceful no-op).
    """

    def __init__(self, state_manager, blob_manager):
        self._state = state_manager
        self._blob = blob_manager
        self._enabled = True

    # ── Run lifecycle ──────────────────────────────────────────────────────

    def create_run(
        self,
        run_id: str,
        pdf_path: str,
        selected_pages: list[int] | None = None,
    ) -> str:
        """Create a new pipeline run. Returns run_id."""
        pdf_hash = self.compute_pdf_hash(pdf_path)
        filename = Path(pdf_path).name
        self._state.create_run(run_id, pdf_hash, filename, selected_pages)
        logger.info("Persistence: created run %s hash=%s", run_id, pdf_hash[:8])
        return run_id

    def get_run(self, run_id: str):
        return self._state.get_run(run_id)

    def list_runs(self, limit: int = 20):
        return self._state.list_runs(limit)

    # ── Step persistence ───────────────────────────────────────────────────

    def start_step(self, run_id: str, step_num: int, step_name: str = ""):
        """Mark step as started in Cosmos DB."""
        try:
            self._state.start_step(run_id, step_num, step_name)
        except Exception:
            logger.warning("Persistence: failed to start step %d", step_num, exc_info=True)

    def persist_step(
        self,
        run_id: str,
        step_num: int,
        step_name: str,
        *,
        result_data: dict[str, Any] | None = None,
        local_artifact_dir: str | Path | None = None,
        pdf_path: str | None = None,
    ):
        """Persist step results to Cosmos DB + upload artifacts to Blob.

        Args:
            run_id: Pipeline run ID
            step_num: Step number (1-5)
            step_name: Step name (e.g. "step1", "step2a")
            result_data: Lightweight metadata to store in Cosmos
            local_artifact_dir: Local directory with images/JSON to upload to Blob
            pdf_path: PDF path (for cache keying)
        """
        artifact_paths: list[str] = []
        blob_prefix = f"{run_id}/step{step_num}"

        # Upload artifacts to Blob
        if local_artifact_dir:
            try:
                artifact_paths = self._blob.upload_step_artifacts(
                    run_id, step_num, local_artifact_dir
                )
            except Exception:
                logger.warning(
                    "Persistence: blob upload failed for step %d", step_num, exc_info=True
                )

        # Save step state to Cosmos
        try:
            self._state.complete_step(
                run_id,
                step_num,
                result_data=result_data,
                artifact_paths=artifact_paths,
                blob_prefix=blob_prefix,
            )
        except Exception:
            logger.warning(
                "Persistence: cosmos save failed for step %d", step_num, exc_info=True
            )

        # Save to file cache (by pdf hash)
        if pdf_path and result_data:
            try:
                pdf_hash = self.compute_pdf_hash(pdf_path)
                filename = Path(pdf_path).name

                # Upload artifacts to cache container too
                if local_artifact_dir:
                    self._blob.upload_to_cache(pdf_hash, step_num, local_artifact_dir)

                self._state.save_cache(
                    pdf_hash, filename, step_num,
                    blob_prefix=f"{pdf_hash}/step{step_num}",
                    artifact_paths=artifact_paths,
                    result_data=result_data,
                )
            except Exception:
                logger.warning(
                    "Persistence: cache save failed for step %d", step_num, exc_info=True
                )

    def fail_step(self, run_id: str, step_num: int, error: str):
        """Mark step as failed."""
        try:
            self._state.fail_step(run_id, step_num, error)
        except Exception:
            logger.warning("Persistence: failed to record step failure", exc_info=True)

    # ── Restore / Rollback ─────────────────────────────────────────────────

    def get_step_state(self, run_id: str, step_num: int):
        """Get step state from Cosmos DB."""
        return self._state.get_step(run_id, step_num)

    def get_all_steps(self, run_id: str):
        """Get all step states for a run."""
        return self._state.get_all_steps(run_id)

    def restore_step_artifacts(
        self,
        run_id: str,
        step_num: int,
        local_dir: str | Path,
    ) -> list[Path]:
        """Download step artifacts from Blob to local dir."""
        try:
            return self._blob.download_step_artifacts(run_id, step_num, local_dir)
        except Exception:
            logger.warning(
                "Persistence: artifact download failed for step %d", step_num, exc_info=True
            )
            return []

    def rollback_to_step(self, run_id: str, target_step: int):
        """Invalidate steps after target_step in Cosmos DB."""
        try:
            return self._state.rollback_to_step(run_id, target_step)
        except Exception:
            logger.warning(
                "Persistence: rollback failed to step %d", target_step, exc_info=True
            )
            return []

    # ── Cache lookup ───────────────────────────────────────────────────────

    def find_cached_steps(self, pdf_path: str) -> dict[int, dict]:
        """Find all cached step results for a PDF file.

        Returns {step_num: result_data} for steps that have cached results.
        """
        pdf_hash = self.compute_pdf_hash(pdf_path)
        cached = self._state.find_all_cached_steps(pdf_hash)
        return {c.step_num: c.result_data for c in cached}

    def restore_from_cache(
        self,
        pdf_path: str,
        step_num: int,
        local_dir: str | Path,
    ) -> tuple[Optional[dict], list[Path]]:
        """Try to restore a step from the file cache.

        Returns (result_data, downloaded_files) or (None, []) if no cache.
        """
        pdf_hash = self.compute_pdf_hash(pdf_path)
        cached = self._state.find_cache(pdf_hash, step_num)
        if not cached:
            return None, []

        # Download cached artifacts
        try:
            files = self._blob.download_cache(pdf_hash, step_num, local_dir)
            return cached.result_data, files
        except Exception:
            logger.warning(
                "Persistence: cache restore failed for step %d", step_num, exc_info=True
            )
            return cached.result_data, []

    def has_cache(self, pdf_path: str, step_num: int) -> bool:
        """Check if cached results exist for this PDF + step."""
        pdf_hash = self.compute_pdf_hash(pdf_path)
        return self._state.find_cache(pdf_hash, step_num) is not None

    # ── Update run flags ───────────────────────────────────────────────────

    def update_run_flags(self, run_id: str, **flags):
        """Update confirmation flags on the run document."""
        try:
            from src.state.models import RunStatus
            run = self._state.get_run(run_id)
            if run:
                for k, v in flags.items():
                    if hasattr(run, k):
                        setattr(run, k, v)
                self._state.update_run(run)
        except Exception:
            logger.warning("Persistence: flag update failed", exc_info=True)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def compute_pdf_hash(pdf_path: str | Path) -> str:
        """Compute SHA-256 hash of a PDF file."""
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


class _NoOpPersistence:
    """Stub that silently ignores all persistence calls."""

    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return None if name != "find_cached_steps" else {}
        return noop


def get_persistence(settings) -> PersistenceLayer | _NoOpPersistence:
    """Factory: returns PersistenceLayer if Azure is configured, else no-op stub.

    Cached per Streamlit session via st.session_state.
    """
    import streamlit as st

    cache_key = "_persistence_layer"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # Check underlying fields directly (avoid pydantic property resolution issues)
    _has_cosmos = bool(getattr(settings, 'azure_cosmos_endpoint', '')) and getattr(settings, 'enable_persistent_state', False)
    _has_blob = bool(getattr(settings, 'azure_storage_blob_endpoint', '')) and getattr(settings, 'enable_persistent_state', False)
    if not _has_cosmos or not _has_blob:
        layer = _NoOpPersistence()
        st.session_state[cache_key] = layer
        return layer

    try:
        from src.state.state_manager import StateManager
        from src.state.blob_manager import BlobManager

        sm = StateManager(
            cosmos_endpoint=settings.azure_cosmos_endpoint,
            database_name=settings.azure_cosmos_database,
            cosmos_key=settings.azure_cosmos_key,
        )
        bm = BlobManager(
            blob_endpoint=settings.azure_storage_blob_endpoint,
            artifacts_container=settings.azure_storage_artifacts_container,
            cache_container=settings.azure_storage_cache_container,
        )
        layer = PersistenceLayer(sm, bm)
        st.session_state[cache_key] = layer
        logger.info("Persistence layer initialized (Cosmos + Blob)")
        return layer
    except Exception:
        logger.warning("Failed to init persistence layer, using no-op", exc_info=True)
        layer = _NoOpPersistence()
        st.session_state[cache_key] = layer
        return layer
