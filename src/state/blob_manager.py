"""Blob Storage manager for pipeline artifacts (images, JSON)."""

from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient, ContentSettings

logger = logging.getLogger(__name__)


class BlobManager:
    """Upload / download pipeline artifacts to Azure Blob Storage.

    Uses DefaultAzureCredential (Managed Identity, az login, etc.).
    """

    def __init__(
        self,
        blob_endpoint: str,
        artifacts_container: str = "pipeline-artifacts",
        cache_container: str = "file-cache",
        credential: Optional[object] = None,
    ):
        cred = credential or DefaultAzureCredential()
        self._client = BlobServiceClient(account_url=blob_endpoint, credential=cred)
        self._artifacts: ContainerClient = self._client.get_container_client(artifacts_container)
        self._cache: ContainerClient = self._client.get_container_client(cache_container)

    # ── Upload ─────────────────────────────────────────────────────────────

    def upload_file(
        self,
        local_path: str | Path,
        blob_name: str,
        *,
        container: str = "artifacts",
        overwrite: bool = True,
    ) -> str:
        """Upload a single file. Returns the blob name."""
        ctr = self._artifacts if container == "artifacts" else self._cache
        content_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        with open(local_path, "rb") as f:
            ctr.upload_blob(
                name=blob_name,
                data=f,
                overwrite=overwrite,
                content_settings=ContentSettings(content_type=content_type),
            )
        logger.debug("Uploaded %s → %s", local_path, blob_name)
        return blob_name

    def upload_step_artifacts(
        self,
        run_id: str,
        step_num: int,
        local_dir: str | Path,
        *,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".svg", ".json"),
    ) -> list[str]:
        """Upload all matching files from *local_dir* to blob storage.

        Blob layout: ``{run_id}/step{step_num}/relative/path``

        Returns list of blob names uploaded.
        """
        local_dir = Path(local_dir)
        if not local_dir.exists():
            logger.warning("upload_step_artifacts: %s does not exist", local_dir)
            return []

        uploaded: list[str] = []
        prefix = f"{run_id}/step{step_num}"

        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                fpath = Path(root) / fname
                if extensions and fpath.suffix.lower() not in extensions:
                    continue
                rel = fpath.relative_to(local_dir)
                blob_name = f"{prefix}/{rel}"
                self.upload_file(fpath, blob_name, container="artifacts")
                uploaded.append(blob_name)

        logger.info(
            "Uploaded %d artifacts for run=%s step=%d", len(uploaded), run_id, step_num
        )
        return uploaded

    # ── Download ───────────────────────────────────────────────────────────

    def download_file(
        self,
        blob_name: str,
        local_path: str | Path,
        *,
        container: str = "artifacts",
    ) -> Path:
        """Download a single blob to a local path."""
        ctr = self._artifacts if container == "artifacts" else self._cache
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            stream = ctr.download_blob(blob_name)
            stream.readinto(f)
        logger.debug("Downloaded %s → %s", blob_name, local_path)
        return local_path

    def download_step_artifacts(
        self,
        run_id: str,
        step_num: int,
        local_dir: str | Path,
    ) -> list[Path]:
        """Download all blobs for a step to *local_dir*. Returns local paths."""
        local_dir = Path(local_dir)
        prefix = f"{run_id}/step{step_num}/"
        downloaded: list[Path] = []

        for blob in self._artifacts.list_blobs(name_starts_with=prefix):
            rel = blob.name[len(prefix):]
            local_path = local_dir / rel
            self.download_file(blob.name, local_path)
            downloaded.append(local_path)

        logger.info(
            "Downloaded %d artifacts for run=%s step=%d", len(downloaded), run_id, step_num
        )
        return downloaded

    # ── Cache ──────────────────────────────────────────────────────────────

    def upload_to_cache(
        self,
        pdf_hash: str,
        step_num: int,
        local_dir: str | Path,
        *,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".svg", ".json"),
    ) -> list[str]:
        """Upload step artifacts to the file-cache container."""
        local_dir = Path(local_dir)
        if not local_dir.exists():
            return []

        uploaded: list[str] = []
        prefix = f"{pdf_hash}/step{step_num}"

        for root, _dirs, files in os.walk(local_dir):
            for fname in files:
                fpath = Path(root) / fname
                if extensions and fpath.suffix.lower() not in extensions:
                    continue
                rel = fpath.relative_to(local_dir)
                blob_name = f"{prefix}/{rel}"
                self.upload_file(fpath, blob_name, container="cache")
                uploaded.append(blob_name)

        return uploaded

    def copy_cache_to_run(
        self,
        pdf_hash: str,
        step_num: int,
        run_id: str,
    ) -> list[str]:
        """Copy cached blobs from file-cache → pipeline-artifacts for a new run.

        Returns list of destination blob names.
        """
        src_prefix = f"{pdf_hash}/step{step_num}/"
        dst_prefix = f"{run_id}/step{step_num}/"
        copied: list[str] = []

        for blob in self._cache.list_blobs(name_starts_with=src_prefix):
            rel = blob.name[len(src_prefix):]
            dst_name = f"{dst_prefix}{rel}"
            src_url = f"{self._cache.url}/{blob.name}"
            self._artifacts.get_blob_client(dst_name).start_copy_from_url(src_url)
            copied.append(dst_name)

        logger.info(
            "Copied %d cached blobs hash=%s step=%d → run=%s",
            len(copied), pdf_hash[:8], step_num, run_id,
        )
        return copied

    def download_cache(
        self,
        pdf_hash: str,
        step_num: int,
        local_dir: str | Path,
    ) -> list[Path]:
        """Download cached step artifacts to local_dir."""
        local_dir = Path(local_dir)
        prefix = f"{pdf_hash}/step{step_num}/"
        downloaded: list[Path] = []

        for blob in self._cache.list_blobs(name_starts_with=prefix):
            rel = blob.name[len(prefix):]
            local_path = local_dir / rel
            self.download_file(blob.name, local_path, container="cache")
            downloaded.append(local_path)

        return downloaded

    def has_cache(self, pdf_hash: str, step_num: int) -> bool:
        """Check if cache exists for a given pdf_hash + step."""
        prefix = f"{pdf_hash}/step{step_num}/"
        for _ in self._cache.list_blobs(name_starts_with=prefix):
            return True
        return False

    # ── Cleanup ────────────────────────────────────────────────────────────

    def delete_run_artifacts(self, run_id: str) -> int:
        """Delete all blobs for a run. Returns count deleted."""
        prefix = f"{run_id}/"
        count = 0
        for blob in self._artifacts.list_blobs(name_starts_with=prefix):
            self._artifacts.delete_blob(blob.name)
            count += 1
        logger.info("Deleted %d blobs for run=%s", count, run_id)
        return count

    # ── URL helpers ────────────────────────────────────────────────────────

    def get_blob_url(self, blob_name: str, container: str = "artifacts") -> str:
        """Return the full URL for a blob (no SAS — use RBAC)."""
        ctr = self._artifacts if container == "artifacts" else self._cache
        return f"{ctr.url}/{blob_name}"

    def list_step_blobs(self, run_id: str, step_num: int) -> list[str]:
        """List all blob names under a step prefix."""
        prefix = f"{run_id}/step{step_num}/"
        return [b.name for b in self._artifacts.list_blobs(name_starts_with=prefix)]
