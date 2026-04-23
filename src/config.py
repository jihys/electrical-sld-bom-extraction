"""Application configuration loaded from environment variables."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All configuration for the CAD panel extraction pipeline."""

    # ── Azure OpenAI ───────────────────────────────────────────────────────
    azure_openai_endpoint: str = ""
    azure_openai_api_key: Optional[str] = None
    azure_openai_api_version: str = "2025-03-01-preview"
    azure_openai_deployment: str = "gpt-5.4"

    # ── Azure Document Intelligence ───────────────────────────────────────
    azure_di_endpoint: str = ""
    azure_di_key: Optional[str] = None  # empty → DefaultAzureCredential
    azure_di_model_id: str = "prebuilt-layout"

    # ── Pipeline parameters ───────────────────────────────────────────────
    hitl_confidence_threshold: float = 0.7
    grid_size: int = 120
    verify_max_tries: int = 10
    max_detection_iterations: int = 3
    max_workflow_iterations: int = 50

    # ── Azure Cosmos DB (persistent state) ────────────────────────────────
    azure_cosmos_endpoint: str = ""
    azure_cosmos_database: str = "sld-bom"
    azure_cosmos_key: Optional[str] = None  # empty → DefaultAzureCredential

    # ── Azure Blob Storage (artifacts) ────────────────────────────────────
    azure_storage_account: str = ""
    azure_storage_blob_endpoint: str = ""
    azure_storage_artifacts_container: str = "pipeline-artifacts"
    azure_storage_cache_container: str = "file-cache"

    # ── Azure Service Bus (progress events) ───────────────────────────────
    azure_servicebus_namespace: str = ""
    azure_servicebus_tasks_queue: str = "pipeline-tasks"
    azure_servicebus_events_topic: str = "pipeline-events"

    # ── Feature flags ─────────────────────────────────────────────────────
    enable_persistent_state: bool = False  # set True to enable Cosmos/Blob persistence

    # ── Paths ─────────────────────────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def checkpoint_path(self) -> Path:
        p = Path(self.checkpoint_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_path(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def has_cosmos(self) -> bool:
        return bool(self.azure_cosmos_endpoint) and self.enable_persistent_state

    @property
    def has_blob_storage(self) -> bool:
        return bool(self.azure_storage_blob_endpoint) and self.enable_persistent_state
