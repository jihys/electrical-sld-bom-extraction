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
    azure_openai_deployment: str = "gpt-4o"

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
