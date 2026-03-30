"""FastAPI application — serves the workflow + Gradio HITL UI."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import gradio as gr
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Settings
from ..hitl.gradio_app import create_gradio_app
from ..hitl.review_queue import review_queue
from ..models.workflow_state import PdfInput
from ..workflow.pipeline import build_pipeline

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

settings = Settings()
app = FastAPI(title="CAD Panel Extraction Agent", version="0.1.0")

# Mount Gradio HITL UI
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

# Static files for frontend
_static_dir = Path(__file__).resolve().parent.parent / "static"

# Active workflow runs
_active_runs: Dict[str, Any] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    pdf_path: str


class RunStatus(BaseModel):
    run_id: str
    status: str
    detail: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def root():
    """Serve the main frontend."""
    return FileResponse(_static_dir / "index.html")


@app.post("/api/upload", response_model=RunStatus)
async def upload_and_run(file: UploadFile = File(...)):
    """Upload a PDF file and immediately start extraction."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    run_id = str(uuid.uuid4())[:8]
    upload_dir = Path(settings.output_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save with run_id prefix to avoid collisions
    safe_name = Path(file.filename).name  # strip any directory components
    dest = upload_dir / f"{run_id}_{safe_name}"
    content = await file.read()
    dest.write_bytes(content)

    return await _start_run(run_id, str(dest))


@app.post("/api/run", response_model=RunStatus)
async def start_run(req: RunRequest):
    """Start a new panel extraction workflow run (server-local path)."""
    pdf_path = Path(req.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {req.pdf_path}")

    run_id = str(uuid.uuid4())[:8]
    return await _start_run(run_id, req.pdf_path)


async def _start_run(run_id: str, pdf_path: str) -> RunStatus:
    """Shared logic: build pipeline and launch workflow."""
    run_settings = settings.model_copy()
    run_settings.output_dir = str(Path(settings.output_dir) / f"run_{run_id}")
    run_settings.checkpoint_dir = str(Path(settings.checkpoint_dir) / f"run_{run_id}")

    builder = build_pipeline(run_settings)
    workflow = builder.build()

    task = asyncio.create_task(_run_workflow(run_id, workflow, pdf_path))
    _active_runs[run_id] = {"task": task, "status": "running", "workflow": workflow}

    return RunStatus(run_id=run_id, status="started")


@app.get("/api/run/{run_id}", response_model=RunStatus)
async def get_run_status(run_id: str):
    """Check the status of a workflow run."""
    run_info = _active_runs.get(run_id)
    if not run_info:
        raise HTTPException(status_code=404, detail="Run not found")

    task = run_info["task"]
    stored_status = run_info.get("status", "running")
    if task.done():
        if stored_status == "failed":
            return RunStatus(run_id=run_id, status="failed", detail=run_info.get("error"))
        exc = task.exception()
        if exc:
            return RunStatus(run_id=run_id, status="failed", detail=str(exc))
        return RunStatus(run_id=run_id, status="completed")

    return RunStatus(run_id=run_id, status=stored_status)


@app.get("/api/reviews")
async def get_pending_reviews():
    """Get all pending HITL review requests."""
    pending = await review_queue.get_pending()
    return [
        {
            "request_id": p["request_id"],
            "panels": [c.panel_name for c in p["crops"]],
            "reasons": p["reasons"],
        }
        for p in pending
    ]


@app.post("/api/reviews/{request_id}")
async def submit_review(request_id: str, corrections: Dict[str, Any]):
    """Submit HITL corrections for a review request."""
    ok = await review_queue.submit_response(request_id, corrections)
    if not ok:
        raise HTTPException(status_code=404, detail="Review request not found or expired")
    return {"status": "accepted", "request_id": request_id}


@app.get("/api/results/{run_id}")
async def get_results(run_id: str):
    """Get the final results of a completed run."""
    summary_path = Path(settings.output_dir) / f"run_{run_id}" / "final_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Results not ready yet")
    return json.loads(summary_path.read_text())


@app.get("/health")
async def health():
    return {"status": "ok"}


# ──────────────────────────────────────────────────────────────────────────────
# Workflow runner with HITL event handling
# ──────────────────────────────────────────────────────────────────────────────

async def _run_workflow(run_id: str, workflow, pdf_path: str):
    """Execute the workflow, handling request_info events for HITL."""
    input_msg = PdfInput(pdf_path=pdf_path)

    try:
        stream = workflow.run(input_msg, stream=True, include_status_events=True)
        async for event in stream:
            event_type = getattr(event, "type", None)

            if event_type == "request_info":
                request_id = event.request_id
                request_data = event.request_data

                # Route to HITL review queue
                page_paths = {}
                if hasattr(request_data, "pages"):
                    page_paths = {p.page_num: p.png_path for p in request_data.pages}

                await review_queue.add_request(
                    request_id=request_id,
                    crops=request_data.crops,
                    page_paths=page_paths,
                    reasons=getattr(request_data, "reasons", {}),
                )

                _active_runs[run_id]["status"] = "waiting_for_human"
                print(f"[Run {run_id}] Waiting for human review (request_id={request_id})")

                # Wait for human response
                response = await review_queue.wait_for_response(request_id)
                if response:
                    _active_runs[run_id]["status"] = "running"

        # Get final result
        final_result = await stream.get_final_response()
        _active_runs[run_id]["status"] = "completed"
        print(f"[Run {run_id}] Workflow completed")

    except Exception as e:
        _active_runs[run_id]["status"] = "failed"
        _active_runs[run_id]["error"] = str(e)
        print(f"[Run {run_id}] Workflow FAILED: {e}")
        import traceback
        traceback.print_exc()
