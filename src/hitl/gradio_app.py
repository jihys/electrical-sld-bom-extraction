"""Gradio HITL Bbox Editor — drag-and-drop bbox correction UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from .review_queue import review_queue


def _draw_review_image(
    page_img_path: str,
    crops: list,
    reasons: Dict[str, str],
) -> np.ndarray:
    """Draw all review panels on the page image with annotations."""
    img = cv2.imread(page_img_path)
    if img is None:
        return np.zeros((600, 800, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for crop in crops:
        x1, y1, x2, y2 = crop.bbox
        color = (0, 165, 255)  # orange
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        reason = reasons.get(crop.panel_name, "")
        label = f"{crop.panel_name} ({reason[:30]})"
        cv2.putText(img, label, (x1, max(y1 - 8, 18)), font, 0.55, color, 2, cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_gradio_app() -> gr.Blocks:
    """Build the Gradio HITL bbox editor interface."""

    with gr.Blocks(title="CAD Panel HITL Review", theme=gr.themes.Soft()) as app:
        gr.Markdown("# CAD Panel HITL Bbox Editor")
        gr.Markdown("Review and correct panel bounding boxes flagged for human review.")

        state = gr.State(value={})

        with gr.Row():
            refresh_btn = gr.Button("Refresh Pending Reviews", variant="primary")
            request_dropdown = gr.Dropdown(
                label="Select Review Request", choices=[], interactive=True,
            )

        with gr.Row():
            with gr.Column(scale=2):
                review_image = gr.Image(label="Page with Panel Bboxes", type="numpy")
            with gr.Column(scale=1):
                panel_info = gr.JSON(label="Panel Details")
                gr.Markdown("### Edit Bbox Coordinates")
                with gr.Group():
                    panel_selector = gr.Dropdown(label="Panel Name", choices=[])
                    x1_input = gr.Number(label="x1", precision=0)
                    y1_input = gr.Number(label="y1", precision=0)
                    x2_input = gr.Number(label="x2", precision=0)
                    y2_input = gr.Number(label="y2", precision=0)
                    approve_btn = gr.Button("Approve / Update Bbox", variant="secondary")

        submit_btn = gr.Button("Submit All Corrections", variant="primary", size="lg")
        status_msg = gr.Textbox(label="Status", interactive=False)

        # ── Event handlers ──

        async def refresh_requests():
            pending = await review_queue.get_pending()
            choices = [p["request_id"] for p in pending]
            return gr.update(choices=choices, value=choices[0] if choices else None)

        async def load_request(request_id, current_state):
            if not request_id:
                return None, {}, gr.update(choices=[]), current_state

            pending = await review_queue.get_pending()
            req = next((p for p in pending if p["request_id"] == request_id), None)
            if not req:
                return None, {}, gr.update(choices=[]), current_state

            crops = req["crops"]
            reasons = req["reasons"]
            page_paths = req.get("page_paths", {})

            # Use first page path available
            page_path = next(iter(page_paths.values()), None) if page_paths else None
            img = None
            if page_path:
                img = _draw_review_image(page_path, crops, reasons)

            panel_names = [c.panel_name for c in crops]
            info = {
                c.panel_name: {
                    "bbox": list(c.bbox),
                    "confidence": c.confidence,
                    "reason": reasons.get(c.panel_name, ""),
                    "status": c.status,
                }
                for c in crops
            }

            new_state = {
                "request_id": request_id,
                "corrections": {c.panel_name: {"bbox": list(c.bbox)} for c in crops},
                "page_path": page_path,
                "crops": crops,
                "reasons": reasons,
            }

            return img, info, gr.update(choices=panel_names, value=panel_names[0] if panel_names else None), new_state

        def load_panel_bbox(panel_name, current_state):
            corrections = current_state.get("corrections", {})
            if panel_name and panel_name in corrections:
                bbox = corrections[panel_name]["bbox"]
                return bbox[0], bbox[1], bbox[2], bbox[3]
            return 0, 0, 0, 0

        def update_bbox(panel_name, x1, y1, x2, y2, current_state):
            if panel_name and current_state:
                current_state.setdefault("corrections", {})[panel_name] = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                }
            return current_state, f"Updated bbox for {panel_name}"

        async def submit_corrections(current_state):
            request_id = current_state.get("request_id")
            corrections = current_state.get("corrections", {})
            if not request_id:
                return "No active review request"
            ok = await review_queue.submit_response(request_id, corrections)
            if ok:
                return f"Submitted corrections for {len(corrections)} panels. Workflow will resume."
            return "Failed to submit — request may have expired."

        # ── Wire events ──
        refresh_btn.click(refresh_requests, outputs=[request_dropdown])
        request_dropdown.change(
            load_request,
            inputs=[request_dropdown, state],
            outputs=[review_image, panel_info, panel_selector, state],
        )
        panel_selector.change(
            load_panel_bbox,
            inputs=[panel_selector, state],
            outputs=[x1_input, y1_input, x2_input, y2_input],
        )
        approve_btn.click(
            update_bbox,
            inputs=[panel_selector, x1_input, y1_input, x2_input, y2_input, state],
            outputs=[state, status_msg],
        )
        submit_btn.click(submit_corrections, inputs=[state], outputs=[status_msg])

    return app
