"""HITL review queue — manages pending human review requests."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from ..models.panel import PanelCrop


class ReviewQueue:
    """Thread-safe queue for HITL review requests.

    Flow:
    1. ValidationExecutor calls ctx.request_info() → MAF emits request_info event
    2. FastAPI event handler catches it and calls queue.add_request()
    3. Gradio UI polls queue.get_pending() and shows bbox editor
    4. Human submits corrections → queue.submit_response()
    5. FastAPI calls workflow.run(responses={request_id: answer}) to resume
    """

    def __init__(self):
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._responses: Dict[str, str] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def add_request(
        self,
        request_id: str,
        crops: List[PanelCrop],
        page_paths: Dict[int, str],
        reasons: Dict[str, str],
    ) -> None:
        """Add a new HITL review request to the queue."""
        async with self._lock:
            self._pending[request_id] = {
                "crops": crops,
                "page_paths": page_paths,
                "reasons": reasons,
            }
            self._events[request_id] = asyncio.Event()

    async def get_pending(self) -> List[Dict[str, Any]]:
        """Get all pending review requests."""
        async with self._lock:
            return [
                {"request_id": rid, **data}
                for rid, data in self._pending.items()
            ]

    async def submit_response(
        self,
        request_id: str,
        corrections: Dict[str, Dict[str, Any]],
    ) -> bool:
        """Submit human corrections for a review request.

        corrections: {panel_name: {"bbox": [x1, y1, x2, y2]}, ...}
        Returns True if the request was found.
        """
        async with self._lock:
            if request_id not in self._pending:
                return False
            self._responses[request_id] = json.dumps(corrections)
            del self._pending[request_id]
            if request_id in self._events:
                self._events[request_id].set()
        return True

    async def wait_for_response(self, request_id: str, timeout: float = 3600) -> Optional[str]:
        """Wait for human response for a given request. Returns JSON string."""
        event = self._events.get(request_id)
        if not event:
            return None
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return self._responses.get(request_id)

    async def get_response(self, request_id: str) -> Optional[str]:
        """Get a submitted response (non-blocking)."""
        async with self._lock:
            return self._responses.get(request_id)


# Singleton instance
review_queue = ReviewQueue()
