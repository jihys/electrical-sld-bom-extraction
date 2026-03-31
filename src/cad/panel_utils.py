import base64
import json
import re
import time
from typing import List, Optional

import cv2
import numpy as np
from openai import AzureOpenAI


def safe_name(s: str) -> str:
    x = re.sub(r"[^\w\-.]+", "_", s.strip())
    return x[:80] if x else "panel"


def sanitize_bbox(bbox, w: int, h: int):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
    except Exception:
        return None
    x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def parse_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def _img_to_data_url(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# Global list to collect LLM call timing records.
# Each entry: {"label", "reasoning_effort", "elapsed_s", "n_images", "timestamp", "source"}
llm_call_log: List[dict] = []


def log_llm_call(label: str, elapsed_s: float, n_images: int = 0,
                 reasoning_effort: str = "", source: str = "") -> None:
    """Append a timing record to the global llm_call_log."""
    llm_call_log.append({
        "label": label,
        "reasoning_effort": reasoning_effort,
        "elapsed_s": elapsed_s,
        "n_images": n_images,
        "timestamp": time.time(),
        "source": source,
    })


def call_llm(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    images: List[np.ndarray],
    label: str = "",
    reasoning_effort: str = "medium",
) -> str:
    content = [{"type": "input_text", "text": prompt}]
    for im in images:
        content.append({"type": "input_image", "image_url": _img_to_data_url(im), "detail": "original"})

    t0 = time.time()
    tag = f"[{label}]" if label else ""
    print(f"    LLM{tag} calling (effort={reasoning_effort})... ", end="", flush=True)
    resp = client.responses.create(
        model=deployment,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": reasoning_effort},
    )
    elapsed = round(time.time() - t0, 3)
    print(f"done ({elapsed:.1f}s)")

    log_llm_call(label, elapsed, len(images), reasoning_effort, source="panel_utils")
    return resp.output_text