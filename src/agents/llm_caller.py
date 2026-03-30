"""LLM caller utility — thin wrapper around Azure OpenAI Responses API."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from ..tools.image_tools import image_to_data_url


def create_llm_client(
    endpoint: str,
    api_key: Optional[str] = None,
    api_version: str = "2025-03-01-preview",
):
    """Create an AzureOpenAI client.

    Uses Managed Identity (matching the notebook setup).
    Falls back to API key only if explicitly provided AND working.
    """
    from openai import AzureOpenAI
    from azure.identity import ManagedIdentityCredential, get_bearer_token_provider

    token_provider = get_bearer_token_provider(
        ManagedIdentityCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )


def call_llm(
    client,
    deployment: str,
    prompt: str,
    image_paths: Optional[List[str]] = None,
    label: str = "",
    temperature: float = 0,
    system_prompt: Optional[str] = None,
) -> str:
    """Call Azure OpenAI Responses API with text + optional images.

    Returns the raw text output.
    """
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for img_path in (image_paths or []):
        url = image_to_data_url(img_path)
        content.append({"type": "input_image", "image_url": url})

    t0 = time.time()
    tag = f" [{label}]" if label else ""
    print(f"  LLM{tag} calling...", end=" ", flush=True)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    response = client.responses.create(
        model=deployment,
        input=messages,
        temperature=temperature,
    )

    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)", flush=True)
    return response.output_text


def parse_json(text: str) -> Optional[Dict]:
    """Extract a JSON object from LLM output, stripping markdown fences."""
    text = text.strip()
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


def sanitize_bbox(bbox, max_w: int, max_h: int) -> Optional[List[int]]:
    """Clamp bbox to image bounds and validate."""
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(max_w, x2), min(max_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]
