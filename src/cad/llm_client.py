"""
llm_client.py
-------------
Azure OpenAI client setup, schemas, prompts, and API call wrappers.
All functions accept explicit parameters (client, reasoning_effort, etc.)
instead of relying on module-level globals.
Extracted/adapted from 02_missing_image_detection.ipynb.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

from .image_utils import (
    GRID_STEP_X,
    GRID_STEP_Y,
    draw_grid_with_cell_numbers,
    image_to_data_url,
    file_to_data_url,
)
from .data_types import ImageCropInfo

load_dotenv()

# ── Schemas ────────────────────────────────────────────────────────────────────

VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["complete", "missing", "no-images"]},
        "full_page": {"type": "boolean"},
        "missing": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "polygon": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 8,
                    },
                    "note": {"type": "string"},
                },
                "required": ["polygon"],
            },
        },
    },
    "required": ["status", "missing"],
}

DI_CROP_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "is_clipped": {
            "type": "boolean",
            "description": "True if electrical circuit content is cut off at any edge of this crop",
        },
        "clipped_edges": {
            "type": "array",
            "items": {"type": "string", "enum": ["top", "bottom", "left", "right"]},
            "description": "Which edges show circuit content being cut off",
        },
        "excess_edges": {
            "type": "array",
            "items": {"type": "string", "enum": ["top", "bottom", "left", "right"]},
            "description": "Which edges have large unnecessary empty margins that can be trimmed without losing any circuit content",
        },
        "suggested_box": {
            "type": "object",
            "description": "Suggested bounding box (pixel coords). Required when is_clipped=true OR excess_edges is non-empty. Expand clipped edges outward; shrink excess edges inward.",
            "properties": {
                "x1": {"type": "number", "description": "Left edge pixel"},
                "y1": {"type": "number", "description": "Top edge pixel"},
                "x2": {"type": "number", "description": "Right edge pixel"},
                "y2": {"type": "number", "description": "Bottom edge pixel"},
            },
            "required": ["x1", "y1", "x2", "y2"],
        },
        "note": {"type": "string", "description": "Brief explanation of clipping/excess found and how the box was adjusted"},
    },
    "required": ["is_clipped", "clipped_edges", "excess_edges"],
}


VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_correct": {"type": "boolean"},
        "issue": {"type": "string"},
        "corrected_polygon": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 8,
        },
    },
    "required": ["is_correct"],
}

DIRECTIONAL_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_correct": {"type": "boolean"},
        "issue": {"type": "string"},
        "x1": {
            "type": "object",
            "description": "Left edge analysis",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["right", "left", "none"],
                    "description": (
                        "left=move left edge further left (include more content on left side); "
                        "right=move left edge right (trim excess whitespace); "
                        "none=correct"
                    ),
                },
                "corrected": {"type": "integer", "description": "Corrected x1 pixel value"},
            },
            "required": ["direction", "corrected"],
        },
        "y1": {
            "type": "object",
            "description": "Top edge analysis",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["down", "up", "none"],
                    "description": (
                        "up=move top edge up (include more content at top, e.g. area designation boxes); "
                        "down=move top edge down (trim excess whitespace at top); "
                        "none=correct"
                    ),
                },
                "corrected": {"type": "integer", "description": "Corrected y1 pixel value"},
            },
            "required": ["direction", "corrected"],
        },
        "x2": {
            "type": "object",
            "description": "Right edge analysis",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["right", "left", "none"],
                    "description": (
                        "right=move right edge further right (include more content on right side); "
                        "left=move right edge left (trim excess whitespace); "
                        "none=correct"
                    ),
                },
                "corrected": {"type": "integer", "description": "Corrected x2 pixel value"},
            },
            "required": ["direction", "corrected"],
        },
        "y2": {
            "type": "object",
            "description": "Bottom edge analysis",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["down", "up", "none"],
                    "description": (
                        "down=move bottom edge down (include more content at bottom, e.g. motors/terminals connected by dashed lines); "
                        "up=move bottom edge up (trim excess whitespace at bottom); "
                        "none=correct"
                    ),
                },
                "corrected": {"type": "integer", "description": "Corrected y2 pixel value"},
            },
            "required": ["direction", "corrected"],
        },
        "corrected_polygon": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Corrected polygon [x1,y1, x2,y1, x2,y2, x1,y2] using corrected edge values",
        },
    },
    "required": ["is_correct", "x1", "y1", "x2", "y2"],
}


# ── Client factory ─────────────────────────────────────────────────────────────

def make_client() -> tuple[AzureOpenAI, str]:
    """Return (client, deployment_name). Reads env vars."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    if not endpoint or not deployment:
        raise EnvironmentError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT in .env")
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        timeout=120,
    )
    return client, deployment


# ── JSON parsing ───────────────────────────────────────────────────────────────

def safe_parse_json(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


# ── Prompts ────────────────────────────────────────────────────────────────────

def _build_detection_prompt(
    crop_boxes: List[Dict],
    image_size: Tuple[int, int],
    content_bbox: Optional[Tuple[int, int, int, int]] = None,
    existing_coverage_pct: Optional[float] = None,
) -> str:
    """Detection prompt (v4) — with optional content_bbox layout hint."""
    content_hint = ""
    if content_bbox:
        cx1, cy1, cx2, cy2 = content_bbox
        cw, ch = cx2 - cx1, cy2 - cy1
        content_hint = (
            f"\n\n**📐 LAYOUT HINT (deterministically computed from image):**\n"
            f"The CONTENT AREA of this page (outer border frame + title block removed) spans:\n"
            f"  → x1={cx1}, y1={cy1}, x2={cx2}, y2={cy2}  (content size: {cw}×{ch}px)\n"
            f"⚠️ If the entire content area is ONE interconnected circuit system, return a SINGLE polygon "
            f"[{cx1},{cy1}, {cx2},{cy1}, {cx2},{cy2}, {cx1},{cy2}] covering this full content bbox.\n"
        )

    coverage_warning = ""
    if existing_coverage_pct is not None and existing_coverage_pct < 60.0:
        coverage_warning = (
            f"\n\n**⚠️ COVERAGE WARNING: Already-captured regions cover only {existing_coverage_pct:.0f}% of the content area!**\n"
            "This is LOW — there are very likely ADDITIONAL circuit regions not yet captured.\n"
            "🔍 Systematically scan every area of the page for additional circuit regions.\n"
            "❌ Do NOT return status='complete' unless you can confirm ALL circuit regions have been captured.\n"
        )

    return (
        "Find MISSING image regions in this electrical/mechanical engineering drawing that were not detected previously. "
        "\n\n**🔌 What are ELECTRICAL CIRCUIT DIAGRAMS (IMAGE REGIONS)?**\n"
        "Electrical single-line diagrams are IMAGE REGIONS that contain:\n"
        "✅ Visual SYMBOLS: transformers (⚡), circuit breakers (□), relays (○), motors (M), switches, contactors\n"
        "✅ CONNECTION LINES: vertical/horizontal/diagonal lines (solid or dashed) showing power flow\n"
        "✅ HIERARCHICAL FLOW: power source → transformer → breaker → load (top-to-bottom)\n"
        "✅ SYSTEM DIAGRAMS: showing how electrical components are interconnected\n"
        "\n❌ EXCLUDE (NOT image regions):\n"
        "- Pure data tables: REVISION DESCRIPTION, RELAY FUNCTION TABLE (rows/columns of text only)\n"
        "- Document metadata tables, title blocks, standalone notes\n"
        "- Page header strips (very wide, very short regions at the top of the page — e.g. drawing number, room name)\n"
        "- Standalone notes/legend/notices boxes that contain only text and tables, no circuit symbols\n"
        "\n**Key distinction**: \n"
        "- Circuit diagram = SYMBOLS + CONNECTION LINES + electrical flow → IMAGE REGION ✅\n"
        "- Data table / notes box / page header = text only, no symbols → NOT an image region ❌\n"
        + content_hint
        + coverage_warning
        + "\n\n**📸 Image description:** The image shows the full page with ORANGE boxes marking already-captured regions.\n"
        + "\n\nJSON schema: " + json.dumps(VALIDATION_SCHEMA)
        + "\n\nExisting boxes: " + json.dumps(crop_boxes)
        + "\n\nFull image size (width, height): " + json.dumps(image_size)
        + "\n\n**RULES (in priority order):**\n"
        "1) 🚨 CRITICAL: Include COMPLETE circuit diagram from TOP to BOTTOM:\n"
        "   - Include ALL connection lines (solid AND dashed) to their endpoints\n"
        "   - Include ALL components at the BOTTOM connected by lines\n"
        "   - Better to include EXTRA space than cut off any content\n"
        "   - DO NOT cut off motors/terminals at the bottom\n"
        "2) 🚨 If ENTIRE page is one large diagram with no separate text areas, set full_page=true and return full-page polygon.\n"
        "3) 🚨 If the content area (see LAYOUT HINT above) is filled with ONE interconnected circuit system\n"
        "   (multiple sub-sections connected by shared bus bars, distribution lines, or common ground),\n"
        "   return ONE polygon for the ENTIRE content area — do NOT split sub-sections into separate polygons.\n"
        "   A diagram can have multiple sub-sections and still be ONE unified circuit.\n"
        "4) 🚨 Do NOT return polygons that overlap with or ENCOMPASS any ORANGE (already-captured) region.\n"
        "   - ❌ FORBIDDEN: A new polygon whose area INCLUDES an orange region inside it\n"
        "   - ❌ FORBIDDEN: A new polygon that fully surrounds or wraps around an orange region\n"
        "   - If the only remaining region is adjacent to (but does NOT contain) an orange region, that is OK.\n"
        "   - THINK: 'Would any orange region fit entirely inside my new polygon?' → If YES, reject it.\n"
        "5) Each polygon should be [x1,y1, x2,y2, x3,y3, x4,y4] for the four corners of a rectangular region.\n"
        "6) If no NEW image regions need extraction (all are in existing_boxes), return status='complete'.\n"
        "7) If there are no image regions at all on this page, return status='no-images'.\n"
        "\n**🚨 COMMON MISTAKES TO AVOID:**\n"
        "❌ Detecting only LEFT or RIGHT portion of a diagram that spans the full width\n"
        "❌ Cutting off motors/terminals at the bottom connected by lines\n"
        "❌ Splitting one interconnected multi-section diagram into multiple separate polygons\n"
        "❌ Detecting page header strips or standalone notes/legend boxes as circuit diagram regions\n"
        "❌ Proposing a LARGE polygon that encompasses an orange (already-captured) region as a sub-region\n"
        "\n✅ CORRECT APPROACH:\n"
        "- Check if all visible circuit sections share a common power bus or distribution line → ONE diagram\n"
        "- Use the LAYOUT HINT content bbox as the starting point for a full-width diagram\n"
        "- Identify the complete circuit from topmost to bottommost elements\n"
        "- When in doubt, use the full content bbox (see LAYOUT HINT) rather than a partial crop\n"
        "- If an orange region already covers part of a circuit, detect only the remaining uncaptured portion (if any)\n"
    )


def _build_verification_prompt_v4(
    crop_boxes: List[Dict],
    current_polygon: Sequence[float],
    image_size: Tuple[int, int],
    has_pending: bool = False,
    history: Optional[List[Dict]] = None,
    grid_step_x: int = GRID_STEP_X,
    grid_step_y: int = GRID_STEP_Y,
    use_grid: bool = True,
    crop_mode: bool = False,
    n_existing: int = 0,
    n_pending: int = 0,
    clipped_edges_hint: Optional[List[str]] = None,
) -> str:
    """Directional verification prompt (v4) — per-edge direction analysis.

    crop_mode=True: sends individual crop images (orange + green + purple) instead of full overlay.
    n_existing: number of orange (already-verified) crop images preceding the purple crop.
    n_pending: number of green (pending) crop images following orange images.
    """
    xs = [current_polygon[i] for i in range(0, len(current_polygon), 2)]
    ys = [current_polygon[i] for i in range(1, len(current_polygon), 2)]
    cx1, cy1 = int(min(xs)), int(min(ys))
    cx2, cy2 = int(max(xs)), int(max(ys))
    w, h = image_size

    if crop_mode:
        # Describe the individual crop images sent instead of the full overlay
        total_ref = n_existing + n_pending
        last_idx = total_ref + 1
        images_desc = f"**📸 You are given {last_idx} image(s) in this request:**\n"
        if n_existing > 0:
            images_desc += (
                f"  Images 1–{n_existing}: Already captured circuit diagram regions (confirmed). "
                "Use these as reference for what a valid circuit diagram looks like on this page.\n"
            )
        if n_pending > 0:
            start = n_existing + 1
            end = n_existing + n_pending
            images_desc += (
                f"  Image{'s' if n_pending > 1 else ''} {start}{'–' + str(end) if n_pending > 1 else ''}: "
                "Other newly detected regions awaiting verification (pending). For context only.\n"
            )
        images_desc += (
            f"  Image {last_idx} (LAST): **The current region being verified** — "
            "this is the crop you must evaluate.\n"
        )
        grid_hint = (
            "\nUse the pixel coordinates provided below to determine precise values for each corrected edge.\n"
        )
        prompt = (
            "You are verifying if a detected region correctly captures a circuit diagram "
            "in a CAD/engineering drawing (electrical single-line diagrams).\n\n"
            + images_desc +
            "Focus your analysis on the **LAST image** (the region being verified). "
            "Compare it against the reference images (images 1 to " + str(total_ref) + ") "
            "to judge whether it is a valid circuit diagram of the same type.\n\n"
            "**🔌 CRITICAL: What are ELECTRICAL CIRCUIT DIAGRAMS (NOT tables)?**\n"
            "Electrical single-line diagrams (power distribution single-line diagrams) are IMAGE REGIONS that contain:\n"
            "✅ Visual SYMBOLS: transformers (⚡), circuit breakers (□), relays (○), motors (M), switches, contactors\n"
            "✅ CONNECTION LINES: vertical/horizontal/diagonal lines (solid or dashed) showing power flow\n"
            "✅ HIERARCHICAL FLOW: power source → transformer → breaker → load (top-to-bottom)\n"
            "\n❌ NOT image regions: Pure data tables, page header strips, standalone notes/legend boxes — no circuit symbols\n"
            "\n**Key distinction**: Circuit diagram = SYMBOLS + CONNECTION LINES ✅ / Header/notes/table = text only ❌\n"
            + grid_hint
            + "\nCurrent region polygon: " + json.dumps(list(current_polygon))
            + f"\nBounding box edges: x1={cx1} (left), y1={cy1} (top), x2={cx2} (right), y2={cy2} (bottom)"
            + f"\nImage size: width={w}, height={h}\n"
        )
    else:
        color_explanation = (
            "- 🟣 PURPLE box = The region currently being verified (FOCUS ON THIS ONE)\n"
        )
        if has_pending:
            color_explanation += (
                "- 🟢 LIME GREEN boxes = Other newly detected regions waiting for verification (ignore these for now)\n"
            )

        overlay_desc = (
            "  1. **OVERLAY image** (1st image): Full page with grid + colored bounding boxes — use this for edge/geometry analysis\n"
            if use_grid else
            "  1. **OVERLAY image** (1st image): Full page with colored bounding boxes — use this for edge/geometry analysis\n"
        )

        grid_hint = (
            f"\nThe image has a pixel-coordinate grid overlay (lines every {grid_step_x}px x-axis / {grid_step_y}px y-axis, axis labels, (col,row) cell labels). "
            "Use these grid labels to determine precise pixel values for each corrected edge.\n"
            if use_grid else
            "\nUse the pixel coordinates provided below to determine precise pixel values for each corrected edge.\n"
        )

        prompt = (
            "You are verifying if a PURPLE bounding box correctly captures an image region "
            "in a CAD/engineering drawing (electrical single-line diagrams).\n\n"
            "**📸 You are given TWO images in this request:**\n"
            + overlay_desc +
            "  2. **CROP image** (2nd image): The actual content currently inside the PURPLE box — use this to verify content type\n"
            "When judging whether the region contains a circuit diagram, rely primarily on the CROP image.\n\n"
            "**🔌 CRITICAL: What are ELECTRICAL CIRCUIT DIAGRAMS (NOT tables)?**\n"
            "Electrical single-line diagrams (power distribution single-line diagrams) are IMAGE REGIONS that contain:\n"
            "✅ Visual SYMBOLS: transformers (⚡), circuit breakers (□), relays (○), motors (M), switches, contactors\n"
            "✅ CONNECTION LINES: vertical/horizontal/diagonal lines (solid or dashed) showing power flow\n"
            "✅ HIERARCHICAL FLOW: power source → transformer → breaker → load (top-to-bottom)\n"
            "\n❌ NOT image regions: Pure data tables, page header strips, standalone notes/legend boxes — no circuit symbols\n"
            "\n**Key distinction**: Circuit diagram = SYMBOLS + CONNECTION LINES ✅ / Header/notes/table = text only ❌\n"
            "\n**CRITICAL: Understand the colored boxes!**\n"
            + color_explanation
            + grid_hint
            + "\nCurrent PURPLE region polygon: " + json.dumps(list(current_polygon))
            + f"\nBounding box edges: x1={cx1} (left), y1={cy1} (top), x2={cx2} (right), y2={cy2} (bottom)"
            + f"\nImage size: width={w}, height={h}\n"
        )

    if clipped_edges_hint:
        edges_str = ", ".join(clipped_edges_hint).upper()
        prompt += (
            f"\n⚠️ **PRE-IDENTIFIED CLIPPING**: A separate edge-clipping check has already confirmed "
            f"that circuit content is cut off at the **{edges_str}** edge(s) of this crop. "
            f"You MUST expand the boundary on those edge(s) to include the missing content. "
            f"Do NOT return 'none' for those directions.\n"
        )

    if history and len(history) > 0:
        prompt += "\n**📜 Previous Verification Attempts (DO NOT repeat same mistakes):**\n"
        for i, entry in enumerate(history, 1):
            prompt += f"\nAttempt {i}:\n"
            prompt += f"- Polygon tried: {json.dumps(entry['polygon'])}\n"
            prompt += f"- Issue found: {entry['issue']}\n"
        prompt += "\n⚠️ Learn from above — do NOT apply the same correction that already failed.\n"

    if crop_mode:
        prompt += (
            "\n\n**🎯 STEP-BY-STEP: Analyze the LAST image (current region) INDEPENDENTLY**\n"
            "Check each edge based on the visible content in the crop image itself.\n\n"

            f"━━ x1 = {cx1} (LEFT edge) ━━\n"
            "Q: Is circuit diagram content visibly cut off at the LEFT side of the crop image?\n"
            "  → YES → direction='left', corrected = [pixel where left boundary should be]\n"
            "Q: Does the crop show excess empty whitespace on the left side?\n"
            "  → YES → direction='right', corrected = [pixel where left boundary should be]\n"
            "  → NEITHER → direction='none', corrected = current x1\n\n"

            f"━━ y1 = {cy1} (TOP edge) ━━\n"
            "⚠️ CRITICAL: Must include area designation boxes (dashed zone labels at top of diagram)\n"
            "Q: Is content (area labels, diagram symbols) cut off at the TOP of the crop image?\n"
            "  → YES → direction='up', corrected = [pixel above the cutoff]\n"
            "Q: Does the crop show excess empty whitespace at the top?\n"
            "  → YES → direction='down', corrected = [pixel where top boundary should be]\n"
            "  → NEITHER → direction='none', corrected = current y1\n\n"

            f"━━ x2 = {cx2} (RIGHT edge) ━━\n"
            "Q: Is circuit diagram content visibly cut off at the RIGHT side of the crop image?\n"
            "  → YES → direction='right', corrected = [pixel where right boundary should be]\n"
            "Q: Does the crop show excess empty whitespace on the right side?\n"
            "  → YES → direction='left', corrected = [pixel where right boundary should be]\n"
            "  → NEITHER → direction='none', corrected = current x2\n\n"

            f"━━ y2 = {cy2} (BOTTOM edge) ━━\n"
            "⚠️ CRITICAL: Must include ALL bottom-connected components (motors M, terminals, switches connected by dashed lines)\n"
            "Q: Are motors, terminals, or components cut off at the BOTTOM of the crop image?\n"
            "  → YES → direction='down', corrected = [pixel below the lowest connected component]\n"
            "Q: Does the crop show excess empty whitespace at the bottom?\n"
            "  → YES → direction='up', corrected = [pixel where bottom boundary should be]\n"
            "  → NEITHER → direction='none', corrected = current y2\n\n"

            "**Constraints for corrected values:**\n"
            f"  x1 must satisfy: 0 ≤ x1 < x2 ≤ {w}\n"
            f"  y1 must satisfy: 0 ≤ y1 < y2 ≤ {h}\n"
            "  If direction='none', corrected = current value (no change).\n\n"

            "**Return JSON matching this schema:**\n"
            + json.dumps(DIRECTIONAL_VERIFICATION_SCHEMA)
            + "\n\n**Decision rules:**\n"
            "1) is_correct=true ONLY when ALL 4 edges are direction='none' AND the last image IS a valid circuit diagram\n"
            "2) If is_correct=false:\n"
            "   a) For bounding box issues: provide corrected edge values + corrected_polygon=[x1_c,y1_c, x2_c,y1_c, x2_c,y2_c, x1_c,y2_c]\n"
            "   b) Non-geometric rejection (all edges 'none', issue='not a circuit diagram') ONLY when the last image contains ZERO circuit symbols — no breaker/switch/transformer symbols, no connection lines, no hierarchical power flow.\n"
            "3) The current region must fully contain the COMPLETE circuit diagram (no cutting off content at any side)\n"
        )
    else:
        prompt += (
            "\n\n**🎯 STEP-BY-STEP: Analyze each edge of the PURPLE box INDEPENDENTLY**\n"
            "For each edge, decide the direction it needs to move (or 'none' if already correct).\n"
            "Use the grid pixel labels to read the corrected value precisely.\n\n"

            f"━━ x1 = {cx1} (LEFT edge) ━━\n"
            "Q: Is the left boundary of the PURPLE box cutting into diagram content on the left?\n"
            "  → YES (content cut off on left side) → direction='left', corrected = [pixel where left boundary should be]\n"
            "Q: Does the left edge have too much empty whitespace?\n"
            "  → YES (too much whitespace) → direction='right', corrected = [pixel where left boundary should be]\n"
            "  → NEITHER (already correct) → direction='none', corrected = current x1\n\n"

            f"━━ y1 = {cy1} (TOP edge) ━━\n"
            "⚠️ CRITICAL: Must include area designation boxes (dashed zone labels at top of diagram)\n"
            "Q: Is the top boundary of the PURPLE box cutting off area designation boxes or diagram content at the top?\n"
            "  → YES (area designation box or content cut off at top) → direction='up', corrected = [pixel above the cutoff]\n"
            "Q: Does the top edge have too much empty whitespace?\n"
            "  → YES → direction='down', corrected = [pixel where top boundary should be]\n"
            "  → NEITHER → direction='none', corrected = current y1\n\n"

            f"━━ x2 = {cx2} (RIGHT edge) ━━\n"
            "Q: Is the right boundary of the PURPLE box cutting off diagram content on the right?\n"
            "  → YES → direction='right', corrected = [pixel where right boundary should be]\n"
            "Q: Does the right edge have too much empty whitespace?\n"
            "  → YES → direction='left', corrected = [pixel where right boundary should be]\n"
            "  → NEITHER → direction='none', corrected = current x2\n\n"

            f"━━ y2 = {cy2} (BOTTOM edge) ━━\n"
            "⚠️ CRITICAL: Must include ALL bottom-connected components (motors M, terminals, switches connected by dashed lines)\n"
            "Q: Is the bottom boundary of the PURPLE box cutting off motors, terminals, or components at the bottom?\n"
            "  → YES → direction='down', corrected = [pixel below the lowest connected component]\n"
            "Q: Does the bottom edge have too much empty whitespace below the diagram?\n"
            "  → YES → direction='up', corrected = [pixel where bottom boundary should be]\n"
            "  → NEITHER → direction='none', corrected = current y2\n\n"

            "**Constraints for corrected values:**\n"
            f"  x1 must satisfy: 0 ≤ x1 < x2 ≤ {w}\n"
            f"  y1 must satisfy: 0 ≤ y1 < y2 ≤ {h}\n"
            "  If direction='none', corrected = current value (no change).\n\n"

            "**Return JSON matching this schema:**\n"
            + json.dumps(DIRECTIONAL_VERIFICATION_SCHEMA)
            + "\n\n**Decision rules:**\n"
            "1) is_correct=true ONLY when ALL 4 edges are direction='none' AND the region IS a valid circuit diagram\n"
            "2) If is_correct=false:\n"
            "   a) For bounding box issues: provide corrected edge values + corrected_polygon=[x1_c,y1_c, x2_c,y1_c, x2_c,y2_c, x1_c,y2_c]\n"
            "   b) Non-geometric rejection (all edges 'none', issue='not a circuit diagram') ONLY when the CROP image (2nd image) contains ZERO circuit symbols — no breaker/switch/transformer symbols, no connection lines, no hierarchical power flow. "
            "If the CROP clearly shows circuit content but the PURPLE box merely extends into a text/notes area on one side, that is a GEOMETRIC issue: move that edge inward to exclude the text area. Do NOT reject as non-circuit in this case.\n"
            "3) PURPLE must fully contain the COMPLETE circuit diagram (no cutting off content at any side)\n"
            "4) IGNORE BLUE boxes\n"
        )

    return prompt


# ── API call wrappers ──────────────────────────────────────────────────────────

def responses_call(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    page_img: np.ndarray,
    reasoning_effort: Optional[str] = None,
    grid_step_x: int = GRID_STEP_X,
    grid_step_y: int = GRID_STEP_Y,
    use_grid: bool = True,
) -> Tuple[str, float]:
    """Detection call — optionally applies numbered grid overlay to the page image.

    Returns (output_text, elapsed_seconds).
    """
    import time

    if use_grid:
        send_img = draw_grid_with_cell_numbers(page_img, step_x=grid_step_x, step_y=grid_step_y)
        h, w = send_img.shape[:2]
        num_cols = (w + grid_step_x - 1) // grid_step_x
        num_rows = (h + grid_step_y - 1) // grid_step_y
        grid_note = (
            f"[Grid overlay: pixel-coordinate grid with lines every {grid_step_x}px on the "
            f"x-axis and every {grid_step_y}px on the y-axis. "
            f"The grid forms {num_cols} columns and {num_rows} rows; "
            f"each cell is labeled (col, row) at its center. "
            f"Use these labels to determine precise pixel coordinates for polygon corners.]\n\n"
        )
        full_prompt = grid_note + prompt
    else:
        send_img = page_img
        full_prompt = prompt

    t0 = time.time()
    response = client.responses.create(
        model=deployment,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": full_prompt},
                    {"type": "input_image", "image_url": image_to_data_url(send_img), "detail": "original"},
                ],
            }
        ],
        **( {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {"temperature": 0} ),
    )
    return response.output_text, time.time() - t0


def responses_call_with_image(
    client: AzureOpenAI,
    deployment: str,
    prompt: str,
    image: Union[np.ndarray, List[np.ndarray]],
    reasoning_effort: Optional[str] = None,
) -> Tuple[str, float]:
    """Verification call — sends one or more images to LLM.

    If *image* is a list, all images are sent in order (e.g. [overlay, crop]).
    Returns (output_text, elapsed_seconds).
    """
    import time

    images: List[np.ndarray] = [image] if isinstance(image, np.ndarray) else image
    content: List[Dict] = [{"type": "input_text", "text": prompt}]
    for img in images:
        content.append({"type": "input_image", "image_url": image_to_data_url(img), "detail": "original"})
    t0 = time.time()
    response = client.responses.create(
        model=deployment,
        input=[{"role": "user", "content": content}],
        **( {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {"temperature": 0} ),
    )
    return response.output_text, time.time() - t0


def recheck_is_circuit_diagram(
    client: AzureOpenAI,
    deployment: str,
    crop_image: np.ndarray,
    reasoning_effort: Optional[str] = None,
) -> Tuple[bool, str]:
    """Second-opinion check: does this crop actually contain a circuit diagram?

    Called when verify_crop_with_llm returns all_none + is_correct=false, to
    distinguish "genuinely not a circuit" from "no geometric fix needed but IS a circuit".

    Returns (is_circuit: bool, reason: str).
    """
    prompt = (
        "You are given a crop image from a CAD/engineering drawing.\n\n"
        "**Task**: Determine if this image contains an electrical circuit diagram "
        "(power distribution single-line diagram / electrical single-line diagram).\n\n"
        "✅ IS a circuit diagram if it contains ANY of:\n"
        "  - Electrical symbols: transformers (⚡), circuit breakers (□), relays (○), motors (M), switches, contactors\n"
        "  - Connection lines showing power flow (vertical/horizontal/diagonal)\n"
        "  - Hierarchical power flow structure (source → transformer → breaker → load)\n\n"
        "❌ NOT a circuit diagram if it contains ONLY:\n"
        "  - Pure text, tables, legends, notes, page headers — no circuit symbols at all\n\n"
        "Return JSON: {\"is_circuit\": <true|false>, \"reason\": \"<brief explanation>\"}"
    )
    content: List[Dict] = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": image_to_data_url(crop_image), "detail": "original"},
    ]
    response = client.responses.create(
        model=deployment,
        input=[{"role": "user", "content": content}],
        **( {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {"temperature": 0} ),
    )
    result = safe_parse_json(response.output_text)
    is_circuit = bool(result.get("is_circuit", False))
    reason = result.get("reason", "")
    return is_circuit, reason


def check_di_crop(
    client: AzureOpenAI,
    deployment: str,
    crop_image: np.ndarray,
    page_overlay_img: np.ndarray,
    current_box: Optional[Tuple[int, int, int, int]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    reasoning_effort: Optional[str] = None,
) -> Tuple[bool, List[str], Optional[Dict[str, int]], str]:
    """Crop check: verify that a DI-detected crop boundary is correctly placed.

    Checks for two issues:
      1. Clipping — circuit content cut off at a crop edge (panel box sliced open, label truncated)
      2. Excess area — large non-circuit margin (title block, blank NOTE column, etc.) at an edge

    Sends BOTH the full-page overlay image (orange box in context) AND the individual
    crop to the LLM. Returns a suggested bounding box when adjustment is needed.

    Args:
        crop_image: The individual DI-detected crop image.
        page_overlay_img: The full page image with orange boxes drawn (DI overlay).
        current_box: Current (x1, y1, x2, y2) pixel coords of the crop.
        image_size: (width, height) of the full page image.

    Returns:
        (is_clipped, clipped_edges, excess_edges, suggested_box, note)
        - is_clipped: True if any edge cuts through circuit content
        - clipped_edges: list of edge names that are clipped, e.g. ["left", "top"]
        - excess_edges: list of edge names with large unnecessary margins
        - suggested_box: dict with x1/y1/x2/y2 when adjustment needed, else None
        - note: brief explanation
    """
    box_hint = ""
    if current_box:
        x1, y1, x2, y2 = current_box
        box_hint = f"\nCurrent crop bounding box (pixel coords): x1={x1}, y1={y1}, x2={x2}, y2={y2}"
    if image_size:
        box_hint += f"\nFull page image size: width={image_size[0]}, height={image_size[1]}"

    prompt = (
        "You are given two images:\n"
        "  Image 1: The full engineering drawing page with an ORANGE box marking the current crop boundary.\n"
        "  Image 2: The actual cropped image — exactly what is inside the orange box.\n\n"
        "**How to use both images**:\n"
        "  - Use Image 2 (the crop) as the PRIMARY source for clipping judgment. "
        "Look at each edge of Image 2 directly: is the content at that edge complete and intact, or is something visibly cut off?\n"
        "  - Use Image 1 (the full page) only for context — to understand what surrounds the crop and whether adjacent regions exist outside the orange box.\n"
        "  - Do NOT judge clipping from how the orange box looks relative to content in Image 1. "
        "Judge only from what you actually see cut off in Image 2.\n\n"
        "**Task**: Evaluate two issues — clipping AND excess area.\n"
        + box_hint + "\n\n"
        "━━ ISSUE 1: CLIPPING (content cut off) ━━\n\n"
        "✅ Mark an edge as CLIPPED only if, looking at Image 2, you see clear physical evidence of cutting at that edge:\n"
        "  - A panel/module/switchgear BOX whose rectangular border is missing one or more sides (the box is sliced open)\n"
        "  - An electrical symbol (breaker, transformer, motor) that is clearly cut in half — part of the symbol is missing\n"
        "  - A panel name/label that is partially cut — the text of a panel box title is truncated at the edge\n\n"
        "✅ If all content in Image 2 appears complete and cleanly bounded on all four sides — even if the orange box in Image 1 sits close to content — return is_clipped=false.\n\n"
        "❌ Do NOT mark as clipped in these common false-positive cases:\n"
        "  - All panel/module boxes in Image 2 are complete with closed borders — NOT clipped even if wires exit at the edges\n"
        "  - Wires or electrical lines that cross the edge — wires connect to adjacent areas, they are not cut content\n"
        "  - Bus bars or distribution lines running along the edge — shared infrastructure, not cut content\n"
        "  - The edge of Image 2 ends at a clean margin, page border, dashed frame, or title block\n"
        "  - The circuit naturally ends at or before the crop boundary\n\n"
        "⚠️ CONSERVATIVE RULE: If unsure, return is_clipped=false. Only flag when there is UNAMBIGUOUS evidence of slicing in Image 2.\n\n"
        "━━ BOUNDARY REFERENCE RULE ━━\n\n"
        "Use PANEL BOX NAMES/LABELS as the primary reference for where crop boundaries should be:\n"
        "  - The crop should start just before the first panel name and end just after the last panel name on each axis\n"
        "  - If a panel name/label is cut or missing at an edge → that edge is clipped\n"
        "  - If a panel name/label is fully visible with clear margin → that edge is correct or has excess\n"
        "  - Electrical wires crossing the boundary are NOT a signal of clipping — ignore wire crossings entirely\n\n"
        "━━ ISSUE 2: EXCESS AREA (unnecessary margin) ━━\n\n"
        "⚠️ CONSERVATIVE: Only flag excess when you are CERTAIN the area adds no circuit value. When in doubt, do NOT flag.\n\n"
        "✅ Mark an edge in `excess_edges` ONLY if ALL of the following are true:\n"
        "  1. The area near that edge is clearly a NON-CIRCUIT structural zone — specifically: "
        "a blank title block, a NOTE column with no note text, a revision table, or a company logo panel\n"
        "     ⚠️ Do NOT treat sheet borders or margins around circuit content as excess — generator and source symbols often sit near the sheet edge\n"
        "  2. There are NO electrical symbols, panel boxes, breakers, cable labels, or wiring within that zone\n\n"
        "❌ Do NOT mark as excess if:\n"
        "  - The area contains ANY circuit-related content (cable labels, wiring, bus bars, symbols — even sparse)\n"
        "  - The area is simply a wide margin or continuation space between circuit sections\n"
        "  - You are not certain it is a structural non-circuit zone (title block, blank NOTE column, etc.)\n\n"
        "━━ SUGGESTED BOX ━━\n\n"
        "Provide `suggested_box` when is_clipped=true OR excess_edges is non-empty.\n"
        "  - Clipped edges: expand outward to include the full panel name/label\n"
        "  - Excess edges: shrink inward to just past the last panel name/label\n"
        "  - All other edges: keep at exactly their current values\n\n"
        "━━ PANEL BOX BOUNDARY & OVERLAP RULE ━━\n\n"
        "Use PANEL BOX BOUNDARIES as cut points to minimize overlap with adjacent crops:\n"
        "  - Clipped edge (content cut off) → expand to include the full panel box at that edge. "
        "Do NOT report an edge as clipped if you intend to shrink it — use excess_edges instead\n"
        "  - If the crop boundary cuts through content but a cleaner cut exists INWARD (shrinking to the nearest complete panel box boundary), "
        "report that edge in excess_edges (not clipped_edges) so the boundary is shrunk cleanly\n"
        "  - NEVER report the same edge as both clipped and excess\n"
        "  - Do NOT expand so far that the crop swallows content that logically belongs to an adjacent, separately bounded crop\n\n"
        "Return JSON matching this schema:\n"
        + json.dumps(DI_CROP_CHECK_SCHEMA)
    )
    content: List[Dict] = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": image_to_data_url(page_overlay_img), "detail": "original"},
        {"type": "input_image", "image_url": image_to_data_url(crop_image), "detail": "original"},
    ]
    response = client.responses.create(
        model=deployment,
        input=[{"role": "user", "content": content}],
        **( {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {"temperature": 0} ),
    )
    result = safe_parse_json(response.output_text)
    is_clipped = bool(result.get("is_clipped", False))
    clipped_edges = result.get("clipped_edges", [])
    excess_edges = result.get("excess_edges", [])
    has_adjustment = is_clipped or bool(excess_edges)
    suggested_box = result.get("suggested_box") if has_adjustment else None
    note = result.get("note", "")
    return is_clipped, clipped_edges, excess_edges, suggested_box, note
