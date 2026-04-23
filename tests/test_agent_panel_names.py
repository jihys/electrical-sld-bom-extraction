"""
Plan-and-Execute Agent for Panel Name Extraction
-------------------------------------------------
Standalone test that uses a tool-calling LLM agent to extract panel names
from an SLD page image. The agent autonomously decides how to tile/crop,
extract, overlay-verify, and iterate until it is confident the list is complete.

Usage:
    python -m tests.test_agent_panel_names --page outputs/pages/page6.png
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Settings
from src.agents.llm_caller import create_llm_client

# ── Output dir ────────────────────────────────────────────────────────────────
WORK_DIR = ROOT / "outputs" / "agent_panel_test"
WORK_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool definitions (JSON schema for LLM)
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tile_image",
            "description": (
                "Split an image into a grid of overlapping tiles for closer inspection. "
                "Returns a list of tile file paths and their pixel coordinates [x1,y1,x2,y2] "
                "on the original image. Use this when the full image is too large to read "
                "small panel name labels accurately."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file to tile.",
                    },
                    "tile_width": {
                        "type": "integer",
                        "description": "Width of each tile in pixels. Default 1200.",
                    },
                    "tile_height": {
                        "type": "integer",
                        "description": (
                            "Height of each tile in pixels. "
                            "Use 0 or omit to use the full image height (no vertical split)."
                        ),
                    },
                    "overlap": {
                        "type": "integer",
                        "description": "Overlap between adjacent tiles in pixels. Default 300.",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crop_region",
            "description": (
                "Crop a specific rectangular region from an image. "
                "Returns the path to the cropped image file. "
                "Use this to zoom into a particular area for closer inspection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the source image.",
                    },
                    "x1": {"type": "integer", "description": "Left edge (pixels)."},
                    "y1": {"type": "integer", "description": "Top edge (pixels)."},
                    "x2": {"type": "integer", "description": "Right edge (pixels)."},
                    "y2": {"type": "integer", "description": "Bottom edge (pixels)."},
                    "label": {
                        "type": "string",
                        "description": "Optional label for the crop file name.",
                    },
                },
                "required": ["image_path", "x1", "y1", "x2", "y2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_names_from_image",
            "description": (
                "Use vision AI to extract panel bay names AND their bounding box "
                "coordinates from an image. Returns a list of {name, bbox, type} objects. "
                "IMPORTANT: the same panel name may appear multiple times on the page "
                "(as a bay label AND as a cross-reference). Each occurrence is returned "
                "separately with its own bbox and type ('label' or 'reference'). "
                "Coordinates are in pixels relative to the input image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image to analyze.",
                    },
                    "context_hint": {
                        "type": "string",
                        "description": (
                            "Optional hint about what to look for, e.g. "
                            "'focus on right side' or 'look for small auxiliary panels'."
                        ),
                    },
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "overlay_names_on_image",
            "description": (
                "Draw the current panel name list as labeled rectangles on the original "
                "full-page image. Returns the path to the annotated image. "
                "Use this to visually verify which names have been found and where they are. "
                "Each name will be placed at its approximate location if coordinates are known, "
                "otherwise listed in a legend. After calling this, inspect the returned image "
                "to check for missing or incorrect names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the original full-page image.",
                    },
                    "panel_names": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "bbox": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "[x1,y1,x2,y2] pixel coords. null if unknown.",
                                },
                            },
                            "required": ["name"],
                        },
                        "description": "List of panel names with optional bbox coordinates.",
                    },
                },
                "required": ["image_path", "panel_names"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_final_result",
            "description": (
                "Submit the final, verified list of panel bay names with their "
                "bounding boxes for this page. Each entry must have a bbox pointing "
                "to the actual PANEL LABEL BOX (not a cross-reference). "
                "Call this ONLY when you are confident the list is complete and "
                "all bboxes point to the correct panel label locations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "panel_names": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "bbox": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "[x1,y1,x2,y2] pixel coords of the panel label box on the FULL page image.",
                                },
                            },
                            "required": ["name", "bbox"],
                        },
                        "description": "Final list of panel names with verified bbox locations on the full page.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of your confidence.",
                    },
                },
                "required": ["panel_names"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Tool implementations
# ═══════════════════════════════════════════════════════════════════════════════

_tile_counter = 0
_crop_counter = 0
_overlay_counter = 0


def _img_to_data_url(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode()}"


def tool_tile_image(
    image_path: str,
    tile_width: int = 1200,
    tile_height: int = 0,
    overlap: int = 300,
) -> str:
    global _tile_counter
    img = cv2.imread(image_path)
    if img is None:
        return json.dumps({"error": f"Cannot read image: {image_path}"})
    h, w = img.shape[:2]
    tw = min(tile_width, w)
    th = tile_height if tile_height > 0 else h
    th = min(th, h)
    ovl_x = min(overlap, tw // 2)
    ovl_y = min(overlap, th // 2) if th < h else 0

    tiles = []
    y = 0
    while y < h:
        y2 = min(y + th, h)
        x = 0
        while x < w:
            x2 = min(x + tw, w)
            tile_img = img[y:y2, x:x2]
            _tile_counter += 1
            fname = f"tile_{_tile_counter:03d}.png"
            fpath = str(WORK_DIR / fname)
            cv2.imwrite(fpath, tile_img)
            tiles.append({
                "tile_path": fpath,
                "coords": [x, y, x2, y2],
                "size": f"{x2-x}x{y2-y}",
            })
            if x2 >= w:
                break
            x += tw - ovl_x
        if y2 >= h:
            break
        y += th - ovl_y

    return json.dumps({
        "total_tiles": len(tiles),
        "original_size": f"{w}x{h}",
        "tiles": tiles,
    })


def tool_crop_region(
    image_path: str,
    x1: int, y1: int, x2: int, y2: int,
    label: str = "",
) -> str:
    global _crop_counter
    img = cv2.imread(image_path)
    if img is None:
        return json.dumps({"error": f"Cannot read image: {image_path}"})
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return json.dumps({"error": "Invalid crop coordinates"})
    crop = img[y1:y2, x1:x2]
    _crop_counter += 1
    tag = f"_{label}" if label else ""
    fname = f"crop_{_crop_counter:03d}{tag}.png"
    fpath = str(WORK_DIR / fname)
    cv2.imwrite(fpath, crop)
    return json.dumps({
        "crop_path": fpath,
        "coords": [x1, y1, x2, y2],
        "size": f"{x2-x1}x{y2-y1}",
    })


def tool_extract_names(
    llm_client,
    deployment: str,
    image_path: str,
    context_hint: str = "",
) -> str:
    img = cv2.imread(image_path)
    if img is None:
        return json.dumps({"error": f"Cannot read image: {image_path}"})

    h_img, w_img = img.shape[:2]
    hint_text = f"\nAdditional hint: {context_hint}\n" if context_hint else ""

    prompt = (
        "You are analyzing a section of an electrical Single Line Diagram (SLD).\n"
        f"Image dimensions: {w_img} x {h_img} pixels.\n\n"
        "## Task\n"
        "Find ALL panel bay name labels visible in this image and return each with\n"
        "its bounding box [x1, y1, x2, y2] in pixels and whether it is a panel\n"
        "bay LABEL or a cross-REFERENCE.\n\n"
        "## What is a panel bay name?\n"
        "A short alphanumeric code in a bordered box (rectangle or hexagon) that\n"
        "labels a ZONE/SECTION of the switchboard. The box sits ON or near the\n"
        "dashed/solid boundary line of that zone.\n\n"
        "## CRITICAL — label vs reference\n"
        "The SAME name (e.g. 'E-H-01B') may appear multiple times on one page:\n"
        "  • 'label'  — the bordered box at a panel zone boundary (this IS the panel)\n"
        "  • 'reference' — mentioned inside annotation text like\n"
        "    'FROM C.S U-4.2 SUBSTATION (E-H-01B)' or 'E-TR-02B / 3.45/0.46kV'\n"
        "Report EVERY occurrence with its bbox and type.\n\n"
        "## Rules\n"
        "- 2-line labels: combine into single string.\n"
        "- Include UPPER/LOWER suffixes if present.\n"
        "- Include ALL panels, even small auxiliary ones at page edges.\n"
        "- bbox: estimate pixel coords [x1, y1, x2, y2] for the label text box.\n\n"
        "## Exclude entirely\n"
        "- Equipment codes (breaker/CT/relay model numbers)\n"
        "- Title block text, revision notes, cable specs\n\n"
        f"{hint_text}"
        "Return JSON:\n"
        '{"panels": [\n'
        '  {"name": "LV-01", "bbox": [100,700,200,740], "type": "label"},\n'
        '  {"name": "E-H-01B", "bbox": [800,500,1100,530], "type": "reference"}\n'
        '], "reasoning": "..."}'
    )

    data_url = _img_to_data_url(img)
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": data_url, "detail": "high"},
    ]

    t0 = time.time()
    response = llm_client.responses.create(
        model=deployment,
        input=[{"role": "user", "content": content}],
        temperature=0,
    )
    elapsed = round(time.time() - t0, 1)

    raw = response.output_text
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        result = json.loads(text)
    except Exception:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        result = json.loads(m.group()) if m else {"panels": [], "reasoning": raw}

    panels = result.get("panels", [])
    # Backwards compat: if old format
    if not panels and "panel_names" in result:
        panels = [{"name": n, "bbox": None, "type": "label"} for n in result["panel_names"]]
    reasoning = result.get("reasoning", "")
    return json.dumps({
        "panels": panels,
        "reasoning": reasoning,
        "elapsed_s": elapsed,
        "image_path": image_path,
        "image_size": [w_img, h_img],
    })


def tool_overlay_names(
    image_path: str,
    panel_names: List[Dict[str, Any]],
) -> str:
    global _overlay_counter
    img = cv2.imread(image_path)
    if img is None:
        return json.dumps({"error": f"Cannot read image: {image_path}"})
    h, w = img.shape[:2]
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(1.5, h / 1500))
    thickness = max(1, int(font_scale * 2))
    color_box = (0, 200, 0)       # green boxes
    color_text = (0, 0, 255)      # red text
    color_unknown = (200, 200, 0) # cyan for unknown-bbox names

    legend_y = 30
    for entry in panel_names:
        name = entry.get("name", "?")
        bbox = entry.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_box, 2)
            cv2.putText(overlay, name, (x1, max(y1 - 8, 15)),
                        font, font_scale, color_text, thickness, cv2.LINE_AA)
        else:
            # No bbox — put in legend area
            cv2.putText(overlay, f"[?] {name}", (10, legend_y),
                        font, font_scale * 0.8, color_unknown, thickness, cv2.LINE_AA)
            legend_y += int(30 * font_scale)

    _overlay_counter += 1
    fname = f"overlay_{_overlay_counter:03d}.png"
    fpath = str(WORK_DIR / fname)
    cv2.imwrite(fpath, overlay)
    return json.dumps({
        "overlay_path": fpath,
        "n_names": len(panel_names),
        "n_with_bbox": sum(1 for e in panel_names if e.get("bbox")),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Agent loop
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert electrical engineering AI agent. Your job is to extract panel \
bay names from an SLD drawing AND mark exactly where each panel label box is located.

## Your goal
Find ALL panel bay names on the page and return each with its CORRECT bounding \
box — the bbox of the actual panel label box (bordered rectangle/hexagon at \
the panel zone boundary), NOT a cross-reference mention elsewhere on the page.

## Why location matters
The same name (e.g. "E-H-01B") can appear in MULTIPLE places on one page:
  1. As the actual PANEL LABEL — bordered box on/near the panel zone boundary
  2. As a CROSS-REFERENCE — in annotation text like "FROM C.S U-4.2 SUBSTATION (E-H-01B)"

The extract_names_from_image tool returns EVERY occurrence with a "type" field \
("label" or "reference"). You must pick the correct one.

## Your approach — Plan and Execute
1. **Assess** — Look at the full page. Note dimensions and layout.
2. **Extract** — Tile the image and call extract_names_from_image on each tile. \
   The tool returns panels with bbox coords (relative to the tile) and type.
3. **Merge** — Combine tile results. Convert tile-local bboxes to full-page \
   coordinates: add tile x_offset to x1,x2 and y_offset to y1,y2. \
   Keep only "label" type entries (discard "reference" entries).
4. **Overlay & Verify** — Call overlay_names_on_image with the merged list \
   (names + bboxes) on the full page. Inspect the overlay image:
   - Are the green boxes drawn at actual panel label locations?
   - Any box placed at a cross-reference text instead of a panel boundary?
   - Any missing panels (especially small ones at page edges)?
5. **Fix** — If any bbox is wrong, crop that area, re-extract, and correct.
6. **Submit** — Call submit_final_result with names + verified full-page bboxes.

## Rules
- Every submitted name MUST have a bbox pointing to its panel LABEL BOX.
- Scan the FULL page width — small panels at the right edge are often missed.
- ALWAYS overlay and verify before submitting.
- Tile-to-page coord conversion: bbox_page = [x1+tile_x, y1+tile_y, x2+tile_x, y2+tile_y].
- Maximum 10 iterations. Be efficient.
"""


def _dispatch_tool(
    tool_name: str,
    args: Dict[str, Any],
    llm_client,
    deployment: str,
) -> str:
    """Execute a tool and return its JSON result string."""
    if tool_name == "tile_image":
        return tool_tile_image(
            image_path=args["image_path"],
            tile_width=args.get("tile_width", 1200),
            tile_height=args.get("tile_height", 0),
            overlap=args.get("overlap", 300),
        )
    elif tool_name == "crop_region":
        return tool_crop_region(
            image_path=args["image_path"],
            x1=args["x1"], y1=args["y1"],
            x2=args["x2"], y2=args["y2"],
            label=args.get("label", ""),
        )
    elif tool_name == "extract_names_from_image":
        return tool_extract_names(
            llm_client=llm_client,
            deployment=deployment,
            image_path=args["image_path"],
            context_hint=args.get("context_hint", ""),
        )
    elif tool_name == "overlay_names_on_image":
        return tool_overlay_names(
            image_path=args["image_path"],
            panel_names=args["panel_names"],
        )
    elif tool_name == "submit_final_result":
        # This is a terminal tool — return the result directly
        return json.dumps({
            "status": "submitted",
            "panel_names": args["panel_names"],
            "reasoning": args.get("reasoning", ""),
        })
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def run_agent(
    page_image_path: str,
    llm_client,
    deployment: str,
    max_iterations: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the plan-and-execute agent loop.

    Returns:
        {
            "panel_names": List[str],
            "iterations": int,
            "total_tool_calls": int,
            "elapsed_s": float,
            "conversation_log": List[dict],  # for debugging
        }
    """
    t0 = time.time()
    img = cv2.imread(page_image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {page_image_path}")
    h, w = img.shape[:2]

    # Initial user message with the full page image
    data_url = _img_to_data_url(img)
    initial_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"Here is a full-page SLD drawing ({w}x{h} pixels). "
                    f"Extract ALL panel bay names from this page. "
                    f"The image path is: {page_image_path}\n\n"
                    f"Plan your approach, use the tools, verify with overlay, and submit the final result."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "high"},
            },
        ],
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        initial_message,
    ]

    total_tool_calls = 0
    final_result = None

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Iteration {iteration}/{max_iterations}")
            print(f"{'='*60}")

        # Call LLM with tools
        response = llm_client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        # Add assistant response to history
        messages.append(assistant_msg)

        # If no tool calls, the agent wants to respond with text
        if not assistant_msg.tool_calls:
            if verbose:
                print(f"  Agent text response: {(assistant_msg.content or '')[:200]}")
            # Check if the agent is done or wants to continue
            if assistant_msg.content and "submit" in assistant_msg.content.lower():
                if verbose:
                    print("  Agent seems done but didn't call submit_final_result.")
            break

        # Process tool calls
        for tc in assistant_msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            total_tool_calls += 1
            if verbose:
                args_summary = {k: (v if not isinstance(v, list) or len(v) < 3
                                    else f"[{len(v)} items]")
                                for k, v in args.items()}
                print(f"  Tool call #{total_tool_calls}: {tool_name}({json.dumps(args_summary, ensure_ascii=False)[:200]})")

            # Check for terminal tool
            if tool_name == "submit_final_result":
                final_result = {
                    "panel_names": args.get("panel_names", []),
                    "reasoning": args.get("reasoning", ""),
                }
                tool_result = json.dumps({"status": "accepted", "message": "Result submitted successfully."})
                if verbose:
                    print(f"  ✅ Final result: {final_result['panel_names']}")
            else:
                # Execute the tool
                tool_result = _dispatch_tool(tool_name, args, llm_client, deployment)
                if verbose:
                    # Summarize result
                    try:
                        parsed = json.loads(tool_result)
                        if "error" in parsed:
                            print(f"    ❌ Error: {parsed['error']}")
                        elif "panels" in parsed:
                            panels = parsed["panels"]
                            labels = [p for p in panels if p.get("type") == "label"]
                            refs = [p for p in panels if p.get("type") == "reference"]
                            label_names = [p["name"] for p in labels]
                            ref_names = [p["name"] for p in refs]
                            print(f"    → {len(labels)} labels: {label_names}")
                            if refs:
                                print(f"    → {len(refs)} references (excluded): {ref_names}")
                        elif "total_tiles" in parsed:
                            print(f"    → Created {parsed['total_tiles']} tiles")
                        elif "overlay_path" in parsed:
                            print(f"    → Overlay: {parsed['overlay_path']} ({parsed['n_names']} names, {parsed['n_with_bbox']} with bbox)")
                        elif "crop_path" in parsed:
                            print(f"    → Crop: {parsed['crop_path']} ({parsed['size']})")
                    except Exception:
                        print(f"    → {tool_result[:150]}")

            # Add tool result to messages — handle image results specially
            tool_msg_content = tool_result

            # If the tool produced an image (overlay, crop, tile), include it for the agent
            try:
                parsed = json.loads(tool_result)
                img_path = (parsed.get("overlay_path") or parsed.get("crop_path"))
                if img_path and Path(img_path).exists():
                    result_img = cv2.imread(img_path)
                    if result_img is not None:
                        img_url = _img_to_data_url(result_img)
                        tool_msg_content = [
                            {"type": "text", "text": tool_result},
                            {"type": "image_url", "image_url": {"url": img_url, "detail": "high"}},
                        ]
            except Exception:
                pass

            messages.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "name": tool_name,
                "content": tool_msg_content if isinstance(tool_msg_content, str) else json.dumps({"result": tool_result, "note": "Image also attached for visual inspection."}),
            })

        if final_result is not None:
            break

    elapsed = round(time.time() - t0, 1)

    if final_result is None:
        # Agent didn't submit — extract from last extraction if possible
        if verbose:
            print("\n⚠️  Agent did not call submit_final_result. Extracting from conversation...")
        final_result = {"panel_names": [], "reasoning": "Agent did not submit."}

    # Normalize result: panel_names can be list of strings or list of {name, bbox}
    raw_names = final_result["panel_names"]
    if raw_names and isinstance(raw_names[0], dict):
        name_list = [p["name"] for p in raw_names]
        panels_with_bbox = raw_names
    else:
        name_list = raw_names
        panels_with_bbox = [{"name": n, "bbox": None} for n in raw_names]

    result = {
        "panel_names": name_list,
        "panels": panels_with_bbox,
        "reasoning": final_result.get("reasoning", ""),
        "iterations": iteration,
        "total_tool_calls": total_tool_calls,
        "elapsed_s": elapsed,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  RESULT")
        print(f"{'='*60}")
        print(f"  Panel names: {result['panel_names']}")
        for p in panels_with_bbox:
            bbox_str = p.get('bbox', 'N/A')
            print(f"    {p['name']:30s} bbox={bbox_str}")
        print(f"  Iterations:  {result['iterations']}")
        print(f"  Tool calls:  {result['total_tool_calls']}")
        print(f"  Time:        {result['elapsed_s']}s")
        print(f"  Reasoning:   {result['reasoning'][:200]}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Plan-and-Execute Panel Name Agent")
    parser.add_argument("--page", required=True, help="Path to page image PNG")
    parser.add_argument("--max-iter", type=int, default=10, help="Max agent iterations")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    settings = Settings()
    client = create_llm_client(
        endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
    deploy = settings.azure_openai_deployment

    page_path = str(Path(args.page).resolve())
    if not Path(page_path).exists():
        print(f"ERROR: {page_path} not found")
        sys.exit(1)

    print(f"Page: {page_path}")
    print(f"Model: {deploy}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Work dir: {WORK_DIR}")

    result = run_agent(
        page_image_path=page_path,
        llm_client=client,
        deployment=deploy,
        max_iterations=args.max_iter,
        verbose=not args.quiet,
    )

    # Save result
    out_path = WORK_DIR / "result.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nResult saved to: {out_path}")

    # Generate final overlay image with verified bboxes
    panels_with_bbox = result.get("panels", [])
    if panels_with_bbox and any(p.get("bbox") for p in panels_with_bbox):
        try:
            final_overlay = tool_overlay_names(page_path, panels_with_bbox)
            parsed_ov = json.loads(final_overlay)
            final_ov_path = parsed_ov.get("overlay_path", "")
            if final_ov_path:
                import shutil
                dest = str(WORK_DIR / "final_overlay.png")
                shutil.copy2(final_ov_path, dest)
                print(f"Final overlay: {dest}")
        except Exception as e:
            print(f"⚠️  Could not generate final overlay: {e}")



if __name__ == "__main__":
    main()
