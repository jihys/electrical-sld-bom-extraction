"""LLM prompt templates and JSON schemas for panel extraction agents.

Ported from the existing cad-image-understanding/src/panel_prompts.py
with adjustments for the agentic pipeline.
"""

from __future__ import annotations

import json
from typing import Dict

# ──────────────────────────────────────────────────────────────────────────────
# JSON response schemas
# ──────────────────────────────────────────────────────────────────────────────

LOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "bbox": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[x1, y1, x2, y2] pixel bbox of the panel area.",
        },
        "reasoning": {"type": "string"},
    },
    "required": ["found", "bbox"],
}

_EDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "expand", "shrink"]},
        "issue": {"type": "string"},
        "delta": {"type": "integer"},
        "corrected": {"type": "integer"},
    },
    "required": ["status", "delta", "corrected"],
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "edges": {
            "type": "object",
            "properties": {
                "x1": _EDGE_SCHEMA,
                "y1": _EDGE_SCHEMA,
                "x2": _EDGE_SCHEMA,
                "y2": _EDGE_SCHEMA,
            },
            "required": ["x1", "y1", "x2", "y2"],
        },
        "corrected_bbox": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
        },
        "reasoning": {"type": "string"},
    },
    "required": ["valid", "edges", "corrected_bbox"],
}

BAY_SCHEMA = {
    "type": "object",
    "properties": {
        "n_bays": {"type": "integer"},
        "bboxes": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
            },
            "minItems": 1,
        },
        "reasoning": {"type": "string"},
    },
    "required": ["n_bays", "bboxes"],
}

EXTRACT_NAMES_SCHEMA = {
    "type": "object",
    "properties": {
        "extractable": {"type": "boolean"},
        "panel_names": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {"type": "string"},
    },
    "required": ["extractable", "panel_names"],
}

DEDUP_NAMES_SCHEMA = {
    "type": "object",
    "properties": {
        "panel_names": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning": {"type": "string"},
    },
    "required": ["panel_names"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────────────────

def build_locate_prompt(
    panel_name: str,
    width: int,
    height: int,
    grid_size: int,
    locate_schema: Dict | None = None,
) -> str:
    schema = locate_schema or LOCATE_SCHEMA
    return f"""
Goal: Find a single bounding box that encompasses the entire area of electrical panel "{panel_name}".

[STEP 1] Locate the target panel
- The blue box (NAME:{panel_name}) indicates the exact coordinates of the target panel name text.
- Using those coordinates as a reference, identify the panel boundary (solid or dashed box).

[STEP 2] Independently identify all 4 edges of the panel boundary
!! TOP PRIORITY RULE: The bbox MUST contain the blue box (NAME:{panel_name}) coordinates inside it !!

Determine each edge independently, using grid labels as reference:

== x1 (left edge): Subtract 2-5px from the x-coordinate of the panel's left boundary line.
== y1 (top edge): Subtract 2-5px from the y-coordinate of the panel's top boundary line.
== x2 (right edge): Add 2-5px to the x-coordinate of the panel's right boundary line.
== y2 (bottom edge): Add 2-5px to the y-coordinate of the panel's bottom boundary line.

!! Each edge is determined independently. Do NOT encroach on adjacent panels !!

Current information:
- target panel name: {panel_name}
- image size: width={width}, height={height}
- grid size: {grid_size}px
- bbox format: [x1, y1, x2, y2] (pixel)

Return JSON only:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


def build_verify_prompt(
    bbox: list,
    width: int,
    height: int,
    verify_schema: Dict | None = None,
) -> str:
    schema = verify_schema or VERIFY_SCHEMA
    x1, y1, x2, y2 = bbox
    return f"""
Goal: Verify whether the orange box (PANEL) area is a correct crop of the electrical panel.

Current PANEL bbox: [x1={x1}, y1={y1}, x2={x2}, y2={y2}]
Image size: width={width}, height={height}

Validation criteria:
[TOP PRIORITY] If all electrical components inside the target panel are fully included without being clipped, it is OK.
If the boundary line is slightly clipped but components are intact, it is OK.
If components from adjacent panels are heavily included, SHRINK.

Analyze each edge (x1/y1/x2/y2) independently:
- ok: No issues
- expand: Insufficient coverage (needs to expand by delta px)
- shrink: Excessive coverage (needs to shrink by delta px)

Return JSON only:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


def build_bay_prompt(
    panel_name: str,
    width: int,
    height: int,
    grid_size: int,
    bay_schema: Dict | None = None,
) -> str:
    schema = bay_schema or BAY_SCHEMA
    return f"""
Goal: Find bay divider lines within the electrical panel "{panel_name}" image and return per-bay bounding boxes.

[STEP 1] Determine the number of bays
- Scan the image horizontally and count the bays separated by gaps + vertical solid lines.
- If there are no bay divider lines, set n_bays=1.

[STEP 2] Per-bay bounding boxes
- Determine bay bboxes in left-to-right order.
- If there is only 1 bay, bboxes = [[0, 0, {width}, {height}]].

Current information:
- panel name: {panel_name}
- image size: width={width}, height={height}
- grid size: {grid_size}px

Return JSON only:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


def build_name_extract_prompt() -> str:
    return """
You are analyzing a cropped section of an electrical Single Line Diagram (SLD).

## Task
Extract the short alphanumeric codes that NAME each enclosed panel bay in this SLD crop.
Each drawing has its own naming convention — read exactly what is written.

## Most important principle
Panel name labels share a CONSISTENT visual pattern throughout the drawing:
same shape (rectangle or hexagon), same border style, same position relative to bay
boundaries. Identify that repeating pattern first, then extract every instance of it.

## Step 1 — Identify the label pattern
Find the repeating bordered boxes (rectangle or hexagon) that name each bay.
Note shape, position, and line count. Apply the pattern consistently to all bays.

## What counts as a panel bay name
A panel bay name identifies a ZONE or SECTION of the switchboard — a bounded area
that encloses multiple electrical components (breakers, busbars, CTs, etc.) together.
The label box may sit ON or JUST OUTSIDE the boundary line of that zone — it does not
need to be inside the bounded area. Look for small labeled boxes (rectangle or hexagon)
at the corners or edges of dashed/solid boundary lines.
Panel names often follow a pattern combining voltage class, floor, and bay identifier
(e.g. 'LV 4F-1': low-voltage panel, 4th floor, bay 1 / 'HV 5F-2A': high-voltage,
5th floor, bay 2A / 'TR 5F': transformer panel, 5th floor). Not all drawings use
this convention, but such prefixes are part of the panel name and must be included.

## Step 2 — Extract
- Panel bay names are often written across TWO LINES inside the label box.
  Combine the top line and bottom line into a single string (with space as separator).
  Do NOT output each line separately.
  Example: top='LV', bottom='4F-CON' → output 'LV 4F-CON'
- Slashes in codes (e.g. 'X/B'): transcribe exactly — do not drop the slash.
- Pure letter codes are valid if they label an enclosed zone.
- If the same label pattern repeats for multiple bays (e.g. RF 2F-1A, RF 2F-2A, RF 3F-1A …),
  list every distinct name — do not collapse repeating patterns into one.

## Exclude
- Room/area names, title block text, plain English phrases with no alphanumeric code
- Labels accompanied by 'TO:' text — these are feeder/circuit destination identifiers,
  not panel bay names (e.g. a box showing 'MCCE-1-F1' next to 'TO: ATS-1-F1' is a
  feeder label, not a bay name)
- Individual equipment/component type codes on device symbols (breakers, transformers,
  current/voltage sensors, protection relays, switches, etc.) — these name a device,
  not a bay area. Note: 'UPS', 'BAT' combined with a floor/number identifier
  (e.g. 'UPS 3F-2', 'BAT 3F-5') are panel bay names — include them.
- Product brand/model numbers, electrical specs (kV, kA, MVA), revision notes

## Cut-off rule
If a label box touches or crosses ANY edge of this image → omit it.
Only include names whose box is fully visible. Other tiles will capture cut-off labels.

## Extractability
`extractable: false` only if this tile has NO bay section boxes at all or all labels
are blanked out. Bays present but no labels → `extractable: true`, `panel_names: []`.

Return JSON only:
""".strip() + "\n" + json.dumps(EXTRACT_NAMES_SCHEMA, indent=2)


def build_name_dedup_prompt(candidates: list[str]) -> str:
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    return f"""
You are reviewing electrical panel bay label extraction for one page of an SLD.

## Candidate panel bay labels (from tile crops of this page)
{candidates_json}

## Your task: CLEAN and DEDUPLICATE only
All output names must appear in the candidates list above (possibly in variant spelling).
Adding new names that are not in the candidates list is not needed.

### Cleaning rules
1. Keep ALL candidates. Do not remove any name.
2. Fix typos and normalize formatting within candidates:
   - Add missing hyphens: 'EHV8' → 'E-HV-08', 'EHV2A' → 'E-HV-02A'
   - Normalize zero-padding to the dominant format in each series
   - OCR corrections: 'E-HV-0B' → 'E-HV-08' (B≈8), 'E-HV-0l' → 'E-HV-01' (l≈1)
   - Merge OCR variants of the same label into one canonical form
   Keep BOTH if they could be genuinely distinct panels.
2b. Fix duplicate-in-series gaps: replace ONE duplicate with the missing adjacent
   number when gap == 1.  Example: [E-HV-08, E-HV-08, E-HV-10] → E-HV-09 replaces
   one E-HV-08.  Replaces a duplicate — does not increase total count.

Return JSON only:
{json.dumps(DEDUP_NAMES_SCHEMA, indent=2)}
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# Batch locate / verify schemas and prompts
# Adapted from cad-image-understanding/src/panel_prompts.py
# ──────────────────────────────────────────────────────────────────────────────

LOCATE_ALL_SCHEMA = {
    "type": "object",
    "properties": {
        "panels": {
            "type": "array",
            "description": "One entry per panel name, covering ALL requested panels.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "found": {"type": "boolean"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "[x1, y1, x2, y2] pixel bbox of the entire panel area.",
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["name", "found", "bbox"],
            },
        },
    },
    "required": ["panels"],
}

VERIFY_ALL_SCHEMA = {
    "type": "object",
    "properties": {
        "panels": {
            "type": "array",
            "description": "One entry per panel, covering ALL requested panels.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "valid": {"type": "boolean"},
                    "edges": {
                        "type": "object",
                        "properties": {
                            "x1": _EDGE_SCHEMA,
                            "y1": _EDGE_SCHEMA,
                            "x2": _EDGE_SCHEMA,
                            "y2": _EDGE_SCHEMA,
                        },
                        "required": ["x1", "y1", "x2", "y2"],
                    },
                    "corrected_bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": ["name", "valid", "edges", "corrected_bbox"],
            },
        },
    },
    "required": ["panels"],
}


def build_locate_all_prompt(
    panel_names: list,
    width: int,
    height: int,
    grid_size: int,
    has_guide: bool = False,
) -> str:
    names_str = ", ".join(f'"{n}"' for n in panel_names)
    guide_note = (
        "- First image: reference example showing panel box boundary styles.\n"
        if has_guide else ""
    )
    return f"""
Goal: Find a bbox enclosing the entire area of EACH of the following electrical panels IN ONE RESPONSE.
Panel names: {names_str}

[Input images]
{guide_note}- {"Next" if has_guide else "The"} image: full page with ALL panel name locations marked as blue "NAME:panel_name" boxes and a coordinate grid.

[STEP 1] For each panel: locate its NAME box → identify its panel boundary
- Each blue box labeled "NAME:xxx" is the exact text coordinate of panel "xxx".
- Use that coordinate as reference to find the corresponding panel boundary (solid or dashed box).

[STEP 2] For each panel: independently determine all 4 edges
!! TOP PRIORITY RULE: each panel's bbox MUST fully contain its corresponding NAME box !!
!! Adjacent panels must NEVER be encroached upon by another panel's bbox !!

━━ x1 (left edge): find left boundary, subtract 2~5px. Exclude adjacent panel content.
━━ y1 (top edge): find top boundary, subtract 2~5px. Exclude adjacent panel content.
   Note: equipment outside the boundary but connected by wiring belongs to this panel
   if the same layout pattern repeats across multiple adjacent panels → expand y1.
━━ x2 (right edge): find right boundary, add 2~5px. Exclude adjacent panel content.
━━ y2 (bottom edge): find bottom boundary, add 2~5px. Exclude adjacent panel content.

Current information:
- panel names: {names_str}
- image size: width={width}, height={height}
- grid size: {grid_size}px
- bbox format: [x1, y1, x2, y2] (pixel)

Return JSON with one entry per panel (include ALL {len(panel_names)} panels):
{json.dumps(LOCATE_ALL_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()


def build_verify_all_prompt(
    panels: list,
    width: int,
    height: int,
    has_guide: bool = False,
) -> str:
    """Build verify-all prompt for N panels in one LLM call.

    Expected image order:
      [guide image (optional)] overlay [crop_1] [crop_2] ...
    """
    names_str = ", ".join(f'"{p["name"]}"' for p in panels)
    guide_note = (
        "- First image: reference example showing panel box boundary styles.\n"
        if has_guide else ""
    )
    overlay_ref = "Next image" if has_guide else "First image"

    crop_lines = []
    for i, p in enumerate(panels):
        crop_lines.append(f'  Panel "{p["name"]}": crop #{i + 1} in the images after the overlay')
    crop_block = "\n".join(crop_lines)

    bbox_lines = []
    for p in panels:
        x1, y1, x2, y2 = p["bbox"]
        bbox_lines.append(f'  "{p["name"]}": bbox=[x1={x1}, y1={y1}, x2={x2}, y2={y2}]')
    bbox_block = "\n".join(bbox_lines)

    edge_blocks = []
    for p in panels:
        x1, y1, x2, y2 = p["bbox"]
        edge_blocks.append(
            f'Panel "{p["name"]}":\n'
            f'  x1={x1}: ok→delta=0/corrected={x1} | expand→corrected={x1}-delta | shrink→corrected={x1}+delta\n'
            f'  y1={y1}: ok→delta=0/corrected={y1} | expand→corrected={y1}-delta | shrink→corrected={y1}+delta\n'
            f'  x2={x2}: ok→delta=0/corrected={x2} | expand→corrected={x2}+delta | shrink→corrected={x2}-delta\n'
            f'  y2={y2}: ok→delta=0/corrected={y2} | expand→corrected={y2}+delta | shrink→corrected={y2}-delta'
        )
    edge_block = "\n\n".join(edge_blocks)

    return f"""
Goal: Verify that each of the following {len(panels)} electrical panel crops is correct.
Panels: {names_str}

[Input images]
{guide_note}- {overlay_ref}: combined grid overlay — shows ALL proposed PANEL:name boxes (orange) and all NAME:name boxes (blue).
- Subsequent images: individual panel crop images, one per panel, in this order:
{crop_block}

Current PANEL bboxes (image size: width={width}, height={height}):
{bbox_block}

Verification purpose: Can all electrical components of each target panel be identified
completely without confusion with adjacent panels?

[Top priority] If all electrical components inside the target panel are fully included
without being cut off, the result is basically ok.
Even if the boundary line itself is slightly cut off, judge as ok if the electrical
components are intact. Adjacent panel text or boundary lines being partially visible is ok.
However, if other panel components/wiring are heavily included causing confusion → shrink.

Verification criteria (apply independently to each panel):
1) Whether target panel electrical components are cut off — MOST IMPORTANT
   - If electrical component symbols (breakers, CT, ZCT, relays, etc.) are cut off → expand
   - If electrical components are fully included, that edge is ok
2) Whether boundary lines are cut off — only a reference when components are intact
   - Only expand when boundary line has completely disappeared and component loss is a concern
3) Whether adjacent panel area is encroached upon
   - Heavy inclusion of another panel's wiring/components → shrink
   - If only another panel's name text is visible → ok

!! Analyze all 4 edges (x1, y1, x2, y2) independently for EACH panel !!

Edge correction values:
{edge_block}

valid = true only when all 4 edge statuses are 'ok'
corrected_bbox = [x1.corrected, y1.corrected, x2.corrected, y2.corrected]

Return JSON with exactly {len(panels)} entries:
{json.dumps(VERIFY_ALL_SCHEMA, ensure_ascii=False, indent=2)}
""".strip()
