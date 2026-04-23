import json
from typing import Dict

LOCATE_SCHEMA = {
    'type': 'object',
    'properties': {
        'found': {'type': 'boolean'},
        'bbox': {
            'type': 'array',
            'items': {'type': 'integer'},
            'minItems': 4,
            'maxItems': 4,
            'description': '[x1, y1, x2, y2] bbox of the full outer rectangle that encloses the entire panel area in crop image pixel coordinates.',
        },
        'exclude_regions': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'type': 'integer'},
                'minItems': 4,
                'maxItems': 4,
            },
            'description': 'List of [x1, y1, x2, y2] sub-regions within the bbox that do NOT belong to this panel (e.g. areas belonging to adjacent panels when the panel is L-shaped or non-rectangular). Empty array [] if the panel is a clean rectangle.',
        },
        'reasoning': {
            'type': 'string',
            'description': 'Explain why this bbox was chosen for the panel.',
        },
    },
    'required': ['found', 'bbox', 'exclude_regions'],
}

_EDGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'status': {
            'type': 'string',
            'enum': ['ok', 'expand', 'shrink'],
        },
        'issue': {'type': 'string', 'description': 'Description of the issue (empty string if ok)'},
        'delta': {'type': 'integer', 'description': 'Number of pixels to change (positive, 0 if ok)'},
        'corrected': {'type': 'integer', 'description': 'Corrected coordinate value (same as current if ok)'},
    },
    'required': ['status', 'delta', 'corrected'],
}

VERIFY_SCHEMA = {
    'type': 'object',
    'properties': {
        'valid': {
            'type': 'boolean',
            'description': 'true if all edge statuses are ok, false if any edge is expand or shrink.',
        },
        'edges': {
            'type': 'object',
            'description': 'Independent analysis result for each of the 4 edges.',
            'properties': {
                'x1': _EDGE_SCHEMA,
                'y1': _EDGE_SCHEMA,
                'x2': _EDGE_SCHEMA,
                'y2': _EDGE_SCHEMA,
            },
            'required': ['x1', 'y1', 'x2', 'y2'],
        },
        'corrected_bbox': {
            'type': 'array',
            'items': {'type': 'integer'},
            'minItems': 4,
            'maxItems': 4,
            'description': '[edges.x1.corrected, edges.y1.corrected, edges.x2.corrected, edges.y2.corrected]',
        },
        'exclude_regions': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'type': 'integer'},
                'minItems': 4,
                'maxItems': 4,
            },
            'description': 'Updated list of [x1, y1, x2, y2] sub-regions within the corrected_bbox that do NOT belong to this panel. Empty array [] if the panel is a clean rectangle.',
        },
        'reasoning': {'type': 'string'},
    },
    'required': ['valid', 'edges', 'corrected_bbox', 'exclude_regions'],
}


def _guide_desc(num_guides: int) -> str:
    """Build the reference image description block for N guide images."""
    lines = []
    for i in range(1, num_guides + 1):
        lines.append(f"- Image{i} (panel_box_explanation): A panel box is the region where electrical equipment is arranged,")
        lines.append(f"  marked together with the panel name. Boundary line styles may vary by drawing;")
        lines.append(f"  the examples in the image show representative patterns.")
        if i == 1:
            lines.append("  Example 1) A closed rectangle with dashed lines on all 4 sides.")
            lines.append("  Example 2) Top/bottom = dashed lines, right = solid line, left = Gap + vertical solid line.")
            lines.append("  Example 3) Non-rectangular (L-shaped / ㄱ-shaped) panel — return the full outer rectangle")
            lines.append("    that wraps the entire region, along with any sub-regions to exclude (exclude_regions)")
            lines.append("    when the panel is not a rectangular shape.")
            lines.append("  (Not limited to these 3 patterns only.)")
    return '\n'.join(lines)


def build_locate_prompt(
    panel_name: str,
    width: int,
    height: int,
    grid_size: int,
    locate_schema: Dict,
    num_guides: int = 1,
) -> str:
    g = num_guides  # first non-guide image index
    return f"""
Goal: Find a single bbox that encloses the entire area of electrical panel "{panel_name}".

[Reference image description]
{_guide_desc(g)}

[Input image]
- Image{g+1}: the full crop with grid overlay and panel name locations marked.

[STEP 1] Locate the target panel
- The position indicated by the blue box (NAME:{panel_name}) is the exact coordinate of the
  target panel name text.
- Use that coordinate as reference to identify the panel boundary (solid or dashed box).

[STEP 2] Independently identify all 4 edges of the panel boundary
!! TOP PRIORITY RULE: the bbox MUST contain the blue box (NAME:{panel_name}) coordinate inside it !!

Determine each edge boundary independently, using the grid labels as reference:

━━ x1 (left edge)
  Find the x-coordinate of the panel left boundary (dashed/solid/Gap+vertical line) and subtract 2~5px.
  Make sure no parts/text from the left adjacent panel shown in the green box (OTHER) are included.

━━ y1 (top edge)
  Find the y-coordinate of the panel top boundary and subtract 2~5px.
  Make sure no area from the top adjacent panel is included.

  !! Rule for including equipment outside the panel box boundary !!
  In electrical drawings, some electrical components/equipment may be located outside the panel
  box boundary but still belong to the panel because they are directly connected by wiring.
  Judgment method: if the same layout pattern (arrangement/form) repeats across multiple adjacent
  panels in the drawing, treat that equipment as a component unique to each panel and expand y1
  up to the top of that equipment even if it is outside the box boundary.
  However, if there is no repeating pattern or adjacent panel ownership is clear, keep the boundary.

━━ x2 (right edge)
  Find the x-coordinate of the panel right boundary and add 2~5px.
  Make sure no parts/text from the right adjacent panel shown in the green box (OTHER) are included.

━━ y2 (bottom edge)
  Find the y-coordinate of the panel bottom boundary and add 2~5px.
  Make sure no area from the bottom adjacent panel is included.

!! Each edge must be determined independently — adjacent panels must NEVER be encroached upon !!

[STEP 3] Handle non-rectangular (L-shaped / ㄱ-shaped) panels
- Some panels are NOT simple rectangles — they may be L-shaped, ㄱ-shaped, or other non-rectangular forms.
- In such cases, the bbox should be the FULL OUTER rectangle that encloses the entire panel region.
- Then identify the sub-regions within that outer bbox that do NOT belong to this panel
  (i.e. areas belonging to adjacent panels or empty space outside the panel boundary).
- Return these as exclude_regions: list of [x1, y1, x2, y2] rectangles (in crop image pixel coordinates).
- If the panel is a clean rectangle, return exclude_regions: [].

Current information:
- target panel name: {panel_name}
- image size: width={width}, height={height}
- grid size: {grid_size}px
- bbox format: [x1, y1, x2, y2] (pixel)

Return JSON only:
{json.dumps(locate_schema, ensure_ascii=False, indent=2)}
""".strip()


BAY_SCHEMA = {
    'type': 'object',
    'properties': {
        'n_bays': {
            'type': 'integer',
            'description': 'Number of bays inside the panel (1 if no bays)',
        },
        'bboxes': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'type': 'integer'},
                'minItems': 4,
                'maxItems': 4,
            },
            'minItems': 1,
            'description': 'List of [x1, y1, x2, y2] bbox per bay, ordered left to right. If 1 bay, use the full image bbox.',
        },
        'reasoning': {'type': 'string'},
    },
    'required': ['n_bays', 'bboxes'],
}


def _bay_guide_desc(num_guides: int) -> str:
    """Build the reference image description block for bay guide images."""
    lines = []
    for i in range(1, num_guides + 1):
        lines.append(f"- Image{i} (bay_example): Guide to how bays are divided when 1 panel is split into multiple bays.")
        if i == 1:
            lines.append("  Bays are separated by the combination of a Gap (empty space) + vertical solid line.")
    return '\n'.join(lines)


def build_bay_prompt(panel_name: str, width: int, height: int, grid_size: int, schema: Dict, num_guides: int = 1) -> str:
    g = num_guides
    overlay_idx = g + 1
    return f"""
Goal: Find the bay dividers within the electrical panel "{panel_name}" image and return the bbox for each bay.
(The entire input image is the area of this panel.)

[Reference image description]
{_bay_guide_desc(g)}

[Input image]
- Image{overlay_idx}: the panel image with grid overlay.

[STEP 1] Count the number of bays
- Scan the image horizontally and count how many bays are separated by Gap + vertical solid lines.
- If there are no bay dividers, n_bays=1 (the entire panel is a single section).
- !! The bboxes list MUST contain exactly as many bboxes as the actual number of bays !!

[STEP 2] Determine the bbox range for each bay
- Each bbox includes the bay boundary (including dividers) with a 2~5px margin.
- Bays are stored in the bboxes list in left-to-right order.
- Adjust boundaries so that electrical component symbols are not cut off at bay edges.
- If there is 1 bay, bboxes = [[0, 0, {width}, {height}]] (full image).

Current information:
- panel name: {panel_name}
- image size: width={width}, height={height}
- grid size: {grid_size}px
- bbox format: [x1, y1, x2, y2] (pixel)

Return JSON only:
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


LOCATE_ALL_SCHEMA = {
    'type': 'object',
    'properties': {
        'panels': {
            'type': 'array',
            'description': 'One entry per panel name, covering ALL requested panels.',
            'items': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'Exact panel name as provided.'},
                    'found': {'type': 'boolean'},
                    'bbox': {
                        'type': 'array',
                        'items': {'type': 'integer'},
                        'minItems': 4,
                        'maxItems': 4,
                        'description': '[x1, y1, x2, y2] bbox of the full outer rectangle that encloses the entire panel area in crop image pixel coordinates.',
                    },
                    'exclude_regions': {
                        'type': 'array',
                        'items': {
                            'type': 'array',
                            'items': {'type': 'integer'},
                            'minItems': 4,
                            'maxItems': 4,
                        },
                        'description': 'List of [x1, y1, x2, y2] sub-regions within the bbox that do NOT belong to this panel. Empty array [] if rectangular.',
                    },
                    'reasoning': {'type': 'string'},
                },
                'required': ['name', 'found', 'bbox', 'exclude_regions'],
            },
        },
    },
    'required': ['panels'],
}


def build_locate_all_prompt(
    panel_names: list,
    width: int,
    height: int,
    grid_size: int,
    schema: Dict,
    num_guides: int = 1,
) -> str:
    g = num_guides
    names_str = ', '.join(f'"{n}"' for n in panel_names)
    return f"""
Goal: Find a bbox enclosing the entire area of EACH of the following electrical panels IN ONE RESPONSE.
Panel names: {names_str}

[Reference image description]
{_guide_desc(g)}

[Input image]
- Image{g+1}: the full crop with ALL panel name locations marked as blue "NAME:panel_name" boxes.

[STEP 0] FIRST — systematic boundary line analysis (MANDATORY)
Before locating any individual panel, you MUST perform this analysis:

(0-A) Identify all HORIZONTAL row-divider lines.
  These are long horizontal lines (dashed or solid) that span across multiple panels,
  creating horizontal "rows" of panels in the drawing.

(0-B) Identify all VERTICAL column-divider lines between horizontally adjacent panels.
  For each pair of side-by-side panels in the same row, find the vertical line(s) separating them.
  IMPORTANT: there may be MORE THAN ONE vertical divider line between two adjacent panels,
  each covering a different y-range. Look carefully!

(0-C) ★ L-SHAPE DIAGNOSTIC TEST (two independent checks) ★

  CHECK 1 — Name label position offset:
    For adjacent panels in the same row or column, compare the positions of their
    NAME boxes (the blue "NAME:xxx" labels in the image).
    (a) Horizontally adjacent panels: if their NAME boxes have significantly different
        y-positions (difference > {grid_size} pixels), this is a STRONG indicator of
        an L-shaped boundary. The panel whose name is HIGHER (smaller y) occupies a
        wider upper region; the one whose name is LOWER (larger y) occupies a wider
        lower region. The boundary shifts near the LOWER panel's name y-coordinate.
    (b) Vertically adjacent panels: if their NAME boxes have significantly different
        x-positions (difference > {grid_size} pixels), this similarly indicates a
        non-rectangular split along the x-axis.

  CHECK 2 — Vertical divider length:
    "Does the vertical divider line between two adjacent panels extend the FULL HEIGHT
     from one horizontal row-divider to the next?"
    - If YES → simple rectangles.
    - If NO → BOTH panels are L-shaped!

  If EITHER check indicates L-shape → treat both adjacent panels as L-shaped.

(0-D) Similarly, check each horizontal divider between vertically stacked panels:
    "Does this horizontal line extend the FULL WIDTH between the left and right boundaries?"
  - If NO → the panels above and below have non-rectangular boundaries.

Record your findings in the "reasoning" field for each affected panel.

[STEP 1] For each panel: locate its NAME box → identify its panel boundary
- Each blue box labeled "NAME:xxx" is the exact text coordinate of panel "xxx".
- Use that coordinate as reference to find the corresponding panel boundary (solid or dashed box).

[STEP 2] For each panel: independently determine all 4 edges
!! TOP PRIORITY RULE: each panel's bbox MUST fully contain its corresponding NAME box !!
!! The panel name label box (hexagon, rectangle, or other shape containing the panel name text)
   is part of the panel area — the bbox MUST include this label box entirely !!
!! Adjacent panels must NEVER be encroached upon by another panel's bbox !!

━━ x1 (left edge): find left boundary, subtract 2~5px. Exclude adjacent panel content.
━━ y1 (top edge): find top boundary, subtract 2~5px. Exclude adjacent panel content.
   Note: equipment outside the boundary but connected by wiring belongs to this panel
   if the same layout pattern repeats across multiple adjacent panels → expand y1.
━━ x2 (right edge): find right boundary, add 2~5px. Exclude adjacent panel content.
━━ y2 (bottom edge): find bottom boundary, add 2~5px. Exclude adjacent panel content.

[STEP 3] For each panel: handle non-rectangular (L-shaped / ㄱ-shaped) shapes
Use the findings from STEP 0 to determine each panel's shape.

(3-A) If STEP 0 diagnostic found that a vertical divider does NOT span the full row height:
  - BOTH panels on either side of that divider are L-shaped.
  - Each panel's bbox = the FULL OUTER rectangle enclosing the entire L-shaped area.
  - The bboxes of two interlocking L-shaped panels WILL OVERLAP — this is expected and correct.
  - For each panel, identify the rectangular sub-region within its bbox that belongs to the
    ADJACENT panel (the overlapping area) and list it in exclude_regions.

(3-B) CRITICAL CONSISTENCY CHECK:
  - If panel A has exclude_regions, there MUST be an adjacent panel B that also has exclude_regions.
  - It is IMPOSSIBLE for only one of two interlocking L-shaped panels to have exclude_regions.
  - The exclude_region of panel A should roughly equal the non-excluded area of panel B that overlaps
    with panel A's bbox, and vice versa.

(3-C) If the panel is a clean rectangle (all dividers span full row/column), return exclude_regions: [].
- panel names: {names_str}
- image size: width={width}, height={height}
- grid size: {grid_size}px
- bbox format: [x1, y1, x2, y2] (pixel)

Return JSON with one entry per panel (include ALL {len(panel_names)} panels):
{json.dumps(schema, ensure_ascii=False, indent=2)}
""".strip()


VERIFY_ALL_SCHEMA = {
    'type': 'object',
    'properties': {
        'panels': {
            'type': 'array',
            'description': 'One entry per panel, covering ALL requested panels.',
            'items': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'Exact panel name as provided.'},
                    'valid': {
                        'type': 'boolean',
                        'description': 'true if all edge statuses are ok, false if any edge is expand or shrink.',
                    },
                    'edges': {
                        'type': 'object',
                        'properties': {
                            'x1': _EDGE_SCHEMA,
                            'y1': _EDGE_SCHEMA,
                            'x2': _EDGE_SCHEMA,
                            'y2': _EDGE_SCHEMA,
                        },
                        'required': ['x1', 'y1', 'x2', 'y2'],
                    },
                    'corrected_bbox': {
                        'type': 'array',
                        'items': {'type': 'integer'},
                        'minItems': 4,
                        'maxItems': 4,
                        'description': '[edges.x1.corrected, edges.y1.corrected, edges.x2.corrected, edges.y2.corrected]',
                    },
                    'exclude_regions': {
                        'type': 'array',
                        'items': {
                            'type': 'array',
                            'items': {'type': 'integer'},
                            'minItems': 4,
                            'maxItems': 4,
                        },
                        'description': 'Updated list of [x1, y1, x2, y2] sub-regions within the corrected_bbox that do NOT belong to this panel. Empty array [] if rectangular.',
                    },
                },
                'required': ['name', 'valid', 'edges', 'corrected_bbox', 'exclude_regions'],
            },
        },
    },
    'required': ['panels'],
}


def build_verify_all_prompt(
    panels: list,  # [{'name': str, 'bbox': [x1,y1,x2,y2], 'exclude_regions': [...]}, ...]
    width: int,
    height: int,
    schema: Dict,
    num_guides: int = 1,
) -> str:
    """Build a batch verify prompt for N panels in a single LLM call.

    Image layout expected by this prompt:
      Image1..N    : panel_box_explanation guide(s)
      Image(N+1)   : combined overlay (all PANEL:name boxes + all NAME:name boxes)
      Image(N+2)+  : individual panel crops, one per panel, in the same order as `panels`
    """
    g = num_guides
    overlay_idx = g + 1
    first_crop_idx = g + 2
    names_str = ', '.join(f'"{p["name"]}"' for p in panels)

    # Per-panel bbox summary block
    bbox_lines = []
    for i, p in enumerate(panels):
        x1, y1, x2, y2 = p['bbox']
        img_idx = i + first_crop_idx
        excl = p.get('exclude_regions', [])
        excl_str = f', exclude_regions={json.dumps(excl)}' if excl else ''
        bbox_lines.append(
            f'  Panel "{p["name"]}": bbox=[x1={x1}, y1={y1}, x2={x2}, y2={y2}]{excl_str}  → crop in Image{img_idx}'
        )
    bbox_block = '\n'.join(bbox_lines)

    # Per-panel edge instruction block
    edge_blocks = []
    for i, p in enumerate(panels):
        x1, y1, x2, y2 = p['bbox']
        img_idx = i + first_crop_idx
        edge_blocks.append(
            f'Panel "{p["name"]}" (Image{img_idx}, PANEL:{p["name"]} box in Image{overlay_idx}):\n'
            f'  x1={x1}: ok→delta=0/corrected={x1} | expand→corrected={x1}-delta | shrink→corrected={x1}+delta\n'
            f'  y1={y1}: ok→delta=0/corrected={y1} | expand→corrected={y1}-delta | shrink→corrected={y1}+delta\n'
            f'  x2={x2}: ok→delta=0/corrected={x2} | expand→corrected={x2}+delta | shrink→corrected={x2}-delta\n'
            f'  y2={y2}: ok→delta=0/corrected={y2} | expand→corrected={y2}+delta | shrink→corrected={y2}-delta'
        )
    edge_block = '\n\n'.join(edge_blocks)

    return (
        f"Goal: Verify that each of the following {len(panels)} electrical panel crops is correct.\n"
        f"Panels: {names_str}\n\n"
        "[Reference image description]\n"
        f"{_guide_desc(g)}\n\n"
        "[Input images]\n"
        f"- Image{overlay_idx}: combined grid overlay — shows ALL panels' current PANEL:name boxes (orange) and\n"
        "  all NAME:name boxes (blue). Use this to understand spatial relationships between panels.\n"
        f"- Image{first_crop_idx}+: individual crop images for each panel in the order below.\n\n"
        f"Current PANEL bboxes (image size: width={width}, height={height}):\n"
        f"{bbox_block}\n\n"
        "Verification purpose: When extracting the BOM from each crop, can all electrical components\n"
        "of the target panel be identified completely without confusion with adjacent panels?\n\n"
        "[Top priority criterion] If the electrical components inside the target panel are fully\n"
        "included without being cut off, it is basically ok.\n"
        "Even if the boundary line itself (solid/dashed) is slightly cut off, judge as ok if the\n"
        "electrical components are intact.\n"
        "It is acceptable if adjacent panel text or boundary lines are partially visible. However,\n"
        "if other panel components/wiring are heavily included causing confusion about panel ownership,\n"
        "then shrink.\n\n"
        "Verification criteria (apply independently to each panel):\n"
        "1) [Required] Whether target panel electrical components are cut off — most important\n"
        "   - If electrical component symbols (breakers, CT, ZCT, relays, etc.) are cut off → expand\n"
        "   - If electrical components are fully included, that edge is ok\n"
        "2) Whether boundary lines are cut off — only a reference when electrical components are intact\n"
        "   - Only expand when the target panel boundary line has completely disappeared and\n"
        "     component loss is a concern\n"
        "   - If adjacent panel boundary line intrudes into the crop → shrink (not expand)\n"
        "3) [Core judgment] Whether adjacent panel area is encroached upon\n"
        "   - If electrical components/wiring belonging to another panel are heavily included → shrink\n"
        "   - If only another panel's name text is visible → ok\n\n"
        "!! Analyze all 4 edges (x1, y1, x2, y2) independently for EACH panel !!\n"
        "!! Use Image2 to understand which adjacent panels are nearby for each target panel !!\n\n"
        "Note for y1: Even if equipment connected by wiring is outside the panel box boundary,\n"
        "if the same layout pattern repeats across multiple adjacent panels, treat that equipment\n"
        "as belonging to this panel and apply expand.\n\n"
        "Edge correction values per panel:\n"
        f"{edge_block}\n\n"
        "valid = true only when all 4 edge statuses are 'ok'\n"
        "corrected_bbox = [x1.corrected, y1.corrected, x2.corrected, y2.corrected]\n\n"
        "[Non-rectangular panel handling]\n"
        "If a panel is L-shaped, ㄱ-shaped, or otherwise non-rectangular:\n"
        "- The bbox/corrected_bbox remains the full outer rectangle enclosing the entire panel.\n"
        "- Return exclude_regions: list of [x1, y1, x2, y2] sub-regions within the corrected_bbox\n"
        "  that do NOT belong to this panel (adjacent panel areas or empty space).\n"
        "- Verify and update the exclude_regions from the locate step if needed.\n"
        "- If the panel is a clean rectangle, return exclude_regions: [].\n\n"
        f"Return JSON with exactly {len(panels)} entries (ALL panels listed above):\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def build_verify_prompt(verify_schema: Dict, bbox: list, width: int, height: int, exclude_regions: list = None, num_guides: int = 1) -> str:
    g = num_guides
    overlay_idx = g + 1
    crop_idx = g + 2
    x1, y1, x2, y2 = bbox
    return (
        "Goal: Verify that the orange box (PANEL) region is a correct electrical panel crop.\n"
        "First identify which panel it is using the panel name text shown in the blue box (NAME:panel name).\n\n"
        "[Reference image description]\n"
        f"{_guide_desc(g)}\n\n"
        f"Current PANEL bbox: [x1={x1}, y1={y1}, x2={x2}, y2={y2}]\n"
        f"Image size: width={width}, height={height}\n\n"
        "Verification purpose: When extracting the BOM (bill of materials) from this crop, can all\n"
        "electrical components of the target panel be identified completely without confusion with\n"
        "adjacent panels?\n\n"
        "[Top priority criterion] If the electrical components inside the target panel are fully\n"
        "included without being cut off, it is basically ok.\n"
        "Even if the boundary line itself (solid/dashed) is slightly cut off, judge as ok if the\n"
        "electrical components are intact.\n"
        "It is acceptable if adjacent panel text or boundary lines are partially visible. However,\n"
        "if other panel components/wiring are heavily included causing confusion about panel ownership,\n"
        "then shrink.\n\n"
        "Verification criteria:\n"
        "1) [Required] Whether target panel electrical components are cut off — most important\n"
        "   - If electrical component symbols belonging to the target panel (breakers, CT, ZCT,\n"
        "     relays, etc.) are actually cut off and missing at the crop edge → expand that edge\n"
        "   - If electrical components are fully included, that edge is ok regardless of boundary state\n"
        "2) Whether boundary lines are cut off — only a reference when electrical components are intact\n"
        "   - !! Note: Always distinguish whether the boundary line belongs to the target panel or\n"
        "     an adjacent panel\n"
        "     · Only expand when the target panel boundary line has completely disappeared and\n"
        "       component loss is a concern\n"
        "     · If boundary line is merely close to the edge or slightly cut → ok\n"
        "     · If adjacent panel boundary line intrudes into the crop → shrink (not expand)\n"
        "3) [Core judgment] Whether adjacent panel area is encroached upon\n"
        "   - If electrical components/wiring belonging to another panel are heavily included\n"
        "     causing confusion → shrink\n"
        "   - If only another panel's name text is visible → ok (shrink not needed)\n\n"
        "!! All 4 edges (x1, y1, x2, y2) must be analyzed independently !!\n\n"
        "Determine status / delta / corrected for each edge:\n\n"
        f"━━ x1 (left edge, current={x1}) ━━\n"
        f"  status='ok'     : No issue on left side → delta=0, corrected={x1}\n"
        f"  status='expand' : Left boundary/component is cut off → corrected = {x1} - delta (expand left)\n"
        f"  status='shrink' : Adjacent panel intrudes on left → corrected = {x1} + delta (shrink right)\n\n"
        f"━━ y1 (top edge, current={y1}) ━━\n"
        f"  status='ok'     : No issue on top → delta=0, corrected={y1}\n"
        f"  status='expand' : Top boundary/component is cut off → corrected = {y1} - delta (expand up)\n"
        f"  status='shrink' : Adjacent panel intrudes on top → corrected = {y1} + delta (shrink down)\n"
        "  Note: Even if equipment connected by wiring is outside the panel box boundary,\n"
        "    if the same layout pattern repeats across multiple adjacent panels, treat that\n"
        "    equipment as belonging to this panel and apply expand.\n\n"
        f"━━ x2 (right edge, current={x2}) ━━\n"
        f"  status='ok'     : No issue on right side → delta=0, corrected={x2}\n"
        f"  status='expand' : Right boundary/component is cut off → corrected = {x2} + delta (expand right)\n"
        f"  status='shrink' : Adjacent panel intrudes on right → corrected = {x2} - delta (shrink left)\n\n"
        f"━━ y2 (bottom edge, current={y2}) ━━\n"
        f"  status='ok'     : No issue on bottom → delta=0, corrected={y2}\n"
        f"  status='expand' : Bottom boundary/component is cut off → corrected = {y2} + delta (expand down)\n"
        f"  status='shrink' : Adjacent panel intrudes on bottom → corrected = {y2} - delta (shrink up)\n\n"
        "valid = true only when all edge statuses are 'ok'\n"
        "corrected_bbox = [x1.corrected, y1.corrected, x2.corrected, y2.corrected]\n\n"
        "[Non-rectangular panel handling]\n"
        "If the panel is L-shaped, ㄱ-shaped, or otherwise non-rectangular:\n"
        "- The bbox remains the full outer rectangle enclosing the entire panel.\n"
        "- Return exclude_regions: list of [x1, y1, x2, y2] sub-regions within the corrected_bbox\n"
        "  that do NOT belong to this panel (adjacent panel areas or empty space).\n"
        "- If the panel is a clean rectangle, return exclude_regions: [].\n"
        + (f"- Current exclude_regions from locate step: {json.dumps(exclude_regions)}\n"
           "  Verify and update these if needed based on the corrected_bbox.\n"
           if exclude_regions else "") +
        "\nInput images:\n"
        f"- Image1..{g}: panel_box_explanation (panel boundary examples)\n"
        f"- Image{overlay_idx}: grid overlay (showing panel area extracted with PANEL box)\n"
        "  · PANEL box  = extracted panel region\n"
        "  · Blue box   = NAME:panel name (location of target panel name text)\n"
        "  · Green box  = OTHER:panel name (locations of adjacent panel name texts)\n"
        f"- Image{crop_idx}: actual cropped panel image\n\n"
        "Return JSON only:\n"
        + json.dumps(verify_schema, ensure_ascii=False, indent=2)
    )
