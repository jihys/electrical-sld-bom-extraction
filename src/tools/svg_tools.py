"""SVG parsing and grid extraction tools."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Annotated, Dict, List, Tuple

from pydantic import Field


def extract_svg_segments(
    svg_path: Annotated[str, Field(description="Path to the SVG file")],
) -> Dict[str, List]:
    """Extract horizontal and vertical line segments from an SVG file.

    Returns {"h_lines": [...], "v_lines": [...]} where each line is
    (coordinate, min_extent, max_extent, stroke_width).
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    h_lines: List[Tuple[float, float, float, float]] = []
    v_lines: List[Tuple[float, float, float, float]] = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if tag == "line":
            x1 = float(elem.get("x1", 0))
            y1 = float(elem.get("y1", 0))
            x2 = float(elem.get("x2", 0))
            y2 = float(elem.get("y2", 0))
            sw = _parse_stroke_width(elem)
            if abs(y2 - y1) < 1.0 and abs(x2 - x1) > 5.0:  # horizontal
                h_lines.append((min(y1, y2), min(x1, x2), max(x1, x2), sw))
            elif abs(x2 - x1) < 1.0 and abs(y2 - y1) > 5.0:  # vertical
                v_lines.append((min(x1, x2), min(y1, y2), max(y1, y2), sw))

        elif tag == "path":
            d = elem.get("d", "")
            sw = _parse_stroke_width(elem)
            _parse_path_segments(d, h_lines, v_lines, sw)

    return {"h_lines": h_lines, "v_lines": v_lines}


def build_grid(
    segments: Annotated[Dict[str, List], Field(description="Output of extract_svg_segments")],
    cluster_tolerance: float = 3.0,
) -> Dict[str, List]:
    """Cluster H/V segments into aligned grid lines.

    Returns {"h_grid": [(y, x_min, x_max), ...], "v_grid": [(x, y_min, y_max), ...]}.
    """
    h_clustered = _cluster_lines(segments["h_lines"], cluster_tolerance)
    v_clustered = _cluster_lines(segments["v_lines"], cluster_tolerance)
    return {"h_grid": h_clustered, "v_grid": v_clustered}


def find_nearest_boundaries(
    grid: Annotated[Dict[str, List], Field(description="Output of build_grid")],
    seed_x: Annotated[float, Field(description="Seed point x coordinate")],
    seed_y: Annotated[float, Field(description="Seed point y coordinate")],
) -> List[int]:
    """Find enclosing grid lines nearest to a seed point.

    Returns [x1, y1, x2, y2] from the nearest H/V grid lines.
    """
    h_grid = sorted(grid["h_grid"], key=lambda l: l[0])
    v_grid = sorted(grid["v_grid"], key=lambda l: l[0])

    y1 = 0
    for coord, _, _ in h_grid:
        if coord <= seed_y:
            y1 = int(coord)
        else:
            break

    y2 = int(h_grid[-1][0]) if h_grid else 0
    for coord, _, _ in h_grid:
        if coord > seed_y:
            y2 = int(coord)
            break

    x1 = 0
    for coord, _, _ in v_grid:
        if coord <= seed_x:
            x1 = int(coord)
        else:
            break

    x2 = int(v_grid[-1][0]) if v_grid else 0
    for coord, _, _ in v_grid:
        if coord > seed_x:
            x2 = int(coord)
            break

    return [x1, y1, x2, y2]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_stroke_width(elem) -> float:
    style = elem.get("style", "")
    m = re.search(r"stroke-width:\s*([0-9.]+)", style)
    if m:
        return float(m.group(1))
    sw = elem.get("stroke-width", "1")
    try:
        return float(sw)
    except ValueError:
        return 1.0


def _parse_path_segments(d: str, h_lines, v_lines, sw: float):
    """Parse simple M/L/H/V path commands into line segments."""
    tokens = re.findall(r"[MLHVZmlhvz]|[-+]?[0-9]*\.?[0-9]+", d)
    cx, cy = 0.0, 0.0
    i = 0
    while i < len(tokens):
        cmd = tokens[i]
        if cmd in ("M", "m"):
            i += 1
            if i + 1 < len(tokens):
                nx, ny = float(tokens[i]), float(tokens[i + 1])
                if cmd == "m":
                    cx, cy = cx + nx, cy + ny
                else:
                    cx, cy = nx, ny
                i += 2
        elif cmd == "L":
            i += 1
            if i + 1 < len(tokens):
                nx, ny = float(tokens[i]), float(tokens[i + 1])
                _classify_segment(cx, cy, nx, ny, sw, h_lines, v_lines)
                cx, cy = nx, ny
                i += 2
        elif cmd == "H":
            i += 1
            if i < len(tokens):
                nx = float(tokens[i])
                _classify_segment(cx, cy, nx, cy, sw, h_lines, v_lines)
                cx = nx
                i += 1
        elif cmd == "V":
            i += 1
            if i < len(tokens):
                ny = float(tokens[i])
                _classify_segment(cx, cy, cx, ny, sw, h_lines, v_lines)
                cy = ny
                i += 1
        else:
            i += 1


def _classify_segment(x1, y1, x2, y2, sw, h_lines, v_lines):
    if abs(y2 - y1) < 1.0 and abs(x2 - x1) > 5.0:
        h_lines.append((min(y1, y2), min(x1, x2), max(x1, x2), sw))
    elif abs(x2 - x1) < 1.0 and abs(y2 - y1) > 5.0:
        v_lines.append((min(x1, x2), min(y1, y2), max(y1, y2), sw))


def _cluster_lines(lines, tolerance: float) -> List[Tuple[float, float, float]]:
    """Cluster line segments by their primary coordinate."""
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda l: l[0])
    clusters: List[List] = [[sorted_lines[0]]]
    for line in sorted_lines[1:]:
        if abs(line[0] - clusters[-1][-1][0]) <= tolerance:
            clusters[-1].append(line)
        else:
            clusters.append([line])

    result = []
    for cluster in clusters:
        coord = sum(l[0] for l in cluster) / len(cluster)
        min_ext = min(l[1] for l in cluster)
        max_ext = max(l[2] for l in cluster)
        result.append((coord, min_ext, max_ext))
    return result
