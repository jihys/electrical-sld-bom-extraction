"""Streamlit HITL Pipeline UI for CAD Panel Extraction.

True pipeline (matching cad-image-understanding notebooks):
  Step 1: PDF → PNG conversion (NB01)
  Step 2: DI 2-pass figure detection (NB02)
  Step 3: Missing image detection via LLM (NB03)
    → HITL #1: confirm/edit detected figure region bboxes
  Step 4: Panel name extraction + bbox matching (NB04)
    → HITL #2: confirm/edit panel names + name-label bboxes
  Step 5: Panel area locate-verify + bay split (NB05)
    → HITL #3: confirm/edit panel area + bay bboxes

Key insight: NB02-03 detect "figure regions" (generic drawing areas) — NOT panel areas.
A single figure region may contain multiple panels. NB04 finds panel NAMES within those
regions. NB05 then uses each panel name's text-label bbox as an anchor to locate the full
panel AREA boundary (the dashed/solid box around each panel's equipment).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Ensure project root on sys.path ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import Settings

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

STEP_LABELS = [
    "1. PDF → PNG",
    "2. DI Figure Detection",
    "3. Missing Image Detection",
    "4. Panel Name Extraction",
    "5. Panel Area / Bay Split",
]

COLORS_BGR = {
    "orange": (0, 165, 255),
    "blue": (255, 0, 0),
    "green": (0, 200, 0),
    "red": (0, 0, 255),
    "purple": (200, 0, 200),
    "cyan": (255, 255, 0),
    "yellow": (0, 255, 255),
}

# Where cad-image-understanding test images live
VISUAL_PROMPT_DIR = (
    Path(__file__).resolve().parents[2].parent
    / "cad-image-understanding" / "notebooks" / "test_images"
)


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_bboxes(
    img: np.ndarray,
    bboxes: List[List[int]],
    labels: List[str],
    color: Tuple[int, int, int] = COLORS_BGR["orange"],
    thickness: int = 3,
) -> np.ndarray:
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if label:
            (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
            cv2.rectangle(out, (x1, max(y1 - th - 10, 0)),
                          (x1 + tw + 4, max(y1, th + 10)), color, -1)
            cv2.putText(out, label, (x1 + 2, max(y1 - 4, th + 6)),
                        font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_img(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    return img


def _iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ──────────────────────────────────────────────────────────────────────────────

def _s(key: str, default: Any = None):
    return st.session_state.get(key, default)


def _ss(key: str, value: Any):
    st.session_state[key] = value


def _init():
    defaults = {
        "step": 0,          # 0=not started, 1..5=completed that step
        "current_view": 0,  # which step panel is shown (0-4)
        "pdf_path": None,
        # Step 1
        "pages": [],        # List[dict] from PageInfo.model_dump()
        "selected_pages": None,   # None = all; set() = none selected; set(nums) = specific
        # Step 2
        "di_summaries": [],  # List[dict] from PageExtractionSummary.model_dump()
        "di_lines_by_page": {},
        # Step 3
        "regions_by_page": {},   # {page_num: [{bbox, source, label}]}
        "regions_confirmed": False,
        # Step 4
        "names_by_page": {},     # {page_num: [name, ...]}
        "matches_by_page": {},   # {page_num: {name: {bbox, method, ...} | None}}
        "names_confirmed": False,
        # Step 5
        "panel_crops": [],       # [{panel_name, page_num, bbox, crop_path, confidence, ...}]
        "bay_results": [],       # [{panel_name, page_num, n_bays, bay_bboxes}]
        "crops_confirmed": False,
        # Visual prompts
        "visual_prompts": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ──────────────────────────────────────────────────────────────────────────────
# Page selection helper
# ──────────────────────────────────────────────────────────────────────────────

def _active_pages() -> List[dict]:
    """Return only the pages selected for processing.

    If selected_pages is empty (or contains all pages), returns all pages.
    selected_pages is a set of page_num integers chosen by the user.
    """
    pages = _s("pages", [])
    selected = _s("selected_pages", None)
    if selected is None or selected == {p["page_num"] for p in pages}:
        return pages
    return [p for p in pages if p["page_num"] in selected]


# ──────────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_settings() -> Settings:
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(str(env_path))
    return Settings()


def _llm(settings: Settings):
    if "llm_client" not in st.session_state:
        from src.agents.llm_caller import create_llm_client
        st.session_state["llm_client"] = create_llm_client(
            settings.azure_openai_endpoint,
            settings.azure_openai_api_key,
            settings.azure_openai_api_version,
        )
    return st.session_state["llm_client"]


def _di(settings: Settings):
    if "di_client" not in st.session_state:
        from src.tools.di_tools import create_di_client
        st.session_state["di_client"] = create_di_client(
            settings.azure_di_endpoint,
            settings.azure_di_key,
        )
    return st.session_state["di_client"]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — PDF → PNG
# ══════════════════════════════════════════════════════════════════════════════

def run_step1(pdf_path: str, settings: Settings):
    from src.tools.pdf_tools import compute_dpi_for_pdf, split_pdf_to_pages
    out = str(settings.output_path / "pages")
    g_dpi, pp_dpi = compute_dpi_for_pdf(pdf_path)
    pages = split_pdf_to_pages(pdf_path, out, dpi=g_dpi, per_page_dpi=pp_dpi)
    _ss("pages", [p.model_dump() for p in pages])
    _ss("step", 1)
    _ss("current_view", 0)  # stay on step1 so user can select pages


def render_step1():
    st.subheader("Step 1 — PDF → PNG")
    pages = _s("pages", [])
    if not pages:
        st.info("Upload a PDF to begin.")
        return

    all_page_nums = sorted(p["page_num"] for p in pages)
    selected = _s("selected_pages", None)
    if selected is None:
        selected = set(all_page_nums)
        _ss("selected_pages", selected)

    st.write(f"**{len(pages)} page(s) converted.** 포함할 페이지를 선택하고 Next를 누르세요:")

    # ── Shortcut buttons (no rerun — just update state then form submit) ──
    ctrl_cols = st.columns([1, 1, 4])
    if ctrl_cols[0].button("✅ All", key="psel_all", use_container_width=True):
        _ss("selected_pages", set(all_page_nums))
        st.rerun()
    if ctrl_cols[1].button("⬜ None", key="psel_none", use_container_width=True):
        _ss("selected_pages", set())
        st.rerun()

    # ── Page selection form — no rerun until Submit ───────────────────────
    with st.form("page_selection_form"):
        cols = st.columns(min(len(pages), 4))
        cb_values: Dict[int, bool] = {}
        for i, p in enumerate(pages):
            pn = p["page_num"]
            is_on = pn in selected
            with cols[i % len(cols)]:
                border_color = "#22c55e" if is_on else "#94a3b8"
                opacity = "1.0" if is_on else "0.35"
                st.markdown(
                    f'<div style="border:3px solid {border_color};border-radius:6px;padding:2px;opacity:{opacity}">',
                    unsafe_allow_html=True,
                )
                st.image(
                    Image.open(p["png_path"]),
                    caption=f"Page {pn}  ({p['width']}×{p['height']})",
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
                cb_values[pn] = st.checkbox(f"Page {pn}", value=is_on, key=f"cb_{pn}")

        submitted = st.form_submit_button("✔ Apply Selection", use_container_width=True)
        if submitted:
            new_sel = {pn for pn, v in cb_values.items() if v}
            _ss("selected_pages", new_sel)
            st.rerun()

    n_selected = len(selected & set(all_page_nums))
    if n_selected == 0:
        st.warning("⚠️ 최소 1개 이상의 페이지를 선택하세요.")
    else:
        st.caption(f"{n_selected} / {len(pages)} 페이지 선택됨")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — DI 2-pass figure detection
# ══════════════════════════════════════════════════════════════════════════════

def run_step2(settings: Settings):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.tools.di_tools import two_pass_detection, analyze_page
    from src.models.page import PageInfo

    pages = [PageInfo(**p) for p in _active_pages()]
    client = _di(settings)
    bar = st.progress(0, "DI detection (parallel) …")
    done_count = 0

    def _process(pg: PageInfo):
        od = str(settings.output_path / f"di_page{pg.page_num}")
        s = two_pass_detection(client, "prebuilt-layout", pg.png_path, od, pg.page_num)
        a = analyze_page(client, "prebuilt-layout", pg.png_path)
        return pg.page_num, s.model_dump(), a["lines"]

    results_map: Dict[int, tuple] = {}
    max_workers = min(len(pages), 2)  # limit concurrency to avoid DI rate-limit (429)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, pg): pg for pg in pages}
        for future in as_completed(futures):
            pn, summary, lines = future.result()
            results_map[pn] = (summary, lines)
            done_count += 1
            bar.progress(done_count / len(pages), f"DI done: {done_count}/{len(pages)} pages")

    # Restore original page order
    summaries = [results_map[pg.page_num][0] for pg in pages]
    di_lines = {pg.page_num: results_map[pg.page_num][1] for pg in pages}

    _ss("di_summaries", summaries)
    _ss("di_lines_by_page", di_lines)
    _ss("step", 2)
    _ss("current_view", 2)
    bar.empty()


def render_step2():
    st.subheader("Step 2 — DI Figure Detection (2-pass)")
    summaries = _s("di_summaries", [])
    pages = _active_pages()
    if not summaries:
        st.info("Run Step 2 first.")
        return
    for pg, sm in zip(pages, summaries):
        figs = sm.get("figure_regions", [])
        st.write(f"**Page {pg['page_num']}**: {len(figs)} figure region(s)")
        if figs:
            img = _load_img(pg["png_path"])
            overlay = _draw_bboxes(img,
                                   [r["bbox"] for r in figs],
                                   [f"DI-{i+1}" for i in range(len(figs))],
                                   COLORS_BGR["orange"])
            st.image(_bgr_to_rgb(overlay),
                     caption=f"Page {pg['page_num']} — DI detections (orange)",
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Missing image detection (LLM) + HITL #1
# ══════════════════════════════════════════════════════════════════════════════

def run_step3(settings: Settings):
    """LLM-based detection of figure regions that DI missed — parallel per page."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.agents.llm_caller import call_llm, parse_json, sanitize_bbox
    from src.models.page import PageInfo

    pages = [PageInfo(**p) for p in _active_pages()]
    summaries = _s("di_summaries", [])
    client = _llm(settings)
    deploy = settings.azure_openai_deployment
    vp_detection = _s("visual_prompts", {}).get("detection")  # read before threads
    output_path = settings.output_path

    def _process_page(pg: PageInfo, sm: dict) -> tuple:
        pn = pg.page_num
        di_figs = sm.get("figure_regions", [])
        regions = [{"bbox": list(r["bbox"]), "source": "di",
                    "label": f"DI-{i+1}"} for i, r in enumerate(di_figs)]

        img = cv2.imread(pg.png_path)
        h, w = img.shape[:2]
        overlay = img.copy()
        for r in regions:
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), COLORS_BGR["orange"], 3)

        overlay_path = str(output_path / f"s3_overlay_p{pn}.png")
        Path(overlay_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(overlay_path, overlay)

        boxes_json = [{"x1": b["bbox"][0], "y1": b["bbox"][1],
                       "x2": b["bbox"][2], "y2": b["bbox"][3]} for b in regions]
        prompt = _missing_detect_prompt(boxes_json, (w, h))
        imgs = ([vp_detection] if vp_detection else []) + [overlay_path]

        raw = call_llm(client, deploy, prompt, image_paths=imgs,
                       label=f"missing_detect p{pn}")
        parsed = parse_json(raw)

        if parsed and parsed.get("status") == "missing":
            for j, item in enumerate(parsed.get("missing", [])):
                poly = item.get("polygon", [])
                if len(poly) >= 8:
                    xs = [poly[k] for k in range(0, len(poly), 2)]
                    ys = [poly[k] for k in range(1, len(poly), 2)]
                    bbox = sanitize_bbox([min(xs), min(ys), max(xs), max(ys)], w, h)
                    if bbox and not any(_iou(bbox, r["bbox"]) > 0.3 for r in regions):
                        regions.append({"bbox": bbox, "source": "llm",
                                        "label": f"LLM-{j+1}"})

        for r in [r for r in regions if r["source"] == "llm"]:
            bbox = list(r["bbox"])
            for _ in range(3):
                x1, y1, x2, y2 = bbox
                crop = img[y1:y2, x1:x2]
                cp = str(output_path / f"s3_v_p{pn}_{r['label']}.png")
                cv2.imwrite(cp, crop)
                vp_text = _region_verify_prompt(bbox, (w, h))
                vraw = call_llm(client, deploy, vp_text,
                                image_paths=[overlay_path, cp],
                                label=f"verify {r['label']}")
                vr = parse_json(vraw)
                if vr and vr.get("is_correct"):
                    break
                if vr:
                    for axis, ai in [("x1", 0), ("y1", 1), ("x2", 2), ("y2", 3)]:
                        edge = vr.get(axis, {})
                        if edge.get("direction", "none") != "none" and edge.get("corrected") is not None:
                            bbox[ai] = int(edge["corrected"])
                    bbox = sanitize_bbox(bbox, w, h) or bbox
                    r["bbox"] = bbox

        return pn, regions

    result: Dict[int, list] = {}
    done = 0
    bar = st.progress(0, "Detecting missing regions (parallel) …")

    max_workers = min(len(pages), 5)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_page, pg, sm): pg
                   for pg, sm in zip(pages, summaries)}
        for future in as_completed(futures):
            pn, regions = future.result()
            result[pn] = regions
            done += 1
            bar.progress(done / len(pages), f"Done {done}/{len(pages)} pages")

    _ss("regions_by_page", result)
    _ss("regions_confirmed", False)
    _ss("step", 3)
    _ss("current_view", 2)
    bar.empty()


def _missing_detect_prompt(boxes: list, size: Tuple[int, int]) -> str:
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["complete", "missing", "no-images"]},
            "missing": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "polygon": {"type": "array", "items": {"type": "number"}, "minItems": 8},
                        "note": {"type": "string"},
                    },
                    "required": ["polygon"],
                },
            },
        },
        "required": ["status", "missing"],
    }
    return (
        "Find MISSING image regions in this electrical engineering drawing.\n\n"
        "Electrical single-line diagrams contain: symbols (transformers, breakers, relays, motors), "
        "connection lines, hierarchical power flow.\n"
        "EXCLUDE: data tables, title blocks, notes, page headers.\n\n"
        "ORANGE boxes = already captured regions.\n"
        f"Existing boxes: {json.dumps(boxes)}\n"
        f"Image size: {json.dumps(size)}\n\n"
        "RULES:\n"
        "1. Include COMPLETE circuit diagrams top-to-bottom\n"
        "2. Do NOT overlap with or encompass ORANGE regions\n"
        "3. Each polygon = [x1,y1, x2,y1, x2,y2, x1,y2]\n"
        "4. If all regions captured, return status='complete'\n\n"
        f"Return JSON only:\n{json.dumps(schema, indent=2)}"
    )


def _region_verify_prompt(bbox: list, size: Tuple[int, int]) -> str:
    schema = {
        "type": "object",
        "properties": {
            "is_correct": {"type": "boolean"},
            "x1": {"type": "object", "properties": {"direction": {"type": "string"}, "corrected": {"type": "integer"}}},
            "y1": {"type": "object", "properties": {"direction": {"type": "string"}, "corrected": {"type": "integer"}}},
            "x2": {"type": "object", "properties": {"direction": {"type": "string"}, "corrected": {"type": "integer"}}},
            "y2": {"type": "object", "properties": {"direction": {"type": "string"}, "corrected": {"type": "integer"}}},
        },
        "required": ["is_correct"],
    }
    x1, y1, x2, y2 = bbox
    w, h = size
    return (
        f"Verify the region [x1={x1}, y1={y1}, x2={x2}, y2={y2}] correctly captures "
        f"a circuit diagram region.\nImage size: {w}×{h}\n"
        "Image 1: full page overlay. Image 2: cropped region.\n"
        "Analyze each edge: direction=none/left/right/up/down.\n"
        f"Return JSON only:\n{json.dumps(schema, indent=2)}"
    )


def render_step3():
    """Step 3 results + HITL #1: region confirmation."""
    st.subheader("Step 3 — Missing Image Detection  ✋ HITL: Confirm Regions")
    regions_by_page = _s("regions_by_page", {})
    pages = _active_pages()
    if not regions_by_page:
        st.info("Run Step 3 first.")
        return

    confirmed = _s("regions_confirmed", False)

    for pg in pages:
        pn = pg["page_num"]
        regions = regions_by_page.get(pn, regions_by_page.get(str(pn), []))
        di_n = sum(1 for r in regions if r["source"] == "di")
        llm_n = sum(1 for r in regions if r["source"] == "llm")
        st.write(f"**Page {pn}**: {di_n} DI + {llm_n} LLM = {len(regions)} total figure regions")

        # overlay
        img = _load_img(pg["png_path"])
        for r in regions:
            c = COLORS_BGR["orange"] if r["source"] == "di" else COLORS_BGR["green"]
            img = _draw_bboxes(img, [r["bbox"]], [f"{r['label']} ({r['source']})"], c)
        st.image(_bgr_to_rgb(img), caption=f"Page {pn} — all figure regions", use_container_width=True)

        if not confirmed:
            st.markdown("#### ✏️ Edit Figure Regions")
            for i, r in enumerate(regions):
                with st.expander(f"{r['label']} ({r['source']}) — {r['bbox']}", expanded=False):
                    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 0.5])
                    nx1 = c1.number_input("x1", value=r["bbox"][0], key=f"s3_{pn}_{i}_x1", step=1)
                    ny1 = c2.number_input("y1", value=r["bbox"][1], key=f"s3_{pn}_{i}_y1", step=1)
                    nx2 = c3.number_input("x2", value=r["bbox"][2], key=f"s3_{pn}_{i}_x2", step=1)
                    ny2 = c4.number_input("y2", value=r["bbox"][3], key=f"s3_{pn}_{i}_y2", step=1)
                    if c5.button("🗑️", key=f"s3d_{pn}_{i}"):
                        regions.pop(i)
                        regions_by_page[pn] = regions
                        _ss("regions_by_page", regions_by_page)
                        st.rerun()
                    nb = [int(nx1), int(ny1), int(nx2), int(ny2)]
                    if nb != r["bbox"]:
                        r["bbox"] = nb
                        regions_by_page[pn] = regions
                        _ss("regions_by_page", regions_by_page)

            with st.expander("➕ Add Region", expanded=False):
                ac = st.columns(4)
                ax1 = ac[0].number_input("x1", value=0, key=f"s3a_{pn}_x1", step=1)
                ay1 = ac[1].number_input("y1", value=0, key=f"s3a_{pn}_y1", step=1)
                ax2 = ac[2].number_input("x2", value=100, key=f"s3a_{pn}_x2", step=1)
                ay2 = ac[3].number_input("y2", value=100, key=f"s3a_{pn}_y2", step=1)
                if st.button("Add", key=f"s3ab_{pn}"):
                    regions.append({"bbox": [int(ax1), int(ay1), int(ax2), int(ay2)],
                                    "source": "human", "label": f"HUMAN-{len(regions)+1}"})
                    regions_by_page[pn] = regions
                    _ss("regions_by_page", regions_by_page)
                    st.rerun()

    if not confirmed:
        st.markdown("---")
        if st.button("✅ Confirm Regions → Proceed to Step 4", type="primary",
                      use_container_width=True):
            _ss("regions_confirmed", True)
            st.rerun()
    else:
        st.success("✅ Regions confirmed. These figure crops feed into panel name extraction.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Panel name extraction + bbox matching + HITL #2
# ══════════════════════════════════════════════════════════════════════════════

def run_step4(settings: Settings):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.agents.llm_caller import call_llm, parse_json
    from src.agents.name_extractor import (
        tile_page, extract_names_from_tile,
        dedup_panel_names, hallucination_guard,
    )
    from src.agents.bbox_matcher import (
        rule_match_panel_name, resolve_conflicts, llm_match_unresolved,
    )
    from src.models.page import PageInfo
    from src.models.panel import BboxMatch

    pages = [PageInfo(**p) for p in _active_pages()]
    di_lines_by_page = _s("di_lines_by_page", {})
    client = _llm(settings)
    deploy = settings.azure_openai_deployment

    def _process_page(pg: PageInfo) -> tuple:
        pn = pg.page_num
        tiles = tile_page(pg.png_path)
        candidates: List[str] = []
        for ti, (tile_img, _tb) in enumerate(tiles):
            tile_names = extract_names_from_tile(tile_img, client, deploy, ti)
            candidates.extend(tile_names)

        deduped = dedup_panel_names(candidates, client, deploy, pg.png_path)
        di_lines = di_lines_by_page.get(pn, di_lines_by_page.get(str(pn), []))
        verified = hallucination_guard(deduped, di_lines)

        all_matches = {}
        for name in verified:
            all_matches[name] = rule_match_panel_name(name, di_lines)

        resolved = resolve_conflicts(all_matches)
        unresolved = [n for n, m in resolved.items() if m is None]
        if unresolved:
            llm_res = llm_match_unresolved(unresolved, di_lines, pg.png_path, client, deploy)
            resolved.update(llm_res)

        matches = {
            name: (m.model_dump() if m else None)
            for name, m in resolved.items()
        }
        return pn, verified, matches

    names_by_page: Dict[int, List[str]] = {}
    matches_by_page: Dict[int, dict] = {}
    done = 0
    bar = st.progress(0, "Extracting panel names (parallel) …")

    max_workers = min(len(pages), 5)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_page, pg): pg for pg in pages}
        for future in as_completed(futures):
            pn, names, matches = future.result()
            names_by_page[pn] = names
            matches_by_page[pn] = matches
            done += 1
            bar.progress(done / len(pages), f"Done {done}/{len(pages)} pages")

    _ss("names_by_page", names_by_page)
    _ss("matches_by_page", matches_by_page)
    _ss("names_confirmed", False)
    _ss("step", 4)
    _ss("current_view", 3)
    bar.empty()


def render_step4():
    """Step 4 results + HITL #2: panel name confirmation."""
    st.subheader("Step 4 — Panel Names  ✋ HITL: Confirm Names & Label Locations")
    names_by_page = _s("names_by_page", {})
    matches_by_page = _s("matches_by_page", {})
    pages = _active_pages()
    if not names_by_page:
        st.info("Run Step 4 first.")
        return

    confirmed = _s("names_confirmed", False)

    for pg in pages:
        pn = pg["page_num"]
        names = names_by_page.get(pn, names_by_page.get(str(pn), []))
        matches = matches_by_page.get(pn, matches_by_page.get(str(pn), {}))

        st.write(f"**Page {pn}**: {len(names)} panel name(s) detected")

        # Overlay — blue boxes around panel name text-labels
        img = _load_img(pg["png_path"])
        m_bboxes, m_labels, unmatched = [], [], []
        for name in names:
            m = matches.get(name)
            if m and m.get("bbox"):
                m_bboxes.append(list(m["bbox"]))
                m_labels.append(name)
            else:
                unmatched.append(name)
        if m_bboxes:
            overlay = _draw_bboxes(img, m_bboxes, m_labels, COLORS_BGR["blue"], 2)
        else:
            overlay = img.copy()
        st.image(_bgr_to_rgb(overlay),
                 caption=f"Page {pn} — panel name text-labels (blue = matched bbox of the name text)",
                 use_container_width=True)
        if unmatched:
            st.warning(f"Unmatched names (no DI text bbox found): {', '.join(unmatched)}")

        if not confirmed:
            st.markdown("#### ✏️ Edit Panel Names & Label Locations")
            for i, name in enumerate(list(names)):
                m = matches.get(name)
                bbox = m.get("bbox") if m else None
                method = m.get("method", "—") if m else "—"
                conf = m.get("confidence", 0) if m else 0

                with st.expander(
                    f"📌 **{name}** — method={method}, conf={conf:.2f}"
                    + (f", bbox={bbox}" if bbox else ", ⚠️ no bbox"),
                    expanded=False,
                ):
                    # Show context crop around the name label
                    if bbox:
                        full = _load_img(pg["png_path"])
                        pad = 80
                        cy1 = max(0, bbox[1] - pad)
                        cx1 = max(0, bbox[0] - pad)
                        cy2 = min(full.shape[0], bbox[3] + pad)
                        cx2 = min(full.shape[1], bbox[2] + pad)
                        crop = full[cy1:cy2, cx1:cx2]
                        dx, dy = cx1, cy1
                        cv2.rectangle(crop,
                                      (bbox[0] - dx, bbox[1] - dy),
                                      (bbox[2] - dx, bbox[3] - dy),
                                      COLORS_BGR["blue"], 2)
                        st.image(_bgr_to_rgb(crop), caption=f"Context around '{name}'", width=400)

                    new_name = st.text_input("Panel Name", value=name, key=f"n4_{pn}_{i}")
                    if new_name != name:
                        idx_n = names.index(name)
                        names[idx_n] = new_name
                        if name in matches:
                            matches[new_name] = matches.pop(name)
                        names_by_page[pn] = names
                        matches_by_page[pn] = matches
                        _ss("names_by_page", names_by_page)
                        _ss("matches_by_page", matches_by_page)

                    if bbox:
                        st.caption("Name-label text bbox (this is the TEXT location, not the panel area):")
                        bc = st.columns(4)
                        bx1 = bc[0].number_input("x1", value=bbox[0], key=f"nb4_{pn}_{i}_x1", step=1)
                        by1 = bc[1].number_input("y1", value=bbox[1], key=f"nb4_{pn}_{i}_y1", step=1)
                        bx2 = bc[2].number_input("x2", value=bbox[2], key=f"nb4_{pn}_{i}_x2", step=1)
                        by2 = bc[3].number_input("y2", value=bbox[3], key=f"nb4_{pn}_{i}_y2", step=1)
                        nb = [int(bx1), int(by1), int(bx2), int(by2)]
                        cur_name = names[i] if i < len(names) else name
                        if nb != list(bbox):
                            matches[cur_name]["bbox"] = tuple(nb)
                            matches_by_page[pn] = matches
                            _ss("matches_by_page", matches_by_page)

                    if st.button(f"🗑️ Remove", key=f"nd4_{pn}_{i}"):
                        rm = names.pop(i)
                        matches.pop(rm, None)
                        names_by_page[pn] = names
                        matches_by_page[pn] = matches
                        _ss("names_by_page", names_by_page)
                        _ss("matches_by_page", matches_by_page)
                        st.rerun()

            with st.expander("➕ Add Panel Name", expanded=False):
                nn = st.text_input("New panel name", key=f"nn4_{pn}")
                if st.button("Add", key=f"na4_{pn}") and nn:
                    names.append(nn)
                    matches[nn] = None
                    names_by_page[pn] = names
                    matches_by_page[pn] = matches
                    _ss("names_by_page", names_by_page)
                    _ss("matches_by_page", matches_by_page)
                    st.rerun()

    if not confirmed:
        st.markdown("---")
        if st.button("✅ Confirm Names → Proceed to Step 5", type="primary",
                      use_container_width=True):
            _ss("names_confirmed", True)
            st.rerun()
    else:
        st.success("✅ Panel names confirmed. Each name + its text-label bbox will be used as anchor for panel area detection.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Panel area locate + verify + bay split + HITL #3
# ══════════════════════════════════════════════════════════════════════════════

def run_step5(settings: Settings):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.agents.locate_verify import locate_and_verify_batch
    from src.agents.llm_caller import call_llm, parse_json
    from src.agents.prompts import build_bay_prompt
    from src.models.page import PageInfo

    pages = [PageInfo(**p) for p in _active_pages()]
    names_by_page = _s("names_by_page", {})
    matches_by_page = _s("matches_by_page", {})
    client = _llm(settings)
    deploy = settings.azure_openai_deployment
    vp_area = _s("visual_prompts", {}).get("area_split")  # read before threads
    guide_paths = [vp_area] if vp_area and Path(vp_area).exists() else []
    output_path = settings.output_path

    def _process_page(pg: PageInfo) -> tuple:
        pn = pg.page_num
        names = names_by_page.get(pn, names_by_page.get(str(pn), []))
        matches = matches_by_page.get(pn, matches_by_page.get(str(pn), {}))
        if not names:
            return pn, [], []

        all_name_bboxes: Dict[str, Optional[List[int]]] = {}
        for name in names:
            m = matches.get(name)
            all_name_bboxes[name] = list(m["bbox"]) if m and m.get("bbox") else None

        out_dir = str(output_path / f"locate_page{pn}")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        panel_crops = locate_and_verify_batch(
            client, deploy, pg.png_path,
            all_name_bboxes, out_dir,
            guide_image_paths=guide_paths,
            grid_size=settings.grid_size,
            max_tries=settings.verify_max_tries,
        )

        crops_out, bays_out = [], []
        full_img = cv2.imread(pg.png_path)
        for crop in panel_crops:
            name = crop.panel_name
            x1, y1, x2, y2 = crop.bbox
            crop_path = str(output_path / f"crops/page{pn}_{name.replace(' ', '_')}.png")
            Path(crop_path).parent.mkdir(parents=True, exist_ok=True)
            cropped = full_img[y1:y2, x1:x2]
            cv2.imwrite(crop_path, cropped)

            crop_dict = crop.model_dump()
            crop_dict["crop_path"] = crop_path
            crop_dict["page_num"] = pn
            crops_out.append(crop_dict)

            h_c, w_c = cropped.shape[:2]
            if w_c > 1000:
                bp = build_bay_prompt(name, w_c, h_c, settings.grid_size)
                br = call_llm(client, deploy, bp,
                              image_paths=guide_paths + [crop_path],
                              label=f"bay {name}")
                bay = parse_json(br)
                if bay:
                    bays_out.append({"panel_name": name, "page_num": pn,
                                     "n_bays": bay.get("n_bays", 1),
                                     "bay_bboxes": bay.get("bboxes", [[0, 0, w_c, h_c]])})
                else:
                    bays_out.append({"panel_name": name, "page_num": pn,
                                     "n_bays": 1, "bay_bboxes": [[0, 0, w_c, h_c]]})
            else:
                bays_out.append({"panel_name": name, "page_num": pn,
                                 "n_bays": 1, "bay_bboxes": [[0, 0, w_c, h_c]]})
        return pn, crops_out, bays_out

    all_crops: List[dict] = []
    all_bays: List[dict] = []
    done = 0
    bar = st.progress(0, "Locating panel areas (parallel batch) …")

    max_workers = min(len(pages), 5)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_page, pg): pg for pg in pages}
        for future in as_completed(futures):
            pn, crops, bays = future.result()
            all_crops.extend(crops)
            all_bays.extend(bays)
            done += 1
            bar.progress(done / len(pages), f"Done {done}/{len(pages)} pages")

    _ss("panel_crops", all_crops)
    _ss("bay_results", all_bays)
    _ss("crops_confirmed", False)
    _ss("step", 5)
    _ss("current_view", 4)
    bar.empty()


def render_step5():
    """Step 5 + HITL #3: panel area & bay confirmation."""
    st.subheader("Step 5 — Panel Areas & Bays  ✋ HITL: Confirm Areas")
    crops = _s("panel_crops", [])
    bays = _s("bay_results", [])
    pages = _active_pages()
    if not crops:
        st.info("Run Step 5 first.")
        return

    confirmed = _s("crops_confirmed", False)
    color_cycle = list(COLORS_BGR.values())
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Group by page
    by_page: Dict[int, list] = {}
    for c in crops:
        by_page.setdefault(c.get("page_num", 0), []).append(c)

    for pg in pages:
        pn = pg["page_num"]
        pc = by_page.get(pn, [])
        if not pc:
            continue

        st.write(f"**Page {pn}**: {len(pc)} panel area(s)")

        # Page-level overlay: show all panel area bboxes on the full page
        img = _load_img(pg["png_path"])
        overlay = img.copy()
        for i, c in enumerate(pc):
            col = color_cycle[i % len(color_cycle)]
            x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 3)
            (tw, th), _ = cv2.getTextSize(c["panel_name"], font, 0.6, 2)
            cv2.rectangle(overlay, (x1, max(y1 - th - 10, 0)),
                          (x1 + tw + 4, max(y1, th + 10)), col, -1)
            cv2.putText(overlay, c["panel_name"], (x1 + 2, max(y1 - 4, th + 6)),
                        font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        st.image(_bgr_to_rgb(overlay), caption=f"Page {pn} — panel areas overlay",
                 use_container_width=True)

        # Per-panel detail
        for ci, c in enumerate(pc):
            name = c["panel_name"]
            cp = c.get("crop_path")
            bay_info = next((b for b in bays
                             if b["panel_name"] == name and b.get("page_num") == pn), None)

            with st.expander(
                f"🔲 {name}  |  conf={c.get('confidence', 0):.2f}  "
                f"status={c.get('status', '?')}  "
                f"bays={bay_info['n_bays'] if bay_info else '?'}",
                expanded=True,
            ):
                col_img, col_edit = st.columns([2, 1])

                with col_img:
                    if cp and Path(cp).exists():
                        ci_img = _load_img(cp)
                        if bay_info and bay_info["n_bays"] > 1:
                            for bi, bb in enumerate(bay_info["bay_bboxes"]):
                                bx1, by1, bx2, by2 = [int(v) for v in bb]
                                cv2.rectangle(ci_img, (bx1, by1), (bx2, by2),
                                              COLORS_BGR["cyan"], 2)
                                cv2.putText(ci_img, f"Bay {bi+1}", (bx1 + 4, by1 + 20),
                                            font, 0.5, COLORS_BGR["cyan"], 1, cv2.LINE_AA)
                        st.image(_bgr_to_rgb(ci_img),
                                 caption=f"{name} — cropped panel (cyan = bay dividers)",
                                 use_container_width=True)
                    else:
                        st.warning("Crop image not found")

                with col_edit:
                    if not confirmed:
                        st.markdown("**Panel area bbox (on full page):**")
                        bbox = list(c["bbox"])
                        ec = st.columns(2)
                        nx1 = ec[0].number_input("x1", value=bbox[0], key=f"s5_{pn}_{ci}_x1", step=1)
                        ny1 = ec[1].number_input("y1", value=bbox[1], key=f"s5_{pn}_{ci}_y1", step=1)
                        nx2 = ec[0].number_input("x2", value=bbox[2], key=f"s5_{pn}_{ci}_x2", step=1)
                        ny2 = ec[1].number_input("y2", value=bbox[3], key=f"s5_{pn}_{ci}_y2", step=1)
                        nb = (int(nx1), int(ny1), int(nx2), int(ny2))
                        if list(nb) != list(bbox):
                            c["bbox"] = nb
                            # Re-save crop with new bbox
                            if cp:
                                full = cv2.imread(pg["png_path"])
                                newcrop = full[nb[1]:nb[3], nb[0]:nb[2]]
                                cv2.imwrite(cp, newcrop)
                            _ss("panel_crops", crops)

                        if bay_info and bay_info["n_bays"] > 1:
                            st.markdown("**Bay bboxes (relative to crop):**")
                            for bi, bb in enumerate(bay_info["bay_bboxes"]):
                                st.caption(f"Bay {bi+1}")
                                bc = st.columns(2)
                                bx1 = bc[0].number_input("x1", value=int(bb[0]),
                                                         key=f"b5_{pn}_{ci}_{bi}_x1", step=1)
                                by1 = bc[1].number_input("y1", value=int(bb[1]),
                                                         key=f"b5_{pn}_{ci}_{bi}_y1", step=1)
                                bx2 = bc[0].number_input("x2", value=int(bb[2]),
                                                         key=f"b5_{pn}_{ci}_{bi}_x2", step=1)
                                by2 = bc[1].number_input("y2", value=int(bb[3]),
                                                         key=f"b5_{pn}_{ci}_{bi}_y2", step=1)
                                bay_info["bay_bboxes"][bi] = [int(bx1), int(by1),
                                                              int(bx2), int(by2)]
                            _ss("bay_results", bays)

    if not confirmed:
        st.markdown("---")
        if st.button("✅ Confirm All Panels & Bays → Finalize", type="primary",
                      use_container_width=True):
            _ss("crops_confirmed", True)
            st.rerun()
    else:
        st.success("✅ All panel areas and bays confirmed!")
        _render_final()


def _render_final():
    st.subheader("📊 Final Summary")
    crops = _s("panel_crops", [])
    bays = _s("bay_results", [])
    settings = _load_settings()

    summary = {
        "panels": [
            {
                "panel_name": c["panel_name"],
                "page_num": c.get("page_num", 0),
                "bbox": list(c["bbox"]),
                "crop_path": c.get("crop_path"),
                "confidence": c.get("confidence", 0),
                "verified_by": c.get("verified_by", "unknown"),
                "bay_info": next((b for b in bays if b["panel_name"] == c["panel_name"]),
                                 {"n_bays": 1, "bay_bboxes": [list(c["bbox"])]}),
            }
            for c in crops
        ],
    }
    st.json(summary)

    sp = settings.output_path / "final_summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    st.success(f"Saved to {sp}")
    st.download_button("📥 Download JSON", json.dumps(summary, indent=2, ensure_ascii=False),
                       "panel_extraction_summary.json", "application/json")


# ══════════════════════════════════════════════════════════════════════════════
#  Visual Prompt Manager (sidebar)
# ══════════════════════════════════════════════════════════════════════════════

def _render_vp_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🖼️ Visual Prompts")
    st.sidebar.caption("Guide images sent alongside LLM prompts")

    available = []
    if VISUAL_PROMPT_DIR.exists():
        for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg"):
            available.extend(VISUAL_PROMPT_DIR.glob(ext))
    available.sort(key=lambda p: p.name)

    for label, key in [
        ("Missing Image Detection", "detection"),
        ("Panel Name Extraction", "name_extraction"),
        ("Panel Area / Bay Split", "area_split"),
    ]:
        st.sidebar.markdown(f"**{label}:**")
        opts = ["None"] + [p.name for p in available]
        choice = st.sidebar.selectbox("Image", opts, key=f"vp_{key}")
        vps = _s("visual_prompts", {})
        if choice != "None":
            vps[key] = str(VISUAL_PROMPT_DIR / choice)
            _ss("visual_prompts", vps)
            st.sidebar.image(str(VISUAL_PROMPT_DIR / choice), width=180)
        else:
            vps.pop(key, None)
            _ss("visual_prompts", vps)

    # Upload custom visual prompt
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📤 Upload Custom:**")
    up = st.sidebar.file_uploader("Image", type=["png", "jpg", "jpeg"], key="vp_up")
    if up:
        ud = _PROJECT_ROOT / "outputs" / "visual_prompts"
        ud.mkdir(parents=True, exist_ok=True)
        sp = ud / up.name
        sp.write_bytes(up.read())
        st.sidebar.success(f"Saved: {sp.name}")
        tgt = st.sidebar.selectbox("Assign to",
                                   ["detection", "name_extraction", "area_split"], key="vp_asgn")
        if st.sidebar.button("Assign", key="vp_asgn_btn"):
            vps = _s("visual_prompts", {})
            vps[tgt] = str(sp)
            _ss("visual_prompts", vps)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="CAD Panel Extraction — HITL",
                       page_icon="🔧", layout="wide",
                       initial_sidebar_state="expanded")
    _init()
    settings = _load_settings()

    # ── Sidebar ──
    st.sidebar.title("🔧 CAD Panel Extraction")

    # PDF input
    uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"], key="pdf_up")
    if uploaded and not _s("pdf_path"):
        ud = settings.output_path / "uploads"
        ud.mkdir(parents=True, exist_ok=True)
        p = ud / uploaded.name
        p.write_bytes(uploaded.read())
        _ss("pdf_path", str(p))

    test_pdfs = list(VISUAL_PROMPT_DIR.glob("*.pdf")) if VISUAL_PROMPT_DIR.exists() else []
    if test_pdfs:
        st.sidebar.markdown("**Or select test PDF:**")
        opts = ["—"] + [p.name for p in test_pdfs]
        choice = st.sidebar.selectbox("Test PDF", opts, key="test_pdf")
        if choice != "—":
            _ss("pdf_path", str(VISUAL_PROMPT_DIR / choice))

    # Progress
    step = _s("step", 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Pipeline Progress")
    for i, lbl in enumerate(STEP_LABELS):
        if i < step:
            icon = "✅"
        elif i == step:
            icon = "▶️"
        else:
            icon = "⬜"
        # Clickable only for completed or current steps
        if i <= step:
            if st.sidebar.button(f"{icon} {lbl}", key=f"nav_{i}", use_container_width=True):
                _ss("current_view", i)
                st.rerun()
        else:
            st.sidebar.markdown(f"{icon} {lbl}")

    hitl_msgs = {
        3: ("regions_confirmed", "Awaiting region confirmation (HITL #1)"),
        4: ("names_confirmed", "Awaiting name confirmation (HITL #2)"),
        5: ("crops_confirmed", "Awaiting crop confirmation (HITL #3)"),
    }
    if step in hitl_msgs:
        k, msg = hitl_msgs[step]
        if not _s(k):
            st.sidebar.warning(f"⏳ {msg}")

    _render_vp_sidebar()

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Pipeline", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k not in ("pdf_up", "test_pdf"):
                del st.session_state[k]
        st.rerun()

    # ── Main area ──
    st.title("CAD Panel Extraction Pipeline")
    st.caption(
        "PDF → DI figure detection → LLM missing region detection → "
        "Panel name extraction → Panel area locate/verify + Bay split"
    )
    st.caption(
        "HITL checkpoints: ① after figure regions ② after panel names ③ after panel areas"
    )

    pdf = _s("pdf_path")
    if not pdf:
        st.info("👈 Upload a PDF or select a test PDF from the sidebar.")
        return

    st.markdown(f"**Input:** `{Path(pdf).name}`")

    # ── Step navigation: current_view controls which panel is shown ──────────
    current_view = _s("current_view", 0)

    # Breadcrumb header
    crumbs = []
    for i, lbl in enumerate(STEP_LABELS):
        if i == current_view:
            crumbs.append(f"**▶ {lbl}**")
        elif i < step:
            crumbs.append(f"✅ {lbl}")
        else:
            crumbs.append(lbl)
    st.markdown(" › ".join(crumbs))
    st.markdown("---")

    def _next_btn(target: int, label: str, key: str):
        if st.button(f"Next → {label} ▶", key=key,
                     use_container_width=True, type="primary"):
            _ss("current_view", target)
            st.rerun()

    # ── View 0: Step 1 ──────────────────────────────────────────────────────
    if current_view == 0:
        if step == 0:
            if st.button("▶️ Run Step 1: PDF → PNG", type="primary", use_container_width=True):
                with st.spinner("Converting PDF …"):
                    run_step1(pdf, settings)
                st.rerun()
        render_step1()
        if step >= 1:
            st.markdown("---")
            n_sel = len(_active_pages())
            if n_sel == 0:
                st.warning("⚠️ 페이지를 1개 이상 선택해야 다음 단계로 진행할 수 있습니다.")
            else:
                _next_btn(1, f"Step 2: DI Detection ({n_sel} pages)", "next1")

    # ── View 1: Step 2 ──────────────────────────────────────────────────────
    elif current_view == 1:
        if step >= 1:
            if step == 1:
                if st.button("▶️ Run Step 2: DI Detection", type="primary", use_container_width=True):
                    with st.spinner("DI 2-pass detection …"):
                        run_step2(settings)
                    st.rerun()
            render_step2()
            if step >= 2:
                st.markdown("---")
                _next_btn(2, "Step 3: Missing Image Detection", "next2")
        else:
            st.info("Complete Step 1 first.")

    # ── View 2: Step 3 ──────────────────────────────────────────────────────
    elif current_view == 2:
        if step >= 2:
            if step == 2:
                if st.button("▶️ Run Step 3: Missing Image Detection", type="primary", use_container_width=True):
                    with st.spinner("Detecting missing regions …"):
                        run_step3(settings)
                    st.rerun()
            render_step3()
            if step >= 3 and _s("regions_confirmed"):
                st.markdown("---")
                _next_btn(3, "Step 4: Panel Names", "next3")
        else:
            st.info("Complete Step 2 first.")

    # ── View 3: Step 4 ──────────────────────────────────────────────────────
    elif current_view == 3:
        if step >= 3 and _s("regions_confirmed"):
            if step == 3:
                if st.button("▶️ Run Step 4: Panel Names", type="primary", use_container_width=True):
                    with st.spinner("Extracting panel names …"):
                        run_step4(settings)
                    st.rerun()
            render_step4()
            if step >= 4 and _s("names_confirmed"):
                st.markdown("---")
                _next_btn(4, "Step 5: Panel Areas + Bays", "next4")
        elif step >= 3:
            st.warning("⏳ Step 3에서 figure regions를 먼저 확인하세요 (HITL #1).")
        else:
            st.info("Complete Step 3 first.")

    # ── View 4: Step 5 ──────────────────────────────────────────────────────
    elif current_view == 4:
        if step >= 4 and _s("names_confirmed"):
            if step == 4:
                if st.button("▶️ Run Step 5: Panel Areas + Bays", type="primary", use_container_width=True):
                    with st.spinner("Locating panel areas / bays …"):
                        run_step5(settings)
                    st.rerun()
            render_step5()
        elif step >= 4:
            st.warning("⏳ Step 4에서 panel names를 먼저 확인하세요 (HITL #2).")
        else:
            st.info("Complete Step 4 first.")


if __name__ == "__main__":
    main()
