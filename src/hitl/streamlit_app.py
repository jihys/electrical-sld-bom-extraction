"""Streamlit HITL Pipeline UI — v3 DI + LLM + GBB Dark UI.

Pipeline (v3):
  Step 1: PDF → PNG
  Step 2A: DI Figure Detection  (Pass 1 + whitefill Pass 2)
  Step 2B: LLM Missing Image Detection  (detect + verify)
    HITL #1: 영역 확인
  Step 3: Panel Name Extraction
    HITL #2: 이름 확인
  Step 4: Panel Area + Bay Split
    HITL #3: 패널 영역 확인
"""
from __future__ import annotations
import json, sys, os, cv2, numpy as np, streamlit as st, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

# ── Safe traceback helper ─────────────────────────────────────────
# Streamlit replaces sys.stdout/stderr per rerun, so module-level wrappers
# are ineffective.  Use traceback.format_exc() → string instead of
# traceback.print_exc() which writes directly to the (possibly closed) fd.
def _safe_print_exc():
    """Print traceback safely; silently ignores OSError if fd is closed."""
    import traceback
    try:
        traceback.print_exc()
    except OSError:
        pass

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.config import Settings

# ── Compatibility shim: streamlit-drawable-canvas 0.9.3 uses image_to_url
# which was removed in newer Streamlit versions. Restore it via monkey-patch.
# Always returns a data URL (base64) — avoids media file manager path issues.
def _compat_image_to_url(image, width, clamp, channels, output_format, image_id, allow_emoji=False):
    import io, base64
    if not hasattr(image, "save"):
        return ""
    buf = io.BytesIO()
    fmt = output_format if output_format in ("PNG", "JPEG") else "PNG"
    image.convert("RGB").save(buf, format=fmt)
    data = buf.getvalue()
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"

import streamlit.elements.image as _st_img_mod
if not hasattr(_st_img_mod, "image_to_url"):
    _st_img_mod.image_to_url = _compat_image_to_url

# Pre-import canvas so the patch is in place before the canvas module uses it
try:
    from streamlit_drawable_canvas import st_canvas as _st_canvas_fn
    _CANVAS_AVAILABLE = True
except Exception:
    _CANVAS_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────
STEP_LABELS = [
    "1. Upload & Select Pages",
    "2. Figure Detection",
    "3. Panel Names",
    "4. Panel Areas + Bay",
    "5. BOM Extraction",
]
COLORS_BGR = {
    "ms_blue":(212,120,0), "ms_purple":(145,45,92), "ms_teal":(139,142,0),
    "ms_magenta":(200,0,200), "ms_cyan":(255,200,0), "blue":(255,0,0), "green":(0,200,0),
}
VISUAL_PROMPT_DIR = Path(__file__).resolve().parents[2] / "data"
MS_LOGO_URL      = "https://img-prod-cms-rt-microsoft-com.akamaized.net/cms/api/am/imageFileData/RE1Mu3b?ver=5c31"
BANNER_PATH      = Path(__file__).resolve().parent / "static" / "banner.png"

# ──────────────────────────────────────────────────────────────────
# GBB Light CSS  — Microsoft.com inspired
# ──────────────────────────────────────────────────────────────────
GBB_CSS = """
<style>
html,body,[data-testid="stApp"]{
  background:#f5f5f5!important;color:#1a1a1a!important;
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif!important;}
#MainMenu,footer,[data-testid="stDecoration"],
[data-testid="stStatusWidget"]{display:none!important;}
[data-testid="stHeader"]{background:transparent!important;border:none!important;
  height:auto!important;min-height:0!important;}
[data-testid="stToolbar"]{background:transparent!important;}
[data-testid="stToolbar"] > *:not(:has([data-testid="stExpandSidebarButton"])){display:none!important;}
[data-testid="stToolbar"] [data-testid="stToolbarActionButton"]{display:none!important;}
[data-testid="stExpandSidebarButton"]{z-index:9999!important;}
[data-testid="stAppViewContainer"]{padding-top:0!important;}
[data-testid="stMainBlockContainer"]{padding-top:1.5rem!important;}

/* sidebar */
[data-testid="stSidebar"]{background:#fff!important;border-right:1px solid #e0e0e0!important;
  box-shadow:2px 0 8px rgba(0,0,0,.04)!important;}
[data-testid="stSidebar"] *{color:#1a1a1a!important;}
[data-testid="stSidebar"] hr{border-color:#e0e0e0!important;}
[data-testid="stSidebar"] .stButton>button{
  background:#f5f5f5!important;border:1px solid #d1d1d1!important;
  color:#1a1a1a!important;border-radius:8px!important;transition:all .15s!important;}
[data-testid="stSidebar"] .stButton>button:hover{
  background:#e8e8e8!important;border-color:#0078D4!important;color:#005a9e!important;}

/* primary button */
.stButton>button[kind="primary"],
[data-testid="stSidebar"] .stButton>button[kind="primary"]{
  background:#0078D4!important;
  color:#fff!important;border:none!important;border-radius:8px!important;
  font-weight:600!important;transition:all .15s!important;
  box-shadow:0 2px 6px rgba(0,120,212,.18)!important;}
.stButton>button[kind="primary"]:hover,
[data-testid="stSidebar"] .stButton>button[kind="primary"]:hover{
  background:#106EBE!important;
  box-shadow:0 4px 12px rgba(0,120,212,.28)!important;transform:translateY(-1px)!important;}

/* secondary button */
.stButton>button{
  background:#fff!important;border:1px solid #d1d1d1!important;
  color:#1a1a1a!important;border-radius:8px!important;transition:all .15s!important;}
.stButton>button:hover{border-color:#0078D4!important;color:#005a9e!important;background:#f0f6ff!important;}
.stButton>button:active,.stButton>button:focus{
  background:#fff!important;border-color:#0078D4!important;
  color:#1a1a1a!important;outline:none!important;box-shadow:none!important;}
.stButton>button[kind="primary"]:active,.stButton>button[kind="primary"]:focus{
  background:#0078D4!important;
  color:#fff!important;outline:none!important;}

/* form submit button */
[data-testid="stFormSubmitButton"]>button{
  background:#fff!important;border:1px solid #0078D4!important;
  color:#0078D4!important;border-radius:8px!important;font-weight:600!important;}
[data-testid="stFormSubmitButton"]>button:hover{
  background:#0078D4!important;color:#fff!important;}
[data-testid="stFormSubmitButton"]>button:active,[data-testid="stFormSubmitButton"]>button:focus{
  background:#fff!important;color:#0078D4!important;outline:none!important;box-shadow:none!important;}

/* radio */
[data-testid="stRadio"] label{color:#333!important;}
[data-testid="stRadio"] [data-checked="true"] div{background:#0078D4!important;}

/* checkbox */
.stCheckbox [data-testid="stWidgetLabel"]{color:#333!important;}

/* progress */
[data-testid="stProgress"]>div>div{
  background:linear-gradient(90deg,#0078D4,#50e6ff)!important;border-radius:4px!important;}
[data-testid="stProgress"]{background:#e0e0e0!important;border-radius:4px!important;}

/* headers */
h1{color:#1a1a1a!important;font-size:1.8rem!important;font-weight:700!important;}
h2{color:#1a1a1a!important;font-size:1.3rem!important;font-weight:600!important;}
h3{color:#333!important;}

/* inputs */
.stTextInput>div>div>input,.stNumberInput>div>div>input{
  background:#fff!important;border:1px solid #d1d1d1!important;
  color:#1a1a1a!important;border-radius:6px!important;}
.stTextInput>div>div>input:focus,.stNumberInput>div>div>input:focus{
  border-color:#0078D4!important;box-shadow:0 0 0 2px rgba(0,120,212,.15)!important;}
.stSelectbox>div>div{
  background:#fff!important;border:1px solid #d1d1d1!important;color:#1a1a1a!important;}

/* text area */
.stTextArea textarea{
  background:#fff!important;border:1px solid #d1d1d1!important;
  color:#1a1a1a!important;border-radius:6px!important;}
.stTextArea textarea:focus{
  border-color:#0078D4!important;box-shadow:0 0 0 2px rgba(0,120,212,.15)!important;}

/* alerts */
[data-testid="stAlert"]{
  background:#fff!important;border-radius:8px!important;
  border-left:3px solid #0078D4!important;color:#1a1a1a!important;
  box-shadow:0 1px 3px rgba(0,0,0,.06)!important;}

/* expander */
[data-testid="stExpander"]{
  background:#fff!important;border:1px solid #e0e0e0!important;border-radius:8px!important;
  box-shadow:0 1px 3px rgba(0,0,0,.04)!important;}
[data-testid="stExpander"] summary{color:#1a1a1a!important;}

/* file uploader */
[data-testid="stFileUploader"]{
  background:#fff!important;border-radius:8px!important;color:#555!important;}
[data-testid="stFileUploaderDropzone"]{
  background:#fafafa!important;border:2px dashed #c8c8c8!important;
  border-radius:8px!important;}
[data-testid="stFileUploaderDropzone"] *{
  color:#555!important;background:transparent!important;}
[data-testid="stFileUploaderDropzone"] button{
  background:#fff!important;border:1px solid #d1d1d1!important;
  color:#333!important;border-radius:6px!important;}
[data-testid="stFileUploaderDropzone"] button:hover{
  border-color:#0078D4!important;color:#005a9e!important;}

/* misc */
hr{border-color:#e0e0e0!important;}
.stCaption{color:#777!important;}
code{background:#f0f0f0!important;color:#0078D4!important;border-radius:4px!important;}
.stCheckbox label{color:#333!important;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{
  background:#fff!important;border-bottom:1px solid #e0e0e0!important;}
.stTabs [data-baseweb="tab"]{color:#555!important;}
.stTabs [aria-selected="true"]{color:#0078D4!important;border-bottom-color:#0078D4!important;}

/* step badge — Microsoft pill */
.step-badge{
  display:inline-block;background:#0078D4;
  color:#fff;border-radius:20px;padding:3px 14px;font-size:.72rem;
  font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px;
  box-shadow:0 1px 3px rgba(0,120,212,.2);}
.step-done-badge{background:#107C10;
  box-shadow:0 1px 3px rgba(16,124,16,.2);}

/* elapsed time chip */
.elapsed-chip{
  display:inline-block;background:#fff;border:1px solid #d1d1d1;
  border-radius:12px;padding:2px 10px;font-size:.72rem;color:#777;
  margin-left:8px;vertical-align:middle;}
.elapsed-chip .val{color:#0078D4;font-weight:600;}

/* pipeline stepper nav */
.pipeline-stepper{padding:8px 0 4px 0;}
.ps-item{display:flex;align-items:flex-start;gap:10px;margin-bottom:0;}
.ps-left{display:flex;flex-direction:column;align-items:center;width:24px;flex-shrink:0;}
.ps-dot{
  width:24px;height:24px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:.65rem;font-weight:700;flex-shrink:0;
  transition:all .2s;}
.ps-dot.done{background:#0078D4;color:#fff;box-shadow:0 0 0 2px rgba(0,120,212,.3);}
.ps-dot.active{background:#5C2D91;color:#fff;box-shadow:0 0 6px rgba(92,45,145,.3);}
.ps-dot.pending{background:#f0f0f0;color:#aaa;border:2px solid #d1d1d1;}
.ps-line{width:2px;flex:1;min-height:14px;background:#d1d1d1;margin:3px 0;}
.ps-label{font-size:.82rem;padding:3px 0 18px 0;line-height:1.3;}
.ps-label.done{color:#0078D4;cursor:pointer;}
.ps-label.done:hover{text-decoration:underline;}
.ps-label.active{color:#5C2D91;font-weight:600;}
.ps-label.pending{color:#aaa;}

/* pill nav buttons */
[data-testid="stSidebar"] .stButton>button:not([kind="primary"]){
  padding:3px 6px!important;font-size:.72rem!important;
  border-radius:10px!important;min-height:0!important;height:28px!important;
  background:#f5f5f5!important;border:1px solid #d1d1d1!important;color:#1a1a1a!important;}
[data-testid="stSidebar"] .stButton>button:not([kind="primary"]):hover{
  border-color:#0078D4!important;color:#005a9e!important;background:#f0f6ff!important;}
[data-testid="stSidebar"] .stButton>button:not([kind="primary"]):active,
[data-testid="stSidebar"] .stButton>button:not([kind="primary"]):focus{
  background:#0078D4!important;color:#fff!important;
  border-color:#0078D4!important;outline:none!important;box-shadow:none!important;}
/* active nav (primary+disabled) — force blue even when disabled */
[data-testid="stSidebar"] .stButton>button[kind="primary"]:disabled,
[data-testid="stSidebar"] .stButton>button[kind="primary"][disabled]{
  background:#0078D4!important;
  color:#fff!important;border:2px solid #106EBE!important;
  opacity:1!important;cursor:default!important;
  box-shadow:0 0 8px rgba(0,120,212,.25)!important;font-weight:800!important;}

/* GBB corner widget */
.gbb-corner{
  position:fixed;bottom:16px;right:16px;z-index:9999;
  display:flex;flex-direction:column;align-items:flex-end;gap:6px;}
.gbb-corner .ms-logo{
  background:rgba(255,255,255,.85);border-radius:6px;padding:5px 10px;
  backdrop-filter:blur(6px);box-shadow:0 1px 4px rgba(0,0,0,.1);}
.gbb-corner .ms-logo img{height:18px;display:block;}

/* breadcrumb */
.breadcrumb{font-size:.8rem;color:#777;margin-bottom:8px;display:flex;align-items:center;flex-wrap:wrap;gap:2px;}
.breadcrumb .active{color:#5C2D91;font-weight:600;}
.breadcrumb .done{color:#0078D4;}
.breadcrumb .sep{color:#c8c8c8;margin:0 2px;}

/* card — Microsoft surface */
.gbb-card{
  background:#fff;border:1px solid #e0e0e0;border-radius:8px;
  padding:14px 18px;margin-bottom:10px;
  box-shadow:0 1px 3px rgba(0,0,0,.06);}
.gbb-card b{color:#1a1a1a;}
.gbb-card .subtitle{font-size:.78rem;color:#777;margin-top:2px;}

/* info banner */
.info-banner{
  background:#e6f2ff;border:1px solid #b3d7ff;border-radius:8px;
  padding:10px 16px;margin:8px 0;font-size:.82rem;color:#004578;}
.info-banner b{color:#0078D4;}

/* error banner */
.error-banner{
  background:#fde7e9;border:1px solid #f5b0b7;border-radius:8px;
  padding:10px 16px;margin:8px 0;font-size:.82rem;color:#6e0811;}
.error-banner b{color:#c50f1f;}

/* success banner */
.success-banner{
  background:#dff6dd;border:1px solid #9be09a;border-radius:8px;
  padding:10px 16px;margin:8px 0;font-size:.82rem;color:#0b6a0b;}
.success-banner b{color:#107C10;}

/* canvas resize handles */
canvas + div rect, .upper-canvas ~ * rect{
  fill:#0078D4!important;stroke:#fff!important;stroke-width:1!important;}
[data-testid="stDrawableCanvas"] {
  border:1px solid #d1d1d1!important;border-radius:4px!important;}

/* panel name list button */
.pname-btn{
  display:block;width:100%;padding:8px 14px;margin:3px 0;border-radius:8px;
  background:#fff;border:1px solid #d1d1d1;color:#1a1a1a;cursor:pointer;
  text-align:left;font-size:.85rem;transition:all .15s;}
.pname-btn:hover{border-color:#0078D4;color:#005a9e;background:#f0f6ff;}
.pname-btn.selected{border-color:#0078D4;background:rgba(0,120,212,.08);color:#0078D4;font-weight:600;}

/* highlight flash border on image */
@keyframes bbox-flash{
  0%,100%{opacity:1;} 50%{opacity:.3;}}
.flash-border{animation:bbox-flash .6s ease-in-out 3;}
</style>
"""

GBB_CORNER_HTML = f"""
<div class="gbb-corner">
  <div class="ms-logo">
    <a href="https://www.microsoft.com" target="_blank">
      <img src="{MS_LOGO_URL}" alt="Microsoft"/>
    </a>
  </div>
</div>
"""

# ──────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────
def _draw_bboxes(img, bboxes, labels, color=None, thickness=3):
    if color is None: color = COLORS_BGR["ms_blue"]
    out = img.copy(); font = cv2.FONT_HERSHEY_SIMPLEX
    for bbox, label in zip(bboxes, labels):
        x1,y1,x2,y2 = [int(v) for v in bbox]
        cv2.rectangle(out,(x1,y1),(x2,y2),color,thickness)
        if label:
            (tw,th),_ = cv2.getTextSize(label,font,.55,2)
            cv2.rectangle(out,(x1,max(y1-th-10,0)),(x1+tw+4,max(y1,th+10)),color,-1)
            cv2.putText(out,label,(x1+2,max(y1-4,th+6)),font,.55,(255,255,255),2,cv2.LINE_AA)
    return out

def _bgr_to_rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def _load_img(path):
    img=cv2.imread(path)
    if img is None: raise FileNotFoundError(path)
    return img
def _iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    if ix2<=ix1 or iy2<=iy1: return 0.0
    inter=(ix2-ix1)*(iy2-iy1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); ab=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+ab-inter) if (aa+ab-inter)>0 else 0.0

# ──────────────────────────────────────────────────────────────────
# Session state helpers
# ──────────────────────────────────────────────────────────────────
def _s(k,d=None): return st.session_state.get(k,d)
def _ss(k,v): st.session_state[k]=v

def _fmt_elapsed(secs):
    """Format elapsed seconds as human-readable string."""
    if secs < 60: return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    return f"{m}m {s}s"

def _elapsed_html(secs):
    """Return HTML chip showing elapsed time."""
    return f'<span class="elapsed-chip">⏱ <span class="val">{_fmt_elapsed(secs)}</span></span>'

def _init():
    defaults={
        "step":0,"current_view":0,"pdf_path":None,
        "pages":[],"selected_pages":None,
        "di_regions_by_page":{},"di_detection_done":False,
        "missing_detection_done":False,
        "regions_by_page":{},"regions_confirmed":False,
        "names_by_page":{},"matches_by_page":{},"names_confirmed":False,
        "panel_crops":[],"bay_results":[],"crops_confirmed":False,
        "bom_results":{},"bom_confirmed":False,
        "visual_prompts":{},
        "step_timings":{},  # {step_name: elapsed_secs}
        "step_errors":{},   # {step_name: error_message}
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

def _active_pages():
    pages=_s("pages",[]); sel=_s("selected_pages",None)
    if sel is None or sel=={p["page_num"] for p in pages}: return pages
    return [p for p in pages if p["page_num"] in sel]

@st.cache_resource
def _load_settings():
    env=_PROJECT_ROOT/".env"
    if env.exists():
        from dotenv import load_dotenv; load_dotenv(str(env))
    return Settings()

def _llm(settings):
    if "llm_client" not in st.session_state:
        from src.agents.llm_caller import create_llm_client
        st.session_state["llm_client"]=create_llm_client(
            settings.azure_openai_endpoint,settings.azure_openai_api_key,
            settings.azure_openai_api_version)
    return st.session_state["llm_client"]

# ══════════════════════════════════════════════════════════════════
#  STEP 1 — PDF → PNG
# ══════════════════════════════════════════════════════════════════
def _make_thumbnail(png_path, thumb_dir, max_w=300):
    """Create a low-res JPEG thumbnail for page selection UI."""
    thumb_dir = Path(thumb_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumb_dir / (Path(png_path).stem + "_thumb.jpg")
    if thumb_path.exists():
        return str(thumb_path)
    img = cv2.imread(str(png_path))
    if img is None:
        return str(png_path)
    h, w = img.shape[:2]
    scale = max_w / w
    small = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(thumb_path), small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return str(thumb_path)

def run_step1(pdf_path, settings):
    from src.tools.pdf_tools import compute_dpi_for_pdf, split_pdf_to_pages
    pages=split_pdf_to_pages(pdf_path,str(settings.output_path/"pages"),
        *compute_dpi_for_pdf(pdf_path))
    page_dicts = [p.model_dump() for p in pages]
    thumb_dir = settings.output_path / "pages" / "thumbs"
    for p in page_dicts:
        p["thumb_path"] = _make_thumbnail(p["png_path"], thumb_dir)
    _ss("pages", page_dicts); _ss("step",1); _ss("current_view",0)

def render_step1():
    st.markdown('<span class="step-badge">Step 1</span>',unsafe_allow_html=True)
    st.subheader("Upload & Select Pages")
    pages=_s("pages",[])
    if not pages: st.info("Select a PDF from the sidebar."); return
    all_pnums=sorted(p["page_num"] for p in pages)
    sel=_s("selected_pages",None)
    if sel is None: sel=set(all_pnums); _ss("selected_pages",sel)
    ver=_s("page_sel_ver",0)
    st.write(f"**Converted {len(pages)} pages.** Select pages to process:")
    cc=st.columns([1,1,4])
    if cc[0].button("All",key="psel_all",use_container_width=True):
        _ss("selected_pages",set(all_pnums)); _ss("page_sel_ver",ver+1); st.rerun()
    if cc[1].button("None",key="psel_none",use_container_width=True):
        _ss("selected_pages",set()); _ss("page_sel_ver",ver+1); st.rerun()
    cols=st.columns(min(len(pages),4)); new_sel=set()
    for i,p in enumerate(pages):
        pn=p["page_num"]; on=pn in sel
        thumb = p.get("thumb_path", p["png_path"])
        with cols[i%len(cols)]:
            bc="#0078D4" if on else "#d1d1d1"
            st.markdown(f'<div style="border:2px solid {bc};border-radius:8px;padding:3px;background:#fff;transition:border-color .2s;box-shadow:0 1px 3px rgba(0,0,0,.06)">',unsafe_allow_html=True)
            st.image(thumb,caption=f"Page {pn}",use_container_width=True)
            st.markdown("</div>",unsafe_allow_html=True)
            checked=st.checkbox(f"Page {pn}",value=on,key=f"cb_{pn}_v{ver}")
            if checked: new_sel.add(pn)
    if new_sel!=sel: _ss("selected_pages",new_sel)
    n=len(new_sel)
    if n==0: st.warning("Select at least one page.")
    else: st.caption(f"{n} / {len(pages)} pages selected")



# ══════════════════════════════════════════════════════════════════
#  STEP 2 — Figure Detection  (Phase A: DI  +  Phase B: Missing LLM)
# ══════════════════════════════════════════════════════════════════

def run_step2a(settings):
    """Phase A: DI-based figure detection (Pass 1 + whitefill Pass 2)."""
    import shutil
    from src.models.page import PageInfo

    from src.cad.di_utils import create_di_client, analyze_page_with_document_intelligence
    from src.cad.region_utils import (
        whitefill_figures, persist_figure_regions,
        PageExtractionSummary,
    )

    pages = [PageInfo(**p) for p in _active_pages()]
    di_client = create_di_client(settings.azure_di_endpoint, settings.azure_di_key)
    di_model = settings.azure_di_model_id

    output_root = settings.output_path / "di_detection"
    output_root.mkdir(parents=True, exist_ok=True)

    bar = st.progress(0, "Phase A: DI figure detection (parallel) …")
    status_container = st.container()
    page_status = {pg.page_num: "⏳ waiting" for pg in pages}
    def _update_status_a():
        lines = [f"**Page {pn}**: {s}" for pn, s in sorted(page_status.items())]
        status_container.markdown(" · ".join(lines), unsafe_allow_html=True)
    _update_status_a()
    t0 = time.time()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Prepare per-page directories first (sequential, fast)
    page_prep = {}  # pn -> (dest_img, page_dir)
    page_dpi_map = {}
    for pg in pages:
        pn = pg.page_num
        page_dpi_map[pn] = pg.dpi
        page_dir = output_root / f"page{pn}"
        page_dir.mkdir(parents=True, exist_ok=True)
        dest_img = page_dir / f"page{pn}.png"
        shutil.copy2(pg.png_path, dest_img)
        page_prep[pn] = (dest_img, page_dir)

    # Define per-page DI analysis function
    def _di_analyze_page(pg):
        pn = pg.page_num
        page_status[pn] = "🔄 analyzing"
        dest_img, page_dir = page_prep[pn]
        summary = PageExtractionSummary(page=pn, page_image=dest_img)

        # Pass 1: full DI analysis
        result1 = analyze_page_with_document_intelligence(
            di_client, di_model, dest_img,
            extract_tables=False, extract_text=False,
        )
        figures1 = getattr(result1, "figures", []) or []
        pass1_count = len(figures1)
        print(f"  [DI] page {pn} Pass 1: {pass1_count} figures")

        # Pass 2: whitefill detected figures → re-run DI
        if figures1:
            img = cv2.imread(str(dest_img))
            whitefilled, n_filled = whitefill_figures(img, figures1)
            wf_path = page_dir / f"page{pn}_whitefill.png"
            cv2.imwrite(str(wf_path), whitefilled)
            result2 = analyze_page_with_document_intelligence(
                di_client, di_model, wf_path,
                extract_tables=False, extract_text=False,
            )
            figures2 = getattr(result2, "figures", []) or []
            print(f"  [DI] page {pn} Pass 2: {len(figures2)} additional figures")
            result1.figures = list(figures1) + list(figures2)

        return pn, result1, pass1_count, summary, dest_img

    # Run DI analysis in parallel
    analysis_results = {}
    page_image_paths = []
    page_summaries = {}
    pass1_count_per_page = {}
    done = 0

    with ThreadPoolExecutor(max_workers=min(len(pages), 3)) as ex:
        futs = {ex.submit(_di_analyze_page, pg): pg for pg in pages}
        for f in as_completed(futs):
            pn, result, p1count, summary, dest_img = f.result()
            analysis_results[pn] = result
            pass1_count_per_page[pn] = p1count
            page_summaries[pn] = summary
            done += 1
            total_figs = len(getattr(result, "figures", []) or [])
            page_status[pn] = f"✅ {total_figs} figures"
            bar.progress(done / (len(pages) + 1), f"Phase A: DI {done}/{len(pages)} pages done")
            _update_status_a()

    # Reconstruct page_image_paths in page order
    for pg in pages:
        page_image_paths.append(page_prep[pg.page_num][0])

    # Persist figure crops
    bar.progress(0.9, "Saving figure crops …")
    persist_figure_regions(
        page_image_paths, analysis_results, page_summaries,
        output_root,
        pass1_count_per_page=pass1_count_per_page,
        page_dpi_map=page_dpi_map,
    )

    # Convert to regions_by_page format
    rbp = {}
    for pn in [pg.page_num for pg in pages]:
        regions = []
        images_dir = output_root / f"page{pn}" / "images"
        if images_dir.exists():
            for crop_path in sorted(images_dir.glob(f"images_{pn}_*.png")):
                if "overlay" in crop_path.stem.lower():
                    continue
                txt_path = crop_path.with_suffix(".txt")
                bbox = None
                if txt_path.exists():
                    raw = txt_path.read_text().strip().replace("\n", ",")
                    coords = [c.strip() for c in raw.split(",") if c.strip()]
                    if len(coords) >= 8:
                        xs = [float(coords[i]) for i in range(0, len(coords), 2)]
                        ys = [float(coords[i]) for i in range(1, len(coords), 2)]
                        img = cv2.imread(str(output_root / f"page{pn}" / f"page{pn}.png"))
                        h, w = img.shape[:2]
                        # Handle normalized 0-1 coords
                        if max(xs + ys) <= 1.5:
                            xs = [v * w for v in xs]
                            ys = [v * h for v in ys]
                        bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                if bbox:
                    label = crop_path.stem.split("_")[-1]
                    regions.append({
                        "bbox": bbox, "source": "di",
                        "label": f"DI-{label}",
                        "crop_path": str(crop_path),
                    })
        rbp[pn] = regions
        total = sum(len(v) for v in rbp.values())
        print(f"  [DI] page {pn}: {len(regions)} regions")

    _ss("di_regions_by_page", rbp)
    _ss("regions_by_page", rbp)
    _ss("di_detection_done", True)
    _ss("missing_detection_done", False)
    _ss("regions_confirmed", False)
    _ss("step", 2)
    _ss("current_view", 1)
    elapsed = time.time() - t0
    timings = _s("step_timings", {}); timings["step2a"] = elapsed; _ss("step_timings", timings)
    bar.empty()
    st.success(f"DI detected {total} figure regions across {len(pages)} pages. ({_fmt_elapsed(elapsed)})")


def run_step2b(settings):
    """Phase B: LLM-based missing image detection using run_page_pipeline."""
    from src.models.page import PageInfo

    from src.cad.pipeline import run_page_pipeline, postprocess_crops, load_all_crops

    pages = [PageInfo(**p) for p in _active_pages()]
    client = _llm(settings)
    deploy = settings.azure_openai_deployment

    output_root = settings.output_path / "di_detection"
    crops_root = settings.output_path / "missing_detection"
    crops_temp = settings.output_path / "missing_det_temp"
    crops_root.mkdir(parents=True, exist_ok=True)
    crops_temp.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    bar = st.progress(0, "Phase B: LLM missing‐image detection (parallel) …")
    status_container_b = st.container()
    page_status_b = {pg.page_num: "⏳ waiting" for pg in pages}
    def _update_status_b():
        lines = [f"**Page {pn}**: {s}" for pn, s in sorted(page_status_b.items())]
        status_container_b.markdown(" · ".join(lines), unsafe_allow_html=True)
    _update_status_b()
    t0 = time.time()
    reports = []
    page_dpi_map = {pg.page_num: pg.dpi for pg in pages}

    def _run_page(pg):
        pn = pg.page_num
        try:
            result = run_page_pipeline(
                pn,
                client=client,
                deployment=deploy,
                output_root=output_root,
                crops_root=crops_root,
                crops_temp_root=crops_temp,
                max_iterations=3,
                detection_reasoning_effort="none",
                verification_reasoning_effort="low",
                page_dpi_map=page_dpi_map,
            )
            return result
        except Exception as e:
            _safe_print_exc()
            return None

    done = 0
    with ThreadPoolExecutor(max_workers=min(len(pages), 5)) as ex:
        futs = {ex.submit(_run_page, pg): pg for pg in pages}
        for f in as_completed(futs):
            pg = futs[f]
            pn = pg.page_num
            result = f.result()
            if result:
                reports.append(result)
                n_v = result.get("n_verified", 0)
                page_status_b[pn] = f"✅ +{n_v} new"
            else:
                page_status_b[pn] = "❌ error"
            done += 1
            bar.progress(done / len(pages), f"Phase B: {done}/{len(pages)} pages done")
            _update_status_b()

    # Post-process: clip to content bbox, drop duplicates
    if reports:
        bar.progress(0.9, "Post-processing crops …")
        postprocess_crops(reports, output_root, crops_root)

    # Rebuild regions_by_page from all final crops
    di_rbp = _s("di_regions_by_page", {})
    rbp = {}
    for pg in pages:
        pn = pg.page_num
        regions = []

        # DI regions (from phase A)
        for r in di_rbp.get(pn, []):
            regions.append(r)

        # New LLM-detected regions from crops_root
        page_crop_dir = crops_root / f"page{pn}"
        if page_crop_dir.exists():
            base_img = cv2.imread(str(output_root / f"page{pn}" / f"page{pn}.png"))
            if base_img is not None:
                h_img, w_img = base_img.shape[:2]
                for crop_path in sorted(page_crop_dir.glob(f"image_{pn}_*.png")):
                    if "overlay" in crop_path.stem.lower():
                        continue
                    txt_path = crop_path.with_suffix(".txt")
                    bbox = None
                    if txt_path.exists():
                        try:
                            raw = txt_path.read_text().strip().replace("\n", ",")
                            coords = [float(v) for v in raw.split(",") if v.strip()]
                            if len(coords) >= 8:
                                xs = [coords[i] for i in range(0, len(coords), 2)]
                                ys = [coords[i] for i in range(1, len(coords), 2)]
                                if max(xs + ys) <= 1.5:
                                    xs = [v * w_img for v in xs]
                                    ys = [v * h_img for v in ys]
                                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                        except Exception:
                            pass
                    if bbox:
                        # Skip if overlaps with existing DI region
                        if any(_iou(bbox, r["bbox"]) > 0.5 for r in regions):
                            continue
                        label = crop_path.stem.split("_")[-1]
                        regions.append({
                            "bbox": bbox, "source": "llm",
                            "label": f"LLM-{label}",
                            "crop_path": str(crop_path),
                        })

        rbp[pn] = regions
        n_di = sum(1 for r in regions if r["source"] == "di")
        n_llm = sum(1 for r in regions if r["source"] == "llm")
        print(f"  page {pn}: {n_di} DI + {n_llm} LLM = {len(regions)} total")

    _ss("regions_by_page", rbp)
    _ss("missing_detection_done", True)
    _ss("regions_confirmed", False)
    elapsed = time.time() - t0
    timings = _s("step_timings", {}); timings["step2b"] = elapsed; _ss("step_timings", timings)
    bar.empty()
    total = sum(len(v) for v in rbp.values())
    n_new = total - sum(len(v) for v in di_rbp.values())
    st.success(f"LLM found {n_new} additional regions. Total: {total} across {len(pages)} pages. ({_fmt_elapsed(elapsed)})")


def render_step2():
    st_canvas = _st_canvas_fn if _CANVAS_AVAILABLE else None
    _CANVAS = _CANVAS_AVAILABLE
    st.markdown('<span class="step-badge">Step 2</span>',unsafe_allow_html=True)
    st.subheader("Figure Detection — HITL: Review Regions")

    di_done = _s("di_detection_done", False)
    missing_done = _s("missing_detection_done", False)

    # Phase indicator
    if di_done and missing_done:
        t2a = _s("step_timings", {}).get("step2a", 0)
        t2b = _s("step_timings", {}).get("step2b", 0)
        timing_html = ""
        if t2a or t2b:
            timing_html = f" {_elapsed_html(t2a + t2b)}"
        st.markdown(f'<div class="info-banner">✔ Phase A (DI) + Phase B (LLM Missing) complete{timing_html}</div>', unsafe_allow_html=True)
    elif di_done:
        st.caption("✔ Phase A (DI) complete · Phase B (LLM Missing) pending")
    else:
        st.info("Run Phase A first.")
        return

    rbp=_s("regions_by_page",{}); pages=_active_pages()
    if not rbp: st.info("No regions detected."); return
    confirmed=_s("regions_confirmed",False)
    DISP_W=700

    # Color scheme: DI=orange, LLM=cyan
    COLOR_DI = (0, 165, 255)       # orange BGR
    COLOR_LLM = (255, 200, 0)      # cyan BGR

    for pg in pages:
        pn=pg["page_num"]
        regions=list(rbp.get(pn,rbp.get(str(pn),[])))
        n_di = sum(1 for r in regions if r.get("source") == "di")
        n_llm = sum(1 for r in regions if r.get("source") == "llm")
        badge = f"{n_di} DI"
        if n_llm: badge += f" + {n_llm} LLM"
        st.markdown(f'<div class="gbb-card"><b>Page {pn}</b> — {len(regions)} regions ({badge})</div>',unsafe_allow_html=True)
        img=_load_img(pg["png_path"]); h_orig,w_orig=img.shape[:2]
        scale=DISP_W/w_orig; disp_h=int(h_orig*scale)
        _canvas_ok = False
        if not confirmed and _CANVAS:
          try:
            # ── Canvas editor inside @st.fragment for scoped reruns ──
            _s2_state_key = f"_s2_canvas_json_{pn}"
            _s2_bg_key = f"_s2_canvas_bg_{pn}"

            # Pre-compute background once and cache in session_state
            if _s2_bg_key not in st.session_state:
                img_resized = cv2.resize(img, (DISP_W, disp_h))
                st.session_state[_s2_bg_key] = Image.fromarray(_bgr_to_rgb(img_resized))
            bg_pil = st.session_state[_s2_bg_key]

            # Build initial objects from regions (used only on first load)
            init_objs = []
            for r in regions:
                color = "#f97316" if r.get("source") == "di" else "#00c8ff"
                init_objs.append({
                    "type": "rect",
                    "left": round(r["bbox"][0] * scale),
                    "top": round(r["bbox"][1] * scale),
                    "width": round((r["bbox"][2] - r["bbox"][0]) * scale),
                    "height": round((r["bbox"][3] - r["bbox"][1]) * scale),
                    "fill": "rgba(0,120,212,0.12)", "stroke": color, "strokeWidth": 2,
                })
            default_drawing = {"version": "4.4.0", "objects": init_objs}

            # Factory function to capture loop variables in a new scope,
            # avoiding the Python closure-in-loop bug where all fragments
            # would share the last iteration's values on fragment rerun.
            def _make_s2_fragment(_pn, _dd, _scale, _w_orig, _h_orig, _disp_h, _regions, _rbp, _bg_key, _DISP_W, _png_path):
                @st.fragment
                def _canvas_fragment():
                    # Re-read bg image from session_state inside fragment
                    # to ensure it's available on fragment reruns
                    _bg_pil = st.session_state.get(_bg_key)
                    if _bg_pil is None:
                        _img = cv2.imread(_png_path)
                        _img_resized = cv2.resize(_img, (_DISP_W, _disp_h))
                        _bg_pil = Image.fromarray(_bgr_to_rgb(_img_resized))
                        st.session_state[_bg_key] = _bg_pil
                    st_canvas_fn = _st_canvas_fn
                    mode = st.radio("Edit Mode", ["Add Region", "Move / Resize"],
                                    key=f"s2_fmode_{_pn}", horizontal=True)
                    draw_mode = "rect" if mode == "Add Region" else "transform"
                    st.caption("🟠 DI  ·  🔵 LLM  |  Add: drag rectangle  ·  Move/Resize: click → drag corners")

                    canvas_result = st_canvas_fn(
                        fill_color="rgba(0,120,212,0.12)", stroke_width=2, stroke_color="#00c8ff",
                        background_image=_bg_pil,
                        initial_drawing=_dd,
                        update_streamlit=True, height=_disp_h, width=_DISP_W,
                        drawing_mode=draw_mode, key=f"canvas_s2_{_pn}",
                    )

                    rects = []
                    if canvas_result.json_data is not None:
                        rects = [o for o in canvas_result.json_data.get("objects", []) if o.get("type") == "rect"]
                    if rects:
                        st.caption(f"📐 {len(rects)} region(s) on canvas")

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✅ Apply Changes", key=f"s2_apply_{_pn}", type="primary", use_container_width=True):
                            if canvas_result.json_data is not None:
                                objs = [o for o in canvas_result.json_data.get("objects", []) if o.get("type") == "rect"]
                                new_regions = []
                                for j, obj in enumerate(objs):
                                    lft = obj.get("left", 0); top_v = obj.get("top", 0)
                                    w = obj.get("width", 0) * obj.get("scaleX", 1)
                                    h = obj.get("height", 0) * obj.get("scaleY", 1)
                                    x1 = max(0, int(lft / _scale)); y1 = max(0, int(top_v / _scale))
                                    x2 = min(_w_orig, int((lft + w) / _scale)); y2 = min(_h_orig, int((top_v + h) / _scale))
                                    if x2 > x1 and y2 > y1:
                                        lbl = _regions[j]["label"] if j < len(_regions) else f"H-{len(new_regions)+1}"
                                        src = _regions[j].get("source", "human") if j < len(_regions) else "human"
                                        new_regions.append({"bbox": [x1, y1, x2, y2], "source": src, "label": lbl})
                                _rbp[_pn] = new_regions; _ss("regions_by_page", _rbp)
                                st.session_state.pop(_bg_key, None)
                                st.rerun()
                    with col_b:
                        if st.button("🗑 Delete Last", key=f"s2_dellast_{_pn}", use_container_width=True):
                            if _regions:
                                _regions.pop(); _rbp[_pn] = _regions; _ss("regions_by_page", _rbp)
                                st.session_state.pop(_bg_key, None)
                                st.rerun()
                return _canvas_fragment

            _make_s2_fragment(pn, default_drawing, scale, w_orig, h_orig, disp_h, regions, rbp, _s2_bg_key, DISP_W, pg["png_path"])()
            _canvas_ok = True
          except Exception as _e2:
            import traceback; traceback.print_exc()
            st.error(f"Canvas error: {_e2}")
            _canvas_ok = False
        if not _canvas_ok and not confirmed:
            # Static image fallback when canvas unavailable
            ov=_load_img(pg["png_path"])
            for r in regions:
                c = COLOR_DI if r.get("source") == "di" else COLOR_LLM
                ov=_draw_bboxes(ov,[r["bbox"]],[r["label"]],c)
            st.image(_bgr_to_rgb(ov),caption=f"Page {pn} — 🟠 DI  🔵 LLM",use_container_width=True)
    if not confirmed:
        st.markdown("---")
        if st.button("Confirm Regions → Step 3",type="primary",use_container_width=True):
            _ss("regions_confirmed",True); st.rerun()
    else:
        st.success("Regions confirmed.")

# ══════════════════════════════════════════════════════════════════
#  STEP 3 — Panel Name Extraction + HITL #2  (구 Step 4)
# ══════════════════════════════════════════════════════════════════
def run_step3(settings):
    """Panel name extraction pipeline:
    1) Tiled LLM extraction → dedup
    2) DI OCR on each crop image → di_lines (full-page coords)
    3) Rule-based matching (Pass 1-6) + serial-align + conflict resolve
    4) LLM fallback for remaining unmatched names
    """
    from src.models.page import PageInfo
    from src.tools.di_tools import create_di_client, analyze_page

    from src.cad.panel_name_extractor import extract_panel_names_from_tile, dedup_panel_names_for_page
    from src.cad.panel_bbox_matcher import run_bbox_matching
    from src.cad.figure_subcropping import discover_crops, tile_crops

    pages = [PageInfo(**p) for p in _active_pages()]
    client = _llm(settings)
    deploy = settings.azure_openai_deployment

    # ── DI client ─────────────────────────────────────────────────────────
    if "di_client" not in st.session_state:
        st.session_state["di_client"] = create_di_client(
            endpoint=settings.azure_di_endpoint,
            key=settings.azure_di_key,
        )
    di_client = st.session_state["di_client"]
    di_model = settings.azure_di_model_id

    from concurrent.futures import ThreadPoolExecutor, as_completed

    bar = st.progress(0, "Step 3: Extracting panel names …")
    t0 = time.time()
    total_steps = len(pages) * 2 + 1  # Phase1 + Phase2 + matching

    # ── Discover crops & compute DPI-aware smart tiles ────────────────────
    rbp = _s("regions_by_page", {})
    figure_crops_dir = settings.output_path / "missing_detection"
    pages_dir = settings.output_path / "di_detection"

    # Build page_images from full-page PNGs
    page_images = {}
    for pg in pages:
        img = cv2.imread(pg.png_path)
        if img is not None:
            page_images[pg.page_num] = img

    # Build per-page DPI map
    page_dpi_map = {pg.page_num: pg.dpi for pg in pages}

    # Try smart tiling via discover_crops + tile_crops (DPI-aware)
    tiles_by_page = {}
    try:
        crops_list, _ = discover_crops(
            figure_crops_dir=figure_crops_dir,
            pages_dir=pages_dir,
            test_pages=[pg.page_num for pg in pages],
        )
        if crops_list:
            pdf_path = _s("pdf_path", "")
            tiles_by_page, _ = tile_crops(
                crops=crops_list,
                page_images=page_images,
                pdf_path=Path(pdf_path) if pdf_path else Path("."),
                overlap_px=200,
                per_page_dpi=page_dpi_map,
                split_threshold_inch=7.0,
            )
    except Exception as e:
        print(f"[Step3] Smart tiling failed, falling back to simple tiling: {e}")
        _safe_print_exc()

    # Fallback: simple fixed-width tiling for pages without tiles
    for pg in pages:
        pn = pg.page_num
        if pn not in tiles_by_page or not tiles_by_page[pn]:
            img = page_images.get(pn)
            if img is None:
                continue
            h, w = img.shape[:2]
            tile_width, overlap = 2000, 400
            tile_bboxes = []
            x = 0
            while x < w:
                x2 = min(x + tile_width, w)
                tile_bboxes.append((x, 0, x2, h))
                if x2 >= w:
                    break
                x += tile_width - overlap
            tiles_by_page[pn] = tile_bboxes

    # ── Phase 1: LLM tiled name extraction (parallel per page) ────────────
    bar.progress(0, "Phase 1: LLM panel name extraction (parallel) …")
    names_by_page = {}

    def _extract_page_names(pg):
        pn = pg.page_num
        full_img = page_images.get(pn)
        if full_img is None:
            return pn, [], []
        tile_bboxes = tiles_by_page.get(pn, [])
        tile_cands = []
        for ti, (tx1, ty1, tx2, ty2) in enumerate(tile_bboxes):
            tile_img = full_img[ty1:ty2, tx1:tx2]
            names, _elapsed = extract_panel_names_from_tile(
                tile_img, client, deploy,
            )
            if names:
                tile_cands.extend(names)
            print(f"  page {pn} tile{ti}: {len(names) if names else 0} candidates: {(names or [])[:5]}")
        verified, _elapsed = dedup_panel_names_for_page(
            tile_cands, full_img, client, deploy,
        )
        print(f"  page {pn}: {len(tile_cands)} candidates → {len(verified)} verified: {verified[:5]}")
        return pn, verified, tile_bboxes

    done = 0
    with ThreadPoolExecutor(max_workers=min(len(pages), 5)) as ex:
        futs = {ex.submit(_extract_page_names, pg): pg for pg in pages}
        for f in as_completed(futs):
            pn, verified, tile_bboxes = f.result()
            names_by_page[pn] = verified
            tiles_by_page[pn] = tile_bboxes
            done += 1
            bar.progress(done / total_steps, f"Phase 1: {done}/{len(pages)} pages extracted")

    # ── Phase 2: DI OCR on crop images (parallel) ──────────────────────────
    bar.progress(done / total_steps, "Phase 2: DI OCR extraction (parallel) …")
    di_lines_by_page = {}

    def _di_ocr_page(pg):
        pn = pg.page_num
        try:
            di_result = analyze_page(di_client, di_model, pg.png_path)
            lines = di_result.get("lines", [])
            for line in lines:
                bb = line.get("bbox", [0, 0, 0, 0])
                cx = (bb[0] + bb[2]) / 2
                for ti, (tx1, ty1, tx2, ty2) in enumerate(tiles_by_page.get(pn, [])):
                    if tx1 <= cx <= tx2:
                        line["tile_idx"] = ti
                        break
                else:
                    line["tile_idx"] = 0
            print(f"  page {pn}: DI found {len(lines)} lines")
            return pn, lines
        except Exception as e:
            print(f"  [WARN] DI failed for page {pn}: {e}")
            return pn, []

    with ThreadPoolExecutor(max_workers=min(len(pages), 3)) as ex:
        futs = {ex.submit(_di_ocr_page, pg): pg for pg in pages}
        for f in as_completed(futs):
            pn, lines = f.result()
            di_lines_by_page[pn] = lines
            done += 1
            bar.progress(done / total_steps, f"Phase 2: DI OCR {done - len(pages)}/{len(pages)} pages")

    # ── Phase 3: Matching (rule-based + LLM fallback) via run_bbox_matching ───
    bar.progress(done / total_steps, "Phase 3: BBox matching (rule + LLM fallback) …")

    result = {"by_page": {int(k): v for k, v in names_by_page.items()}}
    panel_bbox_matches, result = run_bbox_matching(
        result=result,
        di_lines_by_page=di_lines_by_page,
        page_images=page_images,
        tiles_by_page=tiles_by_page,
        llm_client=client,
        deployment=deploy,
    )

    # ── Convert to session state format ───────────────────────────────────
    nbp = {}
    mbp = {}
    for pn in result["by_page"]:
        nbp[pn] = result["by_page"][pn]
        matches = {}
        pn_matches = panel_bbox_matches.get(pn, {})
        for name in nbp[pn]:
            hits = pn_matches.get(name, [])
            if hits:
                # Take best hit (first one after conflict resolution)
                best = hits[0]
                matches[name] = {
                    "bbox": tuple(best["bbox"]),
                    "confidence": best.get("confidence", 0.8),
                    "method": best.get("method", "rule"),
                }
            else:
                matches[name] = None
        mbp[pn] = matches

    _ss("names_by_page", nbp)
    _ss("matches_by_page", mbp)
    _ss("names_confirmed", False)
    _ss("step", 3)
    _ss("current_view", 2)
    elapsed = time.time() - t0
    timings = _s("step_timings", {}); timings["step3"] = elapsed; _ss("step_timings", timings)
    bar.empty()

def render_step3():
    st.markdown('<span class="step-badge">Step 3</span>',unsafe_allow_html=True)
    st.subheader("Panel Name Extraction — HITL: Review Names")
    nbp=_s("names_by_page",{}); mbp=_s("matches_by_page",{}); pages=_active_pages()
    if not nbp: st.info("Run Step 3 first."); return
    confirmed=_s("names_confirmed",False)
    # Show timing
    t3 = _s("step_timings", {}).get("step3")
    if t3:
        total_names = sum(len(v) for v in nbp.values())
        st.markdown(f'<div class="info-banner">Found <b>{total_names} panel names</b> across {len(nbp)} pages {_elapsed_html(t3)}</div>', unsafe_allow_html=True)
    # Determine which name is highlighted (if any)
    hl_name=_s("s3_hl_name"); hl_page=_s("s3_hl_page")
    for pg in pages:
        pn=pg["page_num"]
        names=list(nbp.get(pn,nbp.get(str(pn),[]))); matches=dict(mbp.get(pn,mbp.get(str(pn),{})))
        st.markdown(f'<div class="gbb-card"><b>Page {pn}</b> — {len(names)} panel names</div>',unsafe_allow_html=True)
        img=_load_img(pg["png_path"])
        # --- Tabs: Image | Panel Names | Edit ---
        if not confirmed:
            tab_img,tab_names,tab_edit=st.tabs(["🖼 Image","📋 Panel Names","✏️ Edit"])
        else:
            tab_img,tab_names=st.tabs(["🖼 Image","📋 Panel Names"]); tab_edit=None
        with tab_img:
            # Draw all bboxes; highlight selected name
            overlay=img.copy(); font=cv2.FONT_HERSHEY_SIMPLEX
            for name in names:
                m=matches.get(name)
                if m and m.get("bbox"):
                    bbox=list(m["bbox"]); x1,y1,x2,y2=[int(v) for v in bbox]
                    if name==hl_name and pn==hl_page:
                        # Highlighted: bright cyan thick border + filled label
                        cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,255),6)
                        (tw,th),_=cv2.getTextSize(name,font,.8,2)
                        cv2.rectangle(overlay,(x1,max(y1-th-14,0)),(x1+tw+8,max(y1,th+14)),(0,255,255),-1)
                        cv2.putText(overlay,name,(x1+4,max(y1-6,th+8)),font,.8,(0,0,0),2,cv2.LINE_AA)
                    else:
                        cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,0,0),2)
                        (tw,th),_=cv2.getTextSize(name,font,.55,2)
                        cv2.rectangle(overlay,(x1,max(y1-th-10,0)),(x1+tw+4,max(y1,th+10)),(255,0,0),-1)
                        cv2.putText(overlay,name,(x1+2,max(y1-4,th+6)),font,.55,(255,255,255),2,cv2.LINE_AA)
            cap=f"Page {pn} — Panel name locations"
            if hl_name and pn==hl_page: cap+=f"  ★ {hl_name}"
            st.image(_bgr_to_rgb(overlay),caption=cap,use_container_width=True)
        with tab_names:
            # Clickable name list
            for i,name in enumerate(names):
                m=matches.get(name); conf=m.get("confidence",0) if m else 0
                method=m.get("method","—") if m else "—"
                has_bbox="✓" if (m and m.get("bbox")) else "✗"
                is_sel=(name==hl_name and pn==hl_page)
                btn_label=f"{'★ ' if is_sel else ''}{name}   ({method} {conf:.0%} {has_bbox})"
                if st.button(btn_label,key=f"pn3_sel_{pn}_{i}",use_container_width=True,
                             type="primary" if is_sel else "secondary"):
                    if is_sel:
                        _ss("s3_hl_name",None); _ss("s3_hl_page",None)
                    else:
                        _ss("s3_hl_name",name); _ss("s3_hl_page",pn)
                    st.rerun()
        if tab_edit is not None:
            with tab_edit:
                DISP_W_S3 = 700
                h_orig, w_orig = img.shape[:2]
                scale_s3 = DISP_W_S3 / w_orig
                disp_h_s3 = int(h_orig * scale_s3)
                HEX_COLORS_S3 = ["#0078D4","#5C2D91","#008B8B","#B4009E","#00BCF2","#7FBA00","#3A96DD","#E81123","#FF8C00","#10893E"]
                _canvas_s3_ok = False
                if _CANVAS_AVAILABLE:
                  try:
                    init_objs_s3 = []
                    name_order_s3 = []
                    for idx_s3, nm_s3 in enumerate(names):
                        m_s3 = matches.get(nm_s3)
                        if m_s3 and m_s3.get("bbox"):
                            bx = m_s3["bbox"]
                            init_objs_s3.append({
                                "type": "rect",
                                "left": round(int(bx[0]) * scale_s3),
                                "top": round(int(bx[1]) * scale_s3),
                                "width": round((int(bx[2]) - int(bx[0])) * scale_s3),
                                "height": round((int(bx[3]) - int(bx[1])) * scale_s3),
                                "fill": "rgba(0,0,0,0)",
                                "stroke": HEX_COLORS_S3[idx_s3 % len(HEX_COLORS_S3)],
                                "strokeWidth": 3,
                            })
                            name_order_s3.append(nm_s3)
                    default_s3 = {"version": "4.4.0", "objects": init_objs_s3}
                    _s3_bg_key = f"_s3_canvas_bg_{pn}"
                    if _s3_bg_key not in st.session_state:
                        img_r_s3 = cv2.resize(img, (DISP_W_S3, disp_h_s3))
                        st.session_state[_s3_bg_key] = Image.fromarray(_bgr_to_rgb(img_r_s3))
                    bg_pil_s3 = st.session_state[_s3_bg_key]

                    def _make_s3_fragment(_pn, _dd, _scale, _w_orig, _h_orig, _disp_h, _names, _name_order, _matches, _nbp, _mbp, _bg_key, _DISP_W):
                        @st.fragment
                        def _canvas_s3_fragment():
                            _bg_pil = st.session_state.get(_bg_key)
                            if _bg_pil is None:
                                _bg_pil = Image.fromarray(_bgr_to_rgb(cv2.resize(_load_img(pg["png_path"]), (_DISP_W, _disp_h))))
                                st.session_state[_bg_key] = _bg_pil

                            # Legend: color → name mapping
                            legend_items = []
                            _hc = ["#0078D4","#5C2D91","#008B8B","#B4009E","#00BCF2","#7FBA00","#3A96DD","#E81123","#FF8C00","#10893E"]
                            for li, nm in enumerate(_name_order):
                                c = _hc[li % len(_hc)]
                                legend_items.append(f'<span style="color:{c};font-weight:bold">■</span> {nm}')
                            if legend_items:
                                st.markdown(" &nbsp; ".join(legend_items), unsafe_allow_html=True)

                            st.info("Drag rectangles to move/resize panel name boxes → click **Apply Changes**")
                            st_canvas_fn = _st_canvas_fn
                            canvas_result = st_canvas_fn(
                                fill_color="rgba(0,0,0,0)", stroke_width=3, stroke_color="#0078D4",
                                background_image=_bg_pil,
                                initial_drawing=_dd,
                                update_streamlit=True, height=_disp_h, width=_DISP_W,
                                drawing_mode="transform", key=f"canvas_s3_{_pn}",
                            )
                            if canvas_result.json_data is not None:
                                objs = [o for o in canvas_result.json_data.get("objects", []) if o.get("type") == "rect"]
                                st.caption(f"📐 {len(objs)} bbox(es) on canvas")
                            if st.button("✅ Apply Changes", key=f"s3_apply_{_pn}", type="primary", use_container_width=True):
                                if canvas_result.json_data is not None:
                                    objs = [o for o in canvas_result.json_data.get("objects", []) if o.get("type") == "rect"]
                                    for ci, obj in enumerate(objs[:len(_name_order)]):
                                        lft = obj.get("left", 0); top_v = obj.get("top", 0)
                                        w = obj.get("width", 0) * obj.get("scaleX", 1)
                                        h = obj.get("height", 0) * obj.get("scaleY", 1)
                                        x1 = max(0, int(lft / _scale)); y1 = max(0, int(top_v / _scale))
                                        x2 = min(_w_orig, int((lft + w) / _scale)); y2 = min(_h_orig, int((top_v + h) / _scale))
                                        nm = _name_order[ci]
                                        if nm in _matches and _matches[nm]:
                                            _matches[nm]["bbox"] = (x1, y1, x2, y2)
                                    _mbp[_pn] = _matches
                                    _ss("matches_by_page", _mbp)
                                    st.session_state.pop(_bg_key, None)
                                    st.rerun()
                        return _canvas_s3_fragment

                    _make_s3_fragment(pn, default_s3, scale_s3, w_orig, h_orig, disp_h_s3,
                                     names, name_order_s3, matches, nbp, mbp, _s3_bg_key, DISP_W_S3)()
                    _canvas_s3_ok = True
                  except Exception as _e3:
                    import traceback; traceback.print_exc()
                    st.error(f"Canvas error: {_e3}")
                    _canvas_s3_ok = False
                # Fallback: name list with text editing (always show for add/delete)
                st.markdown("---")
                st.markdown("**Panel Names**")
                for i,name in enumerate(list(names)):
                    m=matches.get(name); bbox=m.get("bbox") if m else None
                    conf=m.get("confidence",0) if m else 0; method=m.get("method","—") if m else "—"
                    col_nm, col_del = st.columns([5, 1])
                    with col_nm:
                        nn=st.text_input(f"{name}", value=name, key=f"n3_{pn}_{i}", label_visibility="collapsed")
                        if nn!=name:
                            ni=names.index(name); names[ni]=nn
                            if name in matches: matches[nn]=matches.pop(name)
                            nbp[pn]=names; mbp[pn]=matches
                            _ss("names_by_page",nbp); _ss("matches_by_page",mbp)
                    with col_del:
                        if st.button("🗑", key=f"nd3_{pn}_{i}"):
                            rm=names.pop(i); matches.pop(rm,None)
                            nbp[pn]=names; mbp[pn]=matches
                            _ss("names_by_page",nbp); _ss("matches_by_page",mbp); st.rerun()
                with st.expander("➕ Add Name",expanded=False):
                    nn=st.text_input("New name",key=f"nn3_{pn}")
                    if st.button("Add",key=f"na3_{pn}") and nn:
                        names.append(nn); matches[nn]=None
                        nbp[pn]=names; mbp[pn]=matches
                        _ss("names_by_page",nbp); _ss("matches_by_page",mbp); st.rerun()
    if not confirmed:
        st.markdown("---")
        if st.button("Confirm Names → Step 4",type="primary",use_container_width=True):
            _ss("names_confirmed",True); st.rerun()
    else:
        st.success("Names confirmed.")

# ══════════════════════════════════════════════════════════════════
#  STEP 4 — Panel Area + Bay Split + HITL #3  (구 Step 5)
# ══════════════════════════════════════════════════════════════════
def run_step4(settings):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.models.page import PageInfo

    from src.cad.panel_pipeline import process_all_panels_batch, apply_exclude_regions
    from src.cad.bay_pipeline import process_one_bay_split
    from src.cad.panel_utils import safe_name

    pages = [PageInfo(**p) for p in _active_pages()]
    nbp = _s("names_by_page", {})
    mbp = _s("matches_by_page", {})
    rbp = _s("regions_by_page", {})
    client = _llm(settings)
    deploy = settings.azure_openai_deployment

    # Load guide images as np arrays
    vp_raw = _s("visual_prompts", {}).get("area_split")
    if isinstance(vp_raw, list):
        guide_paths = [p for p in vp_raw if p and Path(p).exists()]
    elif vp_raw and Path(vp_raw).exists():
        guide_paths = [vp_raw]
    else:
        guide_paths = []
    guide_images = [cv2.imread(str(p)) for p in guide_paths if cv2.imread(str(p)) is not None]

    out = settings.output_path
    t0 = time.time()
    grid_size = getattr(settings, "grid_size", 120)
    verify_max_tries = getattr(settings, "verify_max_tries", 3)

    def _proc(pg):
        pn = pg.page_num
        names = nbp.get(pn, nbp.get(str(pn), []))
        matches = mbp.get(pn, mbp.get(str(pn), {}))
        regions = rbp.get(pn, rbp.get(str(pn), []))

        if not names:
            print(f"[Step4] Page {pn}: SKIP — no names found")
            return pn, [], []

        full_img = cv2.imread(pg.png_path)
        if full_img is None:
            print(f"[Step4] Page {pn}: cannot load image")
            return pn, [], []

        h_full, w_full = full_img.shape[:2]

        # Build name->bbox map from matches (full-page coords)
        nb_map = {}
        for n, m in matches.items():
            if n in names:
                nb_map[n] = list(m["bbox"]) if m and m.get("bbox") else None
        for n in names:
            if n not in nb_map:
                nb_map[n] = None

        # Build crop items from DI/LLM regions
        crop_items = []
        if regions:
            for ri, r in enumerate(regions):
                bbox = r["bbox"]
                crop_path = r.get("crop_path")
                if not crop_path or not Path(crop_path).exists():
                    # Create crop from full page
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    crop_img = full_img[y1:y2, x1:x2]
                    crop_dir = out / f"di_detection" / f"page{pn}" / "images"
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = str(crop_dir / f"crop_{pn}_{ri+1}.png")
                    cv2.imwrite(crop_path, crop_img)
                crop_items.append({
                    "crop_idx": ri + 1,
                    "path": Path(crop_path),
                    "bbox_offset": [int(v) for v in bbox[:2]],  # x1, y1 offset
                })
        else:
            # Fallback: use full page as single crop
            crop_items.append({
                "crop_idx": 1,
                "path": Path(pg.png_path),
                "bbox_offset": [0, 0],
            })

        print(f"[Step4] Page {pn}: {len(names)} names, {len(crop_items)} crops")

        all_results = []
        all_bays = []

        for crop_item in crop_items:
            crop_idx = crop_item["crop_idx"]
            crop_img = cv2.imread(str(crop_item["path"]))
            if crop_img is None:
                continue

            h_crop, w_crop = crop_img.shape[:2]
            ox, oy = crop_item["bbox_offset"]

            # Convert full-page name bboxes to crop-local coordinates
            local_bboxes = {}
            for n, b in nb_map.items():
                if b is not None:
                    lx1, ly1, lx2, ly2 = b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy
                    # Check if name bbox is within this crop
                    cx = (lx1 + lx2) / 2
                    cy = (ly1 + ly2) / 2
                    if 0 <= cx <= w_crop and 0 <= cy <= h_crop:
                        local_bboxes[n] = [max(0, lx1), max(0, ly1),
                                           min(w_crop, lx2), min(h_crop, ly2)]
                    # else: name is in a different crop region
                else:
                    # Include names without bbox only if this is the single crop
                    if len(crop_items) == 1:
                        local_bboxes[n] = None

            if not local_bboxes:
                continue

            crop_names = [n for n in names if n in local_bboxes]
            if not crop_names:
                continue

            page_out_dir = out / f"locate_page{pn}"
            page_out_dir.mkdir(parents=True, exist_ok=True)
            page_debug_dir = out / f"locate_page{pn}" / "_debug"
            page_debug_dir.mkdir(parents=True, exist_ok=True)

            try:
                results = process_all_panels_batch(
                    client, deploy, guide_images,
                    page_num=pn, crop_idx=crop_idx,
                    names=crop_names, img=crop_img,
                    all_name_bboxes=local_bboxes,
                    page_out_dir=page_out_dir,
                    page_debug_dir=page_debug_dir,
                    grid_size=grid_size,
                    verify_max_tries=verify_max_tries,
                    verify_reasoning_effort="none",
                )
            except Exception as e:
                _safe_print_exc()
                print(f"[Step4] Page {pn} crop{crop_idx}: ERROR — {e}")
                results = []

            # Convert results: crop-local bbox → full-page bbox, save panel crops
            for r in results:
                panel_name = r["panel_name"]
                local_bbox = r["bbox"]
                exclude_regions = r.get("exclude_regions", [])

                # Convert to full-page coordinates
                full_bbox = [
                    local_bbox[0] + ox, local_bbox[1] + oy,
                    local_bbox[2] + ox, local_bbox[3] + oy,
                ]

                # Save panel crop (use the one from process_all_panels_batch,
                # which already has exclude_regions white-filled)
                panel_file = r.get("file")
                s = safe_name(panel_name)
                cp = str(out / f"crops/p{pn}_{s}.png")
                Path(cp).parent.mkdir(parents=True, exist_ok=True)
                if panel_file and Path(panel_file).exists():
                    import shutil
                    shutil.copy2(panel_file, cp)
                else:
                    # Fallback: crop from full image
                    x1, y1, x2, y2 = [int(v) for v in full_bbox]
                    panel_crop = full_img[y1:y2, x1:x2]
                    if exclude_regions:
                        panel_crop = apply_exclude_regions(panel_crop, full_bbox, [
                            [er[0] + ox, er[1] + oy, er[2] + ox, er[3] + oy]
                            for er in exclude_regions
                        ])
                    cv2.imwrite(cp, panel_crop)

                all_results.append({
                    "panel_name": panel_name,
                    "bbox": tuple(full_bbox),
                    "crop_path": cp,
                    "page_num": pn,
                    "confidence": 0.95 if r.get("bbox") else 0.5,
                    "status": "verified",
                    "verified_by": "llm_verify_batch",
                    "verify_attempts": verify_max_tries,
                    "exclude_regions": exclude_regions,
                    "crop_w": r.get("crop_w", w_crop),
                })

                # Bay splitting
                hc, wc = cv2.imread(cp).shape[:2] if Path(cp).exists() else (0, 0)
                bay_result = process_one_bay_split(
                    client, deploy, guide_images,
                    panel_name=panel_name,
                    panel_img_path=Path(cp),
                    out_dir=out / f"bay_page{pn}",
                    grid_size=grid_size,
                    min_width=600,
                    crop_w=r.get("crop_w", w_crop),
                    reference_width=2000,
                )
                if bay_result and bay_result.get("bays"):
                    bay_bboxes = [b["bbox"] for b in bay_result["bays"]]
                    all_bays.append({
                        "panel_name": panel_name, "page_num": pn,
                        "n_bays": len(bay_bboxes), "bay_bboxes": bay_bboxes,
                    })
                else:
                    all_bays.append({
                        "panel_name": panel_name, "page_num": pn,
                        "n_bays": 1, "bay_bboxes": [[0, 0, wc, hc]],
                    })

        return pn, all_results, all_bays

    ac, ab, done = [], [], 0
    bar = st.progress(0, "Detecting panel areas …")
    with ThreadPoolExecutor(max_workers=min(len(pages), 5)) as ex:
        futs = {ex.submit(_proc, pg): pg for pg in pages}
        for f in as_completed(futs):
            pn, co, bo = f.result()
            ac.extend(co)
            ab.extend(bo)
            done += 1
            bar.progress(done / len(pages), f"Done: {done}/{len(pages)} pages")
    _ss("panel_crops", ac)
    _ss("bay_results", ab)
    _ss("crops_confirmed", False)
    _ss("step", 4)
    _ss("current_view", 3)
    elapsed = time.time() - t0
    timings = _s("step_timings", {})
    timings["step4"] = elapsed
    _ss("step_timings", timings)
    bar.empty()

def render_step4():
    st_canvas = _st_canvas_fn if _CANVAS_AVAILABLE else None
    _CANVAS = _CANVAS_AVAILABLE
    st.markdown('<span class="step-badge">Step 4</span>',unsafe_allow_html=True)
    st.subheader("Panel Areas & Bay — HITL: Review Regions")
    crops=_s("panel_crops",[]); bays=_s("bay_results",[]); pages=_active_pages()
    if not crops: st.info("Run Step 4 first."); return
    confirmed=_s("crops_confirmed",False)
    # Show timing
    t4 = _s("step_timings", {}).get("step4")
    if t4:
        st.markdown(f'<div class="info-banner">Located <b>{len(crops)} panels</b> {_elapsed_html(t4)}</div>', unsafe_allow_html=True)
    cc=list(COLORS_BGR.values()); font=cv2.FONT_HERSHEY_SIMPLEX
    HEX_COLORS=["#0078D4","#5C2D91","#008B8B","#B4009E","#00BCF2","#7FBA00","#3A96DD"]
    byp={}
    for c in crops: byp.setdefault(c.get("page_num",0),[]).append(c)
    for pg in pages:
        pn=pg["page_num"]; pc=byp.get(pn,[])
        if not pc: continue
        DISP_W=700; img=_load_img(pg["png_path"]); h_orig,w_orig=img.shape[:2]
        scale=DISP_W/w_orig; disp_h=int(h_orig*scale)
        st.markdown(f'<div class="gbb-card"><b>Page {pn}</b> — {len(pc)} panels</div>',unsafe_allow_html=True)
        if not confirmed:
            tab_view, tab_resize, tab_crops = st.tabs(["🖼 Overview", "✋ Resize / Move", "📦 Panel Crops"])
        else:
            tab_view, tab_crops = st.tabs(["🖼 Overview", "📦 Panel Crops"]); tab_resize = None
        with tab_view:
            ov=img.copy()
            for i,c in enumerate(pc):
                col=cc[i%len(cc)]; x1,y1,x2,y2=[int(v) for v in c["bbox"]]
                cv2.rectangle(ov,(x1,y1),(x2,y2),col,3)
                (tw,th),_=cv2.getTextSize(c["panel_name"],font,.6,2)
                cv2.rectangle(ov,(x1,max(y1-th-10,0)),(x1+tw+4,max(y1,th+10)),col,-1)
                cv2.putText(ov,c["panel_name"],(x1+2,max(y1-4,th+6)),font,.6,(255,255,255),2,cv2.LINE_AA)
            st.image(_bgr_to_rgb(ov),caption=f"Page {pn} — Panel regions",use_container_width=True)
        if tab_resize is not None:
            with tab_resize:
                _canvas_ok = False
                if _CANVAS:
                  try:
                    init_objs=[{
                        "type":"rect","left":round(int(c["bbox"][0])*scale),"top":round(int(c["bbox"][1])*scale),
                        "width":round((int(c["bbox"][2])-int(c["bbox"][0]))*scale),
                        "height":round((int(c["bbox"][3])-int(c["bbox"][1]))*scale),
                        "fill":"rgba(0,0,0,0)","stroke":HEX_COLORS[i%len(HEX_COLORS)],"strokeWidth":3,
                    } for i,c in enumerate(pc)]
                    default_s4 = {"version":"4.4.0","objects":init_objs}
                    _s4_state_key = f"_s4_canvas_json_{pn}"
                    _s4_bg_key = f"_s4_canvas_bg_{pn}"

                    if _s4_bg_key not in st.session_state:
                        img_resized = cv2.resize(img, (DISP_W, disp_h))
                        st.session_state[_s4_bg_key] = Image.fromarray(_bgr_to_rgb(img_resized))
                    bg_pil_s4 = st.session_state[_s4_bg_key]

                    # Factory function to capture loop variables in a new scope
                    def _make_s4_fragment(_pn, _dd, _scale, _w_orig, _h_orig, _disp_h, _pc, _crops, _pg, _bg_key, _DISP_W):
                        @st.fragment
                        def _canvas_s4_fragment():
                            # Re-read bg image from session_state inside fragment
                            # to ensure it's available on fragment reruns
                            _bg_pil = st.session_state.get(_bg_key)
                            if _bg_pil is None:
                                _img = cv2.imread(_pg["png_path"])
                                _img_resized = cv2.resize(_img, (_DISP_W, _disp_h))
                                _bg_pil = Image.fromarray(_bgr_to_rgb(_img_resized))
                                st.session_state[_bg_key] = _bg_pil
                            st_canvas_fn = _st_canvas_fn
                            st.info("Click a rectangle → drag handles to resize/move → click **Apply Regions** below")
                            canvas_result = st_canvas_fn(
                                fill_color="rgba(0,0,0,0)", stroke_width=3, stroke_color="#0078D4",
                                background_image=_bg_pil,
                                initial_drawing=_dd,
                                update_streamlit=True, height=_disp_h, width=_DISP_W,
                                drawing_mode="transform", key=f"canvas_s4_{_pn}",
                            )
                            if canvas_result.json_data is not None:
                                objs = [o for o in canvas_result.json_data.get("objects", []) if o.get("type") == "rect"]
                                st.caption(f"📐 {len(objs)} panel(s) on canvas")
                            st.markdown("")
                            if st.button("✅ Apply Regions", key=f"s4_apply_{_pn}", type="primary", use_container_width=True):
                                if canvas_result.json_data is not None:
                                    objs = [o for o in canvas_result.json_data.get("objects", []) if o.get("type") == "rect"]
                                    full_img = cv2.imread(_pg["png_path"])
                                    for ci2, obj in enumerate(objs[:len(_pc)]):
                                        lft = obj.get("left", 0); top_v = obj.get("top", 0)
                                        w = obj.get("width", 0) * obj.get("scaleX", 1)
                                        h = obj.get("height", 0) * obj.get("scaleY", 1)
                                        x1 = max(0, int(lft / _scale)); y1 = max(0, int(top_v / _scale))
                                        x2 = min(_w_orig, int((lft + w) / _scale)); y2 = min(_h_orig, int((top_v + h) / _scale))
                                        if x2 > x1 and y2 > y1:
                                            _pc[ci2]["bbox"] = (x1, y1, x2, y2)
                                            cp = _pc[ci2].get("crop_path")
                                            if cp: cv2.imwrite(cp, full_img[y1:y2, x1:x2])
                                    _ss("panel_crops", _crops)
                                    st.session_state.pop(_bg_key, None)
                                    st.rerun()
                        return _canvas_s4_fragment

                    _make_s4_fragment(pn, default_s4, scale, w_orig, h_orig, disp_h, pc, crops, pg, _s4_bg_key, DISP_W)()
                    _canvas_ok = True
                  except Exception as _e4:
                    import traceback; traceback.print_exc()
                    st.error(f"Canvas error: {_e4}")
                    _canvas_ok = False
                if not _canvas_ok:
                    st.warning("Canvas not available. Use manual bbox editing below.")
                    for ci,c in enumerate(pc):
                        n=c["panel_name"]; bbox=list(c["bbox"])
                        with st.expander(f"**{n}** bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})",expanded=False):
                            bc=st.columns(4)
                            bx1=bc[0].number_input("x1",value=int(bbox[0]),key=f"s4b_{pn}_{ci}_x1",step=1)
                            by1=bc[1].number_input("y1",value=int(bbox[1]),key=f"s4b_{pn}_{ci}_y1",step=1)
                            bx2=bc[2].number_input("x2",value=int(bbox[2]),key=f"s4b_{pn}_{ci}_x2",step=1)
                            by2=bc[3].number_input("y2",value=int(bbox[3]),key=f"s4b_{pn}_{ci}_y2",step=1)
                            nb=[int(bx1),int(by1),int(bx2),int(by2)]
                            if nb!=bbox:
                                c["bbox"]=tuple(nb)
                                full_img=cv2.imread(pg["png_path"])
                                cp=c.get("crop_path")
                                if cp:
                                    cv2.imwrite(cp,full_img[nb[1]:nb[3],nb[0]:nb[2]])
                                _ss("panel_crops",crops); st.rerun()
        with tab_crops:
            for ci,c in enumerate(pc):
                n=c["panel_name"]; cp=c.get("crop_path")
                bi=next((b for b in bays if b["panel_name"]==n and b.get("page_num")==pn),None)
                with st.expander(f"{n}  conf={c.get('confidence',0):.2f}  bays={bi['n_bays'] if bi else '?'}",expanded=False):
                    if cp and Path(cp).exists():
                        ci_img=_load_img(cp)
                        if bi and bi["n_bays"]>1:
                            for bii,bb in enumerate(bi["bay_bboxes"]):
                                bx1,by1,bx2,by2=[int(v) for v in bb]
                                cv2.rectangle(ci_img,(bx1,by1),(bx2,by2),COLORS_BGR["ms_teal"],2)
                                cv2.putText(ci_img,f"Bay {bii+1}",(bx1+4,by1+20),font,.5,COLORS_BGR["ms_teal"],1,cv2.LINE_AA)
                        st.image(_bgr_to_rgb(ci_img),use_container_width=True)
                    else: st.warning("No crop available")
    if not confirmed:
        st.markdown("---")
        if st.button("Confirm → Step 5",type="primary",use_container_width=True):
            _ss("crops_confirmed",True); st.rerun()
    else:
        st.success("Panel areas confirmed.")

def _render_final():
    st.markdown('<span class="step-badge step-done-badge">Done</span>',unsafe_allow_html=True)
    st.subheader("Final Results")
    crops=_s("panel_crops",[]); bays=_s("bay_results",[]); bom=_s("bom_results",{}); settings=_load_settings()
    # Total timing summary
    timings = _s("step_timings", {})
    if timings:
        total_time = sum(timings.values())
        parts = " + ".join(f"{k.replace('step','S')}={_fmt_elapsed(v)}" for k, v in sorted(timings.items()))
        st.markdown(
            f'<div class="success-banner">'
            f'<b>Pipeline complete!</b> Total processing: <b>{_fmt_elapsed(total_time)}</b><br>'
            f'<span style="font-size:.75rem;color:#0b6a0b;">{parts}</span>'
            f'</div>', unsafe_allow_html=True)
    summary={"panels":[{"panel_name":c["panel_name"],"page_num":c.get("page_num",0),
        "bbox":list(c["bbox"]),"confidence":c.get("confidence",0),
        "bay_info":next((b for b in bays if b["panel_name"]==c["panel_name"]),
            {"n_bays":1,"bay_bboxes":[list(c["bbox"])]}),
        "bom":bom.get(c["panel_name"],"")} for c in crops]}
    st.json(summary)
    sp=settings.output_path/"final_summary.json"
    sp.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    st.success(f"Saved: {sp}")
    st.download_button("Download JSON",json.dumps(summary,indent=2,ensure_ascii=False),
        "panel_extraction_summary.json","application/json")

# ══════════════════════════════════════════════════════════════════
#  STEP 5 — BOM Extraction
# ══════════════════════════════════════════════════════════════════
_BOM_SYSTEM_PROMPT = """너는 전장(전기설비) 분야의 입찰/견적 업무를 위한 단선도(Single Line Diagram, SLD) 분석 전문가이다.

역할:
- 단선도 이미지 1장을 한 번의 분석으로 완료한다.
- 도면 전체 기준 기기 총합(Global Count)과 최종 판넬 기준 BOM을 동시에 산출한다.
- Reflection 단계 없이 결과를 확정한다.

==============================
[핵심 분석 원칙]
==============================

1. 도면에 명시적으로 표시된 텍스트와 기기 심볼을 근거로 판단한다.
2. 도면의 모든 영역을 전수 조사한다. 기기 누락은 절대 허용되지 않는다.
3. 수량은 개별 기기 기준의 정수로 표현한다.
4. "SET" 표현은 사용하지 않는다.
5. 빈 셀, 예약 공간(RESERVE), 예비 표기는 제외한다.
6. 텍스트 라벨이 없더라도 기기 심볼이 명확하게 존재하면 해당 기기를 반드시 식별 하고 수량에 포함한다.

==============================
[판넬 규칙]
==============================

1. 이미지당 판넬은 1개이다.
2. 판넬명은 좌측 상단 육각형 내부에 표기되어 있다.
3. 도면에 존재하는 모든 기기는 해당 판넬에 속한다.
4. Section이 존재하더라도 최종 집계는 판넬 기준으로 통합한다.

==============================
[수량 계산 규칙]
==============================

1. 도면 전체에서 각 기기(VCB, CT, ZCT, PT, PTT, CTT, 계전기, SA/LA, TB(단자대), CH(케이블헤드), 퓨즈, MCB/MCCB 등)를 전수 카운트한다.
2. 심볼 내부 표기(CTx3, ZCTx1, 3EA 등)는 기본 수량(N)으로 해석한다.
3. 반복 구조는 실제 그려진 개수를 직접 카운트한다.
4. 동일 사양 장비는 각각 독립적으로 카운트한다.
5. 텍스트 라벨이 없는 경우에도 동일한 전기 심볼이 반복되면 각각 독립된 기기로 판단하여 카운트한다.
6. 최종 판넬 BOM 합계는 Global Count와 반드시 일치해야 한다.
7. 불일치 발생 시 도면을 재검토하여 수정 후 확정한다.

==============================
[누락 방지 강제 규칙: 소형 보조기기]
==============================

1. CTT/PTT/시험단자대/단자대(TB)/케이블헤드(CH)는 "작게 그려져도" 기기이며 반드시 수량에 포함한다.
2. 메인 기기 옆에 붙은 보조기기(예: CT 옆 CTT, PT 옆 PTT)는 메인 기기 수량에 흡수하지 말고 별도 품목으로 카운트한다.
3. 텍스트가 없고 심볼만 있는 TB/CH/CTT/PTT도 카운트 대상이다.
4. 도면에 'TEST', 'TT', 'TERMINAL', 'TB' 표기가 보이면 주변 심볼을 우선 탐색하여 CTT/PTT/TB로 매핑한다.
5. 보조기기가 메인 기기보다 작게 그려지는 경우가 많으므로, 작은 심볼도 반드시 식별하여 수량에 포함한다.
6. 누락 방지 규칙에 따라 보조기기를 식별할 때는 메인 기기와의 상대적 위치와 크기를 반드시 고려한다.
7. 보조기기가 명확히 구분되지 않는 경우에도, 도면에 해당 기기의 존재가 암시되면 반드시 수량에 포함한다.

==============================
[Panel Schedule 해석 규칙]
==============================

1. 도면 우측에 "PANEL NAME / AF / AT / KA" 형식의 표가 존재하면 이를 Panel Schedule로 인식한다.
2. Panel Schedule의 각 행(1,2,3...)은 왼쪽 회로 심볼과 1:1로 대응되는 회로(Feeder)이다.
3. SPARE로 표시된 회로도 실제 MCCB가 설치된 회로이므로 반드시 카운트한다.
4. MCCB 사양은 다음 규칙으로 해석한다.

   MCCB 사양 = 
   [좌측 MCCB 심볼] + [Panel Schedule AF/AT/KA 값]

5. 동일 사양 MCCB는 다음과 같이 그룹화하여 수량을 계산한다.

예:
250AF / 150AT / 50kA → 1EA  
250AF / 250AT / 50kA → 1EA  
125AF / 125AT / 50kA → 4EA  
125AF / 75AT / 50kA → 4EA

중간 추론 과정은 출력하지 않는다.
최종 결과만 출력한다.
"""

_BOM_USER_PROMPT = """이 단선도면을 분석하여 아래 항목을 모두 추출하세요.

1) Drawing Type
- Key SLD인지 일반 SLD인지 판단

2) Panel Name
- 좌측 상단 육각형 내부에 표기된 값

3) Final Panel BOM

| Panel | Device Symbol | Device Name | Specification | Qty | Confidence |

## Confidence Criteria
- High : 도면에서 기기와 사양이 명확하게 확인됨
- Medium : 기기는 확인되지만 일부 정보가 불명확함
- Low : 정보가 흐리거나 추정이 필요한 경우

※ Confidence must be one of **High / Medium / Low**
※ 중간 설명 없이 최종 결과만 출력하세요.
"""

def run_step5(settings):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.agents.llm_caller import call_llm
    crops=_s("panel_crops",[])
    client=_llm(settings); deploy=settings.azure_openai_deployment
    bom={}
    done=0; bar=st.progress(0,"Extracting BOM from panel crops …")
    t0 = time.time()
    def _proc(c):
        cp=c.get("crop_path")
        if not cp or not Path(cp).exists(): return c["panel_name"],"No crop image available."
        try:
            result=call_llm(client,deploy,_BOM_USER_PROMPT,
                system_prompt=_BOM_SYSTEM_PROMPT,
                image_paths=[cp],label=f"bom {c['panel_name']}")
            return c["panel_name"],result
        except Exception as e:
            print(f"  [BOM ERROR] {c['panel_name']}: {e}")
            return c["panel_name"],f"ERROR: {e}"
    with ThreadPoolExecutor(max_workers=min(len(crops),5)) as ex:
        futs={ex.submit(_proc,c):c for c in crops}
        for f in as_completed(futs):
            name,result=f.result(); bom[name]=result; done+=1
            bar.progress(done/len(crops),f"Done: {done}/{len(crops)} panels")
    _ss("bom_results",bom); _ss("bom_confirmed",False)
    _ss("step",5); _ss("current_view",4)
    elapsed = time.time() - t0
    timings = _s("step_timings", {}); timings["step5"] = elapsed; _ss("step_timings", timings)
    bar.empty()

def render_step5():
    st.markdown('<span class="step-badge">Step 5</span>',unsafe_allow_html=True)
    st.subheader("BOM Extraction — Review Results")
    bom=_s("bom_results",{}); crops=_s("panel_crops",[])
    if not bom: st.info("Run Step 5 first."); return
    confirmed=_s("bom_confirmed",False)
    # Show timing
    t5 = _s("step_timings", {}).get("step5")
    if t5:
        st.markdown(f'<div class="info-banner">Extracted BOM for <b>{len(crops)} panels</b> {_elapsed_html(t5)}</div>', unsafe_allow_html=True)
    for c in crops:
        name=c["panel_name"]; result=bom.get(name,"")
        cp=c.get("crop_path")
        is_error = result.startswith("ERROR:") if result else False
        with st.expander(f"{'⚠️ ' if is_error else ''}{name} — BOM",expanded=True):
            if is_error:
                st.markdown(f'<div class="error-banner"><b>Error extracting BOM:</b> {result}</div>', unsafe_allow_html=True)
                if st.button(f"🔄 Retry {name}", key=f"retry_bom_{name}"):
                    from src.agents.llm_caller import call_llm
                    settings = _load_settings()
                    client = _llm(settings)
                    try:
                        new_result = call_llm(client, settings.azure_openai_deployment,
                            _BOM_USER_PROMPT, system_prompt=_BOM_SYSTEM_PROMPT,
                            image_paths=[cp], label=f"bom {name}")
                        bom[name] = new_result; _ss("bom_results", bom); st.rerun()
                    except Exception as e:
                        st.error(f"Retry failed: {e}")
                continue
            cols=st.columns([1,2])
            with cols[0]:
                if cp and Path(cp).exists():
                    st.image(Image.open(cp),caption=name,use_container_width=True)
                else:
                    st.warning("No crop image")
            with cols[1]:
                if not confirmed:
                    view_mode = st.radio("View", ["📊 Table", "✏️ Edit"], key=f"bom_view_{name}", horizontal=True)
                    if view_mode == "✏️ Edit":
                        edited=st.text_area("BOM Result",value=result,height=400,key=f"bom_{name}")
                        if edited!=result:
                            bom[name]=edited; _ss("bom_results",bom)
                    else:
                        st.markdown(result)
                else:
                    st.markdown(result)
    if not confirmed:
        st.markdown("---")
        if st.button("Confirm BOM → Done",type="primary",use_container_width=True):
            _ss("bom_confirmed",True); st.rerun()
    else:
        st.markdown('<div class="success-banner"><b>BOM extraction complete!</b> All panels processed successfully.</div>', unsafe_allow_html=True)
        _render_final()

# ══════════════════════════════════════════════════════════════════
#  Visual Prompt Sidebar
# ══════════════════════════════════════════════════════════════════
def _render_vp_sidebar():
    st.sidebar.markdown("---"); st.sidebar.markdown("**Visual Prompts**")
    # Sample visual prompts
    _VP_SAMPLES = {
        "panel_name": [
            VISUAL_PROMPT_DIR / "panel_name_box_example1.png",
            VISUAL_PROMPT_DIR / "panel_name_box_example2.png",
        ],
        "area_split": [
            VISUAL_PROMPT_DIR / "panel_box_explanation.png",
            VISUAL_PROMPT_DIR / "bay_example.png",
        ],
    }
    for label,key in [("Panel Name Detection","panel_name"),("Panel Area Detection","area_split")]:
        st.sidebar.markdown(f"**{label}:**")
        samples = [p for p in _VP_SAMPLES.get(key, []) if p.exists()]
        opts=[p.name for p in samples]
        paths_map = {p.name: str(p) for p in samples}
        chosen=st.sidebar.multiselect("Images",opts,key=f"vp_{key}")
        vps=_s("visual_prompts",{})
        if chosen:
            full_paths = [paths_map[c] for c in chosen if c in paths_map]
            vps[key]=full_paths; _ss("visual_prompts",vps)
            cols=st.sidebar.columns(min(len(full_paths),3))
            for i,fp in enumerate(full_paths):
                if Path(fp).exists(): cols[i%len(cols)].image(fp,width=120)
        else:
            vps.pop(key,None); _ss("visual_prompts",vps)

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(page_title="Electrical SLD (Single Line Diagram) BOM Extraction — GBB",
        page_icon=None,layout="wide",initial_sidebar_state="expanded")
    _init()
    st.markdown(GBB_CSS,unsafe_allow_html=True)
    st.markdown(GBB_CORNER_HTML,unsafe_allow_html=True)
    settings=_load_settings()

    # ── Sidebar ──
    if BANNER_PATH.exists():
        import base64 as _b64
        _banner_b64 = _b64.b64encode(BANNER_PATH.read_bytes()).decode()
        st.sidebar.markdown(
            f'<div style="border-radius:8px;overflow:hidden;margin:-0.5rem 0 0.6rem 0;">'
            f'<img src="data:image/png;base64,{_banner_b64}" '
            f'style="width:100%;display:block;border-radius:8px;" />'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown(
        "### Electrical SLD BOM Extraction\n"
        '<span style="font-size:.75rem;color:#0078D4;font-weight:600;display:inline-flex;align-items:center;gap:5px;"><img src="https://gbb.azureai.win/assets-gbb/logo/gbb-icon-color.svg" style="height:14px;display:inline-block;vertical-align:middle;"/> Global Black Belt · AI Apps</span>',
        unsafe_allow_html=True)
    st.sidebar.markdown("---")
    uploaded=st.sidebar.file_uploader("Upload PDF",type=["pdf"],key="pdf_up")
    if uploaded and not _s("pdf_path"):
        ud=settings.output_path/"uploads"; ud.mkdir(parents=True,exist_ok=True)
        p=ud/uploaded.name; p.write_bytes(uploaded.read()); _ss("pdf_path",str(p))
    test_pdfs=list(VISUAL_PROMPT_DIR.glob("*.pdf")) if VISUAL_PROMPT_DIR.exists() else []
    if test_pdfs:
        st.sidebar.markdown("**Or use a test PDF:**")
        opts=["—"]+[p.name for p in test_pdfs]
        choice=st.sidebar.selectbox("Test PDF",opts,key="test_pdf")
        if choice!="—": _ss("pdf_path",str(VISUAL_PROMPT_DIR/choice))
    step=_s("step",0)
    st.sidebar.markdown("---")
    # ── Pipeline Stepper (visual) ──
    # Step 2 sub-phase awareness
    di_done_s = _s("di_detection_done", False)
    missing_done_s = _s("missing_detection_done", False)
    items_html=[]
    for i,lbl in enumerate(STEP_LABELS):
        # Step 2 (i==1): only mark done when BOTH phases complete + confirmed
        if i == 1:
            if step > 2:
                state = "done"
            elif di_done_s and missing_done_s and _s("regions_confirmed", False):
                state = "done"
            elif di_done_s:
                state = "active"
            elif step >= 1:
                state = "active"
            else:
                state = "pending"
            sub = ""
            if state == "active":
                if not di_done_s:
                    sub = ' <span style="font-size:.7rem;color:#5C2D91;">· Phase A</span>'
                elif not missing_done_s:
                    sub = ' <span style="font-size:.7rem;color:#5C2D91;">· Phase B</span>'
                elif not _s("regions_confirmed", False):
                    sub = ' <span style="font-size:.7rem;color:#5C2D91;">· HITL</span>'
            lbl_html = lbl + sub
        else:
            state="done" if i<step else ("active" if i==step else "pending")
            lbl_html = lbl
        dot="✓" if state=="done" else str(i+1)
        line='<div class="ps-line"></div>' if i<len(STEP_LABELS)-1 else ""
        items_html.append(f'<div class="ps-item"><div class="ps-left"><div class="ps-dot {state}">{dot}</div>{line}</div><div class="ps-label {state}">{lbl_html}</div></div>')
    st.sidebar.markdown(f'<div class="pipeline-stepper">{chr(10).join(items_html)}</div>',unsafe_allow_html=True)

    # ── Step navigation buttons (click to go back) ──
    # Only show buttons for steps whose prerequisites are met
    _nav_ok = [
        step >= 0,                                          # Step 1: always
        step >= 1,                                          # Step 2: after step1
        step >= 2 and _s("regions_confirmed", False),       # Step 3: after regions confirmed
        step >= 3 and _s("names_confirmed", False),         # Step 4: after names confirmed
        step >= 4 and _s("crops_confirmed", False),         # Step 5: after crops confirmed
    ]
    nav_cols = st.sidebar.columns(len(STEP_LABELS))
    for i in range(len(STEP_LABELS)):
        with nav_cols[i]:
            if _nav_ok[i]:
                disabled = (i == _s("current_view", 0))
                if st.button(str(i+1), key=f"step_nav_{i}", use_container_width=True,
                             disabled=disabled, type="primary" if disabled else "secondary"):
                    _ss("current_view", i); st.rerun()

    # ── Region count summary in sidebar ──
    rbp_s = _s("regions_by_page", {})
    if rbp_s:
        total_r = sum(len(v) for v in rbp_s.values())
        n_di_r = sum(1 for v in rbp_s.values() for r in v if r.get("source") == "di")
        n_llm_r = total_r - n_di_r
        st.sidebar.markdown(
            f'<div style="background:#f0f6ff;border:1px solid #b3d7ff;border-radius:8px;padding:8px 12px;'
            f'font-size:.8rem;margin:4px 0 8px 0;">'
            f'<b style="color:#0078D4;">Regions:</b> '
            f'<span style="color:#d83b01;">{n_di_r} DI</span>'
            + (f' + <span style="color:#005a9e;">{n_llm_r} LLM</span>' if n_llm_r else '')
            + f' = <b style="color:#1a1a1a;">{total_r}</b>'
            f'</div>', unsafe_allow_html=True)

    cv=_s("current_view",0)
    hitl={2:("regions_confirmed","Awaiting region review (HITL #1)"),
          3:("names_confirmed","Awaiting name review (HITL #2)"),
          4:("crops_confirmed","Awaiting panel review (HITL #3)")}
    if step in hitl:
        k,msg=hitl[step]
        if not _s(k): st.sidebar.warning(f"{msg}")
    _render_vp_sidebar()
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset",use_container_width=True):
        for k in list(st.session_state.keys()):
            if k not in ("pdf_up","test_pdf"): del st.session_state[k]
        st.rerun()
    st.sidebar.caption("v3 DI+LLM | Phase A: DI, Phase B: LLM")

    # ── Main ──
    st.markdown("# Electrical SLD (Single Line Diagram) BOM Extraction\n"
        '<p style="color:#777;font-size:.9rem;margin-top:-8px;">Single Line Diagram · PDF → Figure Detection → Panel Names → Panel Areas + Bay → BOM Extraction</p>',
        unsafe_allow_html=True)
    pdf=_s("pdf_path")
    if not pdf:
        st.markdown(
            '<div class="gbb-card" style="padding:36px 32px;">'
            '<h2 style="margin:0 0 12px 0;color:#1a1a1a!important;font-size:1.2rem!important;">Getting Started</h2>'
            '<p style="color:#555;font-size:.88rem;margin:0 0 20px 0;">Upload an Electrical Single Line Diagram (SLD) PDF to extract a Bill of Materials (BOM). '
            'Use the sidebar to upload your file or select a test PDF.</p>'
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;">'
            '  <div style="background:#f0f6ff;border-radius:8px;padding:14px 16px;border:1px solid #d6e8ff;">'
            '    <div style="font-size:1.1rem;margin-bottom:4px;">📄</div>'
            '    <b style="color:#0078D4;font-size:.82rem;">Step 1 · Upload & Select Pages</b>'
            '    <p style="color:#555;font-size:.78rem;margin:4px 0 0;">Upload a PDF and convert pages to images for analysis.</p></div>'
            '  <div style="background:#f0f6ff;border-radius:8px;padding:14px 16px;border:1px solid #d6e8ff;">'
            '    <div style="font-size:1.1rem;margin-bottom:4px;">🔍</div>'
            '    <b style="color:#0078D4;font-size:.82rem;">Step 2 · Figure Detection</b>'
            '    <p style="color:#555;font-size:.78rem;margin:4px 0 0;">DI + LLM detect panel regions and bounding boxes on each page.</p></div>'
            '  <div style="background:#f0f6ff;border-radius:8px;padding:14px 16px;border:1px solid #d6e8ff;">'
            '    <div style="font-size:1.1rem;margin-bottom:4px;">🏷️</div>'
            '    <b style="color:#0078D4;font-size:.82rem;">Step 3 · Panel Names</b>'
            '    <p style="color:#555;font-size:.78rem;margin:4px 0 0;">Extract and label each detected panel with its name via OCR + LLM.</p></div>'
            '  <div style="background:#f0f6ff;border-radius:8px;padding:14px 16px;border:1px solid #d6e8ff;">'
            '    <div style="font-size:1.1rem;margin-bottom:4px;">✂️</div>'
            '    <b style="color:#0078D4;font-size:.82rem;">Step 4 · Panel Crops</b>'
            '    <p style="color:#555;font-size:.78rem;margin:4px 0 0;">Locate, verify and crop individual panel areas from the page images.</p></div>'
            '  <div style="background:#f0f6ff;border-radius:8px;padding:14px 16px;border:1px solid #d6e8ff;">'
            '    <div style="font-size:1.1rem;margin-bottom:4px;">📋</div>'
            '    <b style="color:#0078D4;font-size:.82rem;">Step 5 · BOM Extraction</b>'
            '    <p style="color:#555;font-size:.78rem;margin:4px 0 0;">Extract component data from each panel crop to produce the final BOM.</p></div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True)
        return
    st.markdown(f'<p style="color:#555;font-size:.85rem;"><code>{Path(pdf).name}</code></p>',unsafe_allow_html=True)
    cv=_s("current_view",0)
    crumbs=[]
    for i,lbl in enumerate(STEP_LABELS):
        if i==cv: crumbs.append(f'<span class="active">{lbl}</span>')
        elif i<step: crumbs.append(f'<span class="done">✓ {lbl}</span>')
        else: crumbs.append(f'<span style="color:#333">{lbl}</span>')
    sep = '<span class="sep"> › </span>'
    st.markdown(f'<div class="breadcrumb">{sep.join(crumbs)}</div>',unsafe_allow_html=True)
    # Show cumulative timing summary
    timings = _s("step_timings", {})
    if timings:
        chips = []
        for k, v in sorted(timings.items()):
            label = k.replace("step", "S")
            chips.append(f'<span class="elapsed-chip">{label}: <span class="val">{_fmt_elapsed(v)}</span></span>')
        st.markdown(f'<div style="margin:-4px 0 8px 0;">{"".join(chips)}</div>', unsafe_allow_html=True)
    st.markdown("---")

    def _nb(target,label,key):
        if st.button(f"Next → {label}",key=key,use_container_width=True,type="primary"):
            _ss("current_view",target); st.rerun()

    if cv==0:
        if step==0:
            if st.button("Run Step 1: Upload & Convert",type="primary",use_container_width=True):
                with st.spinner("Converting…"): run_step1(pdf,settings)
                st.rerun()
        render_step1()
        if step>=1:
            st.markdown("---"); n=len(_active_pages())
            if n==0: st.warning("Select at least one page.")
            else: _nb(1,f"Step 2: Figure Detection ({n}p)","next1")
    elif cv==1:
        if step>=1:
            di_done = _s("di_detection_done", False)
            missing_done = _s("missing_detection_done", False)
            last_err = _s("step_errors", {}).get("step2a") or _s("step_errors", {}).get("step2b")
            if last_err:
                st.markdown(f'<div class="error-banner"><b>Last error:</b> {last_err}</div>', unsafe_allow_html=True)
            if not di_done:
                if st.button("Run Step 2: Figure Detection",type="primary",use_container_width=True):
                    try:
                        with st.spinner("Running Phase A: DI detection (Pass 1 + 2)…"): run_step2a(settings)
                        errs = _s("step_errors", {}); errs.pop("step2a", None); _ss("step_errors", errs)
                    except Exception as e:
                        errs = _s("step_errors", {}); errs["step2a"] = str(e); _ss("step_errors", errs)
                        st.error(f"Step 2A failed: {e}")
                        _safe_print_exc()
                        st.rerun()
                    # Auto-run Phase B after Phase A succeeds
                    try:
                        with st.spinner("Running Phase B: LLM missing-image detection…"): run_step2b(settings)
                        errs = _s("step_errors", {}); errs.pop("step2b", None); _ss("step_errors", errs)
                    except Exception as e:
                        errs = _s("step_errors", {}); errs["step2b"] = str(e); _ss("step_errors", errs)
                        st.error(f"Step 2B failed: {e}")
                        _safe_print_exc()
                    st.rerun()
            elif not missing_done:
                # Fallback: if 2A done but 2B not (e.g. previous error), allow manual retry
                if st.button("Retry Step 2B: LLM Missing Detection",type="primary",use_container_width=True):
                    try:
                        with st.spinner("Running LLM missing-image detection…"): run_step2b(settings)
                        errs = _s("step_errors", {}); errs.pop("step2b", None); _ss("step_errors", errs)
                    except Exception as e:
                        errs = _s("step_errors", {}); errs["step2b"] = str(e); _ss("step_errors", errs)
                        st.error(f"Step 2B failed: {e}")
                        _safe_print_exc()
                    st.rerun()
            render_step2()
            if step>=2 and _s("regions_confirmed"):
                st.markdown("---"); _nb(2,"Step 3: Panel Names","next2")
        else: st.info("Complete Step 1 first.")
    elif cv==2:
        if step>=2 and _s("regions_confirmed"):
            last_err = _s("step_errors", {}).get("step3")
            if last_err:
                st.markdown(f'<div class="error-banner"><b>Last error:</b> {last_err}</div>', unsafe_allow_html=True)
            if step==2 or last_err:
                if st.button("Run Step 3: Extract Panel Names",type="primary",use_container_width=True):
                    try:
                        with st.spinner("Extracting names…"): run_step3(settings)
                        errs = _s("step_errors", {}); errs.pop("step3", None); _ss("step_errors", errs)
                    except Exception as e:
                        errs = _s("step_errors", {}); errs["step3"] = str(e); _ss("step_errors", errs)
                        st.error(f"Step 3 failed: {e}")
                        _safe_print_exc()
                    st.rerun()
            render_step3()
            if step>=3 and _s("names_confirmed"):
                st.markdown("---"); _nb(3,"Step 4: Panel Areas + Bay","next3")
        elif step>=2: st.warning("Confirm Step 2 regions first.")
        else: st.info("Complete Step 2 first.")
    elif cv==3:
        if step>=3 and _s("names_confirmed"):
            last_err = _s("step_errors", {}).get("step4")
            if last_err:
                st.markdown(f'<div class="error-banner"><b>Last error:</b> {last_err}</div>', unsafe_allow_html=True)
            if step==3 or last_err:
                if st.button("Run Step 4: Panel Areas + Bay",type="primary",use_container_width=True):
                    try:
                        with st.spinner("Detecting areas…"): run_step4(settings)
                        errs = _s("step_errors", {}); errs.pop("step4", None); _ss("step_errors", errs)
                    except Exception as e:
                        errs = _s("step_errors", {}); errs["step4"] = str(e); _ss("step_errors", errs)
                        st.error(f"Step 4 failed: {e}")
                        _safe_print_exc()
                    st.rerun()
            render_step4()
            if step>=4 and _s("crops_confirmed"):
                st.markdown("---"); _nb(4,"Step 5: BOM Extraction","next4")
        elif step>=3: st.warning("Confirm Step 3 names first.")
        else: st.info("Complete Step 3 first.")
    elif cv==4:
        if step>=4 and _s("crops_confirmed"):
            last_err = _s("step_errors", {}).get("step5")
            if last_err:
                st.markdown(f'<div class="error-banner"><b>Last error:</b> {last_err}</div>', unsafe_allow_html=True)
            if step==4 or last_err:
                if st.button("Run Step 5: BOM Extraction",type="primary",use_container_width=True):
                    try:
                        with st.spinner("Extracting BOM from panels…"): run_step5(settings)
                        errs = _s("step_errors", {}); errs.pop("step5", None); _ss("step_errors", errs)
                    except Exception as e:
                        errs = _s("step_errors", {}); errs["step5"] = str(e); _ss("step_errors", errs)
                        st.error(f"Step 5 failed: {e}")
                        _safe_print_exc()
                    st.rerun()
            render_step5()
        elif step>=4: st.warning("Confirm Step 4 panels first.")
        else: st.info("Complete Step 4 first.")

if __name__=="__main__":
    main()
