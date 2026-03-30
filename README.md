# Electrical SLD (Single Line Diagram) BOM Extraction

Agentic pipeline for extracting electrical panel areas and Bill of Materials (BOM) from CAD Single Line Diagrams.

Uses **Azure OpenAI** (GPT-5.4) + **Azure Document Intelligence** for detection and extraction, with a **Streamlit** Human-in-the-Loop UI.

## Demo Screenshots

| Getting Started | Step 1 — Upload & Select Pages |
|:---:|:---:|
| ![Getting Started](docs/images/01_getting_started.png) | ![Upload Pages](docs/images/02_upload_pages.png) |

| Step 2 — Figure Detection (HITL #1) | Step 4 — Panel Areas & Bay |
|:---:|:---:|
| ![Figure Detection](docs/images/03_figure_detection.png) | ![Panel Areas](docs/images/04_panel_areas.png) |

| Step 4 — Panel Crops | Step 5 — BOM Extraction |
|:---:|:---:|
| ![Panel Crops](docs/images/05_panel_crops.png) | ![BOM Extraction](docs/images/06_bom_extraction.png) |

## Prerequisites

- **Python 3.10+**
- **Azure OpenAI** resource (GPT-5.4 or later with Responses API support)
- **Azure Document Intelligence** resource (prebuilt-layout model)

## Quick Start

### Option A — One-command setup

```bash
git clone https://github.com/<your-org>/electrical-sld-bom-extraction.git
cd electrical-sld-bom-extraction
bash setup.sh          # Creates venv, installs deps, copies .env.example
```

### Option B — Manual setup

```bash
git clone https://github.com/<your-org>/electrical-sld-bom-extraction.git
cd electrical-sld-bom-extraction

# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env with your Azure credentials (see below)

# 4. Create output directories
mkdir -p outputs checkpoints
```

### Configure Azure Credentials

Edit `.env` with your actual values:

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>          # or leave empty for DefaultAzureCredential
AZURE_OPENAI_API_VERSION=2025-03-01-preview
AZURE_OPENAI_DEPLOYMENT=<your-deployment>  # e.g. gpt-5.4

# Azure Document Intelligence
AZURE_DI_ENDPOINT=https://<your-di>.cognitiveservices.azure.com
AZURE_DI_KEY=<your-key>                  # or leave empty for DefaultAzureCredential
```

### Run the Demo

```bash
source venv/bin/activate

# Streamlit UI (recommended)
make run
# or directly:
streamlit run src/hitl/streamlit_app.py --server.port 8501

# FastAPI server (optional)
make run-api
```

Open **http://localhost:8501** in your browser. Select `public_sld_1.pdf` from the sidebar dropdown to try the sample data.

## Architecture

```
Segment → Detect → MissingDetect → NameExt → Match → LocateVerify → Validation → BaySplit
                         ↕              ↕                    ↕
                    HITL #1         HITL #2              HITL #3
              (confirm regions)  (confirm names)   (confirm areas/bays)
```

### Pipeline Stages

| # | Executor | Role | HITL |
|---|----------|------|------|
| 1 | **SegmentExecutor** | PDF → per-page PNG + SVG (PyMuPDF with auto-DPI) | — |
| 2 | **DetectExecutor** | Azure Document Intelligence 2-pass figure detection | — |
| 3 | **MissingDetectExecutor** | LLM-based detection of figure regions DI missed + verification | **HITL #1**: confirm/edit figure region bboxes |
| 4 | **NameExtExecutor** | Tile-based panel name extraction via LLM + hallucination guard | — |
| 5 | **MatchExecutor** | 6-pass rule engine + LLM fallback to match names → DI text bboxes | **HITL #2**: confirm/edit panel names + label bboxes |
| 6 | **LocateVerifyExecutor** | Iterative locate→verify loop for panel area boundaries | — |
| 7 | **ValidationExecutor** | Cross-panel validation + confidence-based HITL routing | — |
| 8 | **BaySplitExecutor** | Bay subdivision within verified panels | **HITL #3**: confirm/edit panel area + bay bboxes |

### Streamlit UI (5 Steps + 3 HITL Checkpoints)

| UI Step | Action | Wait Time |
|---------|--------|-----------|
| Step 1 | Upload PDF → Select Pages → Convert to PNG | ~5s |
| Step 2 | Figure Detection (DI + LLM) | 30-120s |
| HITL #1 | Review/edit figure region bounding boxes | — |
| Step 3 | Panel Name Extraction (OCR + LLM) | 30-120s |
| HITL #2 | Review/edit panel names + label boxes | — |
| Step 4 | Panel Area + Bay Detection (LLM) | 60-180s |
| HITL #3 | Review/edit panel areas + bay divisions | — |
| Step 5 | BOM Extraction | 30-120s |

### HITL Flow (3 Checkpoints)

1. **HITL #1 — Figure Regions**: DI + LLM detected regions with bboxes overlaid. Drag to move, add/remove regions.
2. **HITL #2 — Panel Names**: Extracted names with blue overlay boxes. Add/rename/delete names, toggle overlay.
3. **HITL #3 — Panel Areas & Bays**: Panel area bboxes + bay dividers. Adjust boundaries, add/modify cropped regions.

Visual prompt images can be selected in the sidebar to improve LLM accuracy:
- **Panel Name Detection**: `panel_name_box_example1.png`, `panel_name_box_example2.png`
- **Panel Area Detection**: `panel_box_explanation.png`

## Project Structure

```
electrical-sld-bom-extraction/
├── pyproject.toml          # Dependencies & build config
├── requirements.lock.txt   # Pinned versions for reproducibility
├── setup.sh                # One-command environment setup
├── Makefile                # Common commands
├── .env.example            # Template for Azure credentials
├── data/                   # Sample SLD PDFs + visual prompt images
│   ├── public_sld_1.pdf
│   ├── public_sld_2.pdf
│   ├── panel_box_explanation.png
│   ├── panel_name_box_example1.png
│   ├── panel_name_box_example2.png
│   └── bay_example.png
├── src/
│   ├── config.py
│   ├── models/             # Pydantic data models
│   ├── tools/              # Pure functions (PDF, image, geometry)
│   ├── agents/             # LLM-based logic (prompts, matching, extraction)
│   ├── validators/         # Cross-panel validation
│   ├── workflow/           # Pipeline executor classes
│   ├── hitl/               # Streamlit UI + review queue
│   └── api/                # FastAPI routes
└── tests/
    ├── e2e_demo_test.py    # Playwright E2E test
    └── e2e_full_test.py    # Full scenario E2E test
```

## Make Commands

| Command | Description |
|---------|-------------|
| `make setup` | Full environment setup (venv + deps + .env) |
| `make run` | Start Streamlit UI (port 8501) |
| `make run-bg` | Start in tmux (disconnect-safe) |
| `make run-api` | Start FastAPI server (port 8000) |
| `make test` | Run pytest |
| `make test-e2e` | Run Playwright E2E test |
| `make venv-fix` | Recreate venv (fixes broken shebang) |
| `make freeze` | Update requirements.lock.txt |
| `make status` | Check server/tmux/venv status |
| `make clean` | Remove outputs and caches |

## Testing

```bash
# Unit tests
make test

# E2E test (requires running Streamlit server + Playwright)
pip install playwright && playwright install chromium
make test-e2e
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/run` | Start extraction (`{"pdf_path": "..."}`) |
| `GET` | `/api/run/{id}` | Check run status |
| `GET` | `/api/reviews` | List pending HITL reviews |
| `POST` | `/api/reviews/{id}` | Submit HITL corrections |
| `GET` | `/api/results/{id}` | Get final results |
| `GET` | `/health` | Health check |

## Key Design Decisions

- **Multi-stage pipeline** — each stage has a dedicated executor class
- **Confidence-based HITL** — only panels below threshold (0.7) trigger human review
- **Hallucination guard** — LLM-extracted panel names validated against DI OCR output
- **Oscillation detection** — prevents locate→verify from endlessly toggling edge coordinates
