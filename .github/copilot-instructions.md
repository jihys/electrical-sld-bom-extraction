# Electrical SLD BOM Extraction

CAD Single Line Diagram(SLD)에서 전기 패널 영역을 추출하는 멀티 에이전트 시스템.

## Tech Stack
- **Backend**: Python, FastAPI, Microsoft Agent Framework SDK
- **Frontend**: Streamlit (HITL UI) at `http://localhost:8501`
- **LLM**: Azure OpenAI (Responses API)
- **OCR**: Azure Document Intelligence (2-pass figure detection)
- **PDF**: PyMuPDF

## Run Commands
- **Streamlit UI**: `streamlit run src/hitl/streamlit_app.py --server.port 8501`
- **API Server**: `uvicorn src.api.app:app --reload --port 8000`

## Pipeline Overview
8단계 파이프라인, 3개의 HITL 체크포인트:
```
Segment → Detect → MissingDetect → NameExt → Match → LocateVerify → Validation → BaySplit
                        ↕              ↕                    ↕
                   HITL #1         HITL #2              HITL #3
```

Streamlit UI에서는 이를 5개 Step + 3개 HITL 확인 단계로 표시.

## Key Source Paths
- `src/hitl/streamlit_app.py` — Streamlit HITL UI (메인 데모)
- `src/workflow/executors.py` — 파이프라인 Executor 클래스
- `src/agents/` — LLM 호출 로직
- `src/tools/` — PDF, 이미지, 기하 유틸리티
- `tests/e2e_demo_test.py` — Playwright E2E 테스트
