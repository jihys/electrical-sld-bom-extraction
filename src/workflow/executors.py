"""Workflow Executors — MAF Executor nodes for the panel extraction pipeline.

Each Executor handles one stage of the pipeline:
  1. SegmentExecutor   — PDF → per-page PNG/SVG
  2. DetectExecutor    — DI two-pass figure detection
  3. NameExtExecutor   — Tile-based panel name extraction
  4. MatchExecutor     — Rule-based + LLM bbox matching
  5. LocateVerifyExecutor — Locate→Verify loop per panel
  6. ValidationExecutor — Cross-panel validation + HITL routing
  7. BaySplitExecutor  — Bay subdivision
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_framework import Executor, WorkflowContext, handler, response_handler

from ..config import Settings
from ..models.page import PageInfo
from ..models.panel import PanelCrop, BboxMatch
from ..models.region import PageExtractionSummary
from ..models.workflow_state import (
    BaySplitResult,
    FinalResult,
    HitlReviewRequest,
    MatchesReady,
    NamesExtracted,
    PagesReady,
    PanelsCropped,
    PdfInput,
    RegionsDetected,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Segment: PDF → Pages
# ──────────────────────────────────────────────────────────────────────────────

class SegmentExecutor(Executor):
    """Split a PDF into per-page PNG + SVG files."""

    def __init__(self, settings: Settings):
        super().__init__(id="segment")
        self.settings = settings

    @handler
    async def on_pdf(self, msg: PdfInput, ctx: WorkflowContext) -> None:
        from ..tools.pdf_tools import compute_dpi_for_pdf, split_pdf_to_pages

        output_dir = str(Path(self.settings.output_dir) / "pages")
        global_dpi, per_page_dpi = compute_dpi_for_pdf(msg.pdf_path)
        pages = split_pdf_to_pages(
            msg.pdf_path, output_dir,
            dpi=global_dpi, per_page_dpi=per_page_dpi,
        )
        print(f"[SegmentEx] Split {msg.pdf_path} → {len(pages)} pages (global DPI={global_dpi})")
        await ctx.send_message(PagesReady(pages=pages))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Detect: DI two-pass figure/table detection
# ──────────────────────────────────────────────────────────────────────────────

class DetectExecutor(Executor):
    """Run Azure Document Intelligence 2-pass detection on each page."""

    def __init__(self, settings: Settings):
        super().__init__(id="detect")
        self.settings = settings
        self._di_client = None

    def _get_di_client(self):
        if self._di_client is None:
            from ..tools.di_tools import create_di_client
            self._di_client = create_di_client(
                self.settings.azure_di_endpoint,
                self.settings.azure_di_key,
            )
        return self._di_client

    @handler
    async def on_pages(self, msg: PagesReady, ctx: WorkflowContext) -> None:
        from ..tools.di_tools import two_pass_detection

        client = self._get_di_client()
        summaries: List[PageExtractionSummary] = []
        di_lines_by_page: Dict[int, list] = {}

        for page in msg.pages:
            out_dir = str(Path(self.settings.output_dir) / f"di_page{page.page_num}")
            summary = two_pass_detection(
                client, "prebuilt-layout", page.png_path, out_dir, page.page_num,
            )
            summaries.append(summary)

            # Also run a simple analysis for OCR lines
            from ..tools.di_tools import analyze_page
            analysis = analyze_page(client, "prebuilt-layout", page.png_path)
            di_lines_by_page[page.page_num] = analysis["lines"]

        print(f"[DetectEx] Processed {len(summaries)} pages via DI 2-pass")
        await ctx.send_message(RegionsDetected(
            pages=msg.pages,
            summaries=summaries,
            di_lines_by_page=di_lines_by_page,
        ))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Name Extraction: Tile → LLM → Dedup → Hallucination guard
# ──────────────────────────────────────────────────────────────────────────────

class NameExtExecutor(Executor):
    """Extract panel names from each page via tiling + LLM."""

    def __init__(self, settings: Settings):
        super().__init__(id="name_ext")
        self.settings = settings
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from ..agents.llm_caller import create_llm_client
            self._llm = create_llm_client(
                self.settings.azure_openai_endpoint,
                self.settings.azure_openai_api_key,
                self.settings.azure_openai_api_version,
            )
        return self._llm

    @handler
    async def on_regions(self, msg: RegionsDetected, ctx: WorkflowContext) -> None:
        from ..agents.name_extractor import (
            tile_page, extract_names_from_tile,
            dedup_panel_names, hallucination_guard,
        )

        llm = self._get_llm()
        deploy = self.settings.azure_openai_deployment
        names_by_page: Dict[int, List[str]] = {}

        for page in msg.pages:
            tiles = tile_page(page.png_path)
            all_candidates: List[str] = []

            for i, (tile_img, _bbox) in enumerate(tiles):
                tile_names = extract_names_from_tile(tile_img, llm, deploy, i)
                all_candidates.extend(tile_names)

            deduped = dedup_panel_names(all_candidates, llm, deploy, page.png_path)

            di_lines = msg.di_lines_by_page.get(page.page_num, [])
            verified = hallucination_guard(deduped, di_lines)

            names_by_page[page.page_num] = verified
            print(f"[NameExtEx] Page {page.page_num}: {len(verified)} panel names: {verified}")

        await ctx.send_message(NamesExtracted(
            pages=msg.pages,
            summaries=msg.summaries,
            di_lines_by_page=msg.di_lines_by_page,
            names_by_page=names_by_page,
        ))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Match: Rule-based + LLM fallback bbox matching
# ──────────────────────────────────────────────────────────────────────────────

class MatchExecutor(Executor):
    """Match panel names to DI text-line bounding boxes."""

    def __init__(self, settings: Settings):
        super().__init__(id="match")
        self.settings = settings
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from ..agents.llm_caller import create_llm_client
            self._llm = create_llm_client(
                self.settings.azure_openai_endpoint,
                self.settings.azure_openai_api_key,
                self.settings.azure_openai_api_version,
            )
        return self._llm

    @handler
    async def on_names(self, msg: NamesExtracted, ctx: WorkflowContext) -> None:
        from ..agents.bbox_matcher import (
            rule_match_panel_name, resolve_conflicts, llm_match_unresolved,
        )

        llm = self._get_llm()
        deploy = self.settings.azure_openai_deployment
        matches_by_page: Dict[int, Dict[str, Optional[BboxMatch]]] = {}

        for page in msg.pages:
            pn = page.page_num
            names = msg.names_by_page.get(pn, [])
            di_lines = msg.di_lines_by_page.get(pn, [])

            # Rule-based matching
            all_matches: Dict[str, List[BboxMatch]] = {}
            for name in names:
                all_matches[name] = rule_match_panel_name(name, di_lines)

            resolved = resolve_conflicts(all_matches)

            # LLM fallback for unresolved
            unresolved = [n for n, m in resolved.items() if m is None]
            if unresolved:
                llm_matches = llm_match_unresolved(
                    unresolved, di_lines, page.png_path, llm, deploy,
                )
                resolved.update(llm_matches)

            matches_by_page[pn] = resolved
            matched_count = sum(1 for m in resolved.values() if m is not None)
            print(f"[MatchEx] Page {pn}: {matched_count}/{len(names)} matched")

        await ctx.send_message(MatchesReady(
            pages=msg.pages,
            names_by_page=msg.names_by_page,
            di_lines_by_page=msg.di_lines_by_page,
            matches_by_page=matches_by_page,
        ))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Locate + Verify: Iterative loop per panel
# ──────────────────────────────────────────────────────────────────────────────

class LocateVerifyExecutor(Executor):
    """Run locate→verify loop for each matched panel."""

    def __init__(self, settings: Settings):
        super().__init__(id="locate_verify")
        self.settings = settings
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from ..agents.llm_caller import create_llm_client
            self._llm = create_llm_client(
                self.settings.azure_openai_endpoint,
                self.settings.azure_openai_api_key,
                self.settings.azure_openai_api_version,
            )
        return self._llm

    @handler
    async def on_matches(self, msg: MatchesReady, ctx: WorkflowContext) -> None:
        from ..agents.locate_verify import locate_and_verify_panel

        llm = self._get_llm()
        deploy = self.settings.azure_openai_deployment
        all_crops: List[PanelCrop] = []

        for page in msg.pages:
            pn = page.page_num
            matches = msg.matches_by_page.get(pn, {})
            output_dir = str(Path(self.settings.output_dir) / f"locate_page{pn}")
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Build other_panels dict for spatial context
            other_panels: Dict[str, List[int]] = {}
            for name, match in matches.items():
                if match is not None:
                    other_panels[name] = list(match.bbox)

            for name, match in matches.items():
                if match is None:
                    print(f"[LocateVerifyEx] Page {pn}: skipping '{name}' — no bbox match")
                    continue

                # Exclude self from other_panels
                others = {n: b for n, b in other_panels.items() if n != name}

                crop = locate_and_verify_panel(
                    llm, deploy, page.png_path,
                    name, list(match.bbox), others,
                    output_dir,
                    grid_size=self.settings.grid_size,
                    max_tries=self.settings.verify_max_tries,
                )

                if crop:
                    crop.crop_path = str(
                        Path(self.settings.output_dir) / f"crops/page{pn}_{crop.panel_name.replace(' ', '_')}.png"
                    )
                    # Save the actual crop
                    import cv2
                    full_img = cv2.imread(page.png_path)
                    x1, y1, x2, y2 = crop.bbox
                    cropped = full_img[y1:y2, x1:x2]
                    Path(crop.crop_path).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(crop.crop_path, cropped)

                    all_crops.append(crop)

        print(f"[LocateVerifyEx] Total crops: {len(all_crops)}")
        await ctx.send_message(PanelsCropped(
            pages=msg.pages,
            crops=all_crops,
            names_by_page=msg.names_by_page,
        ))


# ──────────────────────────────────────────────────────────────────────────────
# 6. Validation + HITL routing
# ──────────────────────────────────────────────────────────────────────────────

class ValidationExecutor(Executor):
    """Validate crops and route low-confidence ones to HITL."""

    def __init__(self, settings: Settings):
        super().__init__(id="validation")
        self.settings = settings

    @handler
    async def on_crops(self, msg: PanelsCropped, ctx: WorkflowContext) -> None:
        from ..validators.panel_validator import (
            validate_crop, validate_cross_panel, validate_name_coverage, needs_human_review,
        )

        page_dims = {p.page_num: (p.width, p.height) for p in msg.pages}
        review_items: List[Dict] = []
        approved_crops: List[PanelCrop] = []

        # Cross-panel validation per page
        cross_issues_by_page: Dict[int, List[str]] = {}
        for page in msg.pages:
            pn = page.page_num
            page_crops = [c for c in msg.crops if c.panel_name in msg.names_by_page.get(pn, [])]
            _, cross_issues = validate_cross_panel(page_crops, page.width, page.height)
            cross_issues_by_page[pn] = cross_issues

        for crop in msg.crops:
            # Find page for this crop
            page_num = None
            for pn, names in msg.names_by_page.items():
                if crop.panel_name in names:
                    page_num = pn
                    break

            if page_num is None:
                continue

            w, h = page_dims.get(page_num, (0, 0))
            is_valid, issues = validate_crop(crop, w, h)

            cross_issues = cross_issues_by_page.get(page_num, [])
            need_review, reason = needs_human_review(
                crop, self.settings.hitl_confidence_threshold, cross_issues,
            )

            if need_review:
                review_items.append({
                    "crop": crop,
                    "page_num": page_num,
                    "reason": reason,
                    "issues": issues,
                })
            else:
                approved_crops.append(crop)

        if review_items:
            print(f"[ValidationEx] {len(review_items)} panels need human review")
            await ctx.request_info(
                request_data=HitlReviewRequest(
                    crops=[item["crop"] for item in review_items],
                    pages=msg.pages,
                    reasons={item["crop"].panel_name: item["reason"] for item in review_items},
                ),
                response_type=str,
            )
        else:
            print(f"[ValidationEx] All {len(approved_crops)} panels approved")
            await ctx.send_message(PanelsCropped(
                pages=msg.pages,
                crops=approved_crops,
                names_by_page=msg.names_by_page,
            ))

    @response_handler
    async def on_hitl_response(
        self,
        original: HitlReviewRequest,
        response: str,
        ctx: WorkflowContext,
    ) -> None:
        """Handle human corrections from HITL."""
        corrections = json.loads(response) if isinstance(response, str) else response

        updated_crops: List[PanelCrop] = []
        for crop in original.crops:
            name = crop.panel_name
            if name in corrections:
                corr = corrections[name]
                crop.bbox = tuple(corr["bbox"])
                crop.confidence = 1.0
                crop.verified_by = "human"
                crop.status = "human_verified"
            updated_crops.append(crop)

        await ctx.send_message(PanelsCropped(
            pages=original.pages,
            crops=updated_crops,
            names_by_page={},
        ))


# ──────────────────────────────────────────────────────────────────────────────
# 7. Bay Split
# ──────────────────────────────────────────────────────────────────────────────

class BaySplitExecutor(Executor):
    """Subdivide verified panels into bays."""

    def __init__(self, settings: Settings):
        super().__init__(id="bay_split")
        self.settings = settings
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from ..agents.llm_caller import create_llm_client
            self._llm = create_llm_client(
                self.settings.azure_openai_endpoint,
                self.settings.azure_openai_api_key,
                self.settings.azure_openai_api_version,
            )
        return self._llm

    @handler
    async def on_approved(self, msg: PanelsCropped, ctx: WorkflowContext) -> None:
        from ..agents.llm_caller import call_llm as _call_llm, parse_json
        from ..agents.prompts import build_bay_prompt

        llm = self._get_llm()
        deploy = self.settings.azure_openai_deployment
        results: List[Dict] = []

        for crop in msg.crops:
            if not crop.crop_path:
                results.append({
                    "panel_name": crop.panel_name,
                    "n_bays": 1,
                    "bay_bboxes": [list(crop.bbox)],
                })
                continue

            import cv2
            img = cv2.imread(crop.crop_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            prompt = build_bay_prompt(crop.panel_name, w, h, self.settings.grid_size)
            raw = _call_llm(
                llm, deploy, prompt,
                image_paths=[crop.crop_path],
                label=f"bay_split {crop.panel_name}",
            )
            result = parse_json(raw)
            if result:
                results.append({
                    "panel_name": crop.panel_name,
                    "n_bays": result.get("n_bays", 1),
                    "bay_bboxes": result.get("bboxes", [[0, 0, w, h]]),
                })
            else:
                results.append({
                    "panel_name": crop.panel_name,
                    "n_bays": 1,
                    "bay_bboxes": [[0, 0, w, h]],
                })

        # Emit final result
        summary = {
            "panels": [
                {
                    "panel_name": crop.panel_name,
                    "bbox": list(crop.bbox),
                    "crop_path": crop.crop_path,
                    "confidence": crop.confidence,
                    "verified_by": crop.verified_by,
                    "bay_info": next(
                        (r for r in results if r["panel_name"] == crop.panel_name),
                        {"n_bays": 1, "bay_bboxes": [list(crop.bbox)]},
                    ),
                }
                for crop in msg.crops
            ]
        }

        # Save summary
        summary_path = Path(self.settings.output_dir) / "final_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[BaySplitEx] Final summary saved: {summary_path}")

        await ctx.send_message(FinalResult(summary=summary))
