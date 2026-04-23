"""
Custom E2E test: Run test.pdf pages 5 & 6 through full pipeline with LLM method.
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from playwright.sync_api import sync_playwright, Page
except ImportError:
    print("ERROR: pip install playwright && playwright install chromium")
    sys.exit(1)

BASE_URL = "http://localhost:8501"
PDF_NAME = "test.pdf"
KEEP_PAGES = {5, 6}  # Only these pages
LLM_TIMEOUT_S = 300  # 5 min per LLM step
SS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "test_p56_screenshots"


class TestPDF56Runner:
    def __init__(self, page: Page):
        self.page = page
        self.timings: dict[str, float] = {}
        self.panel_names: dict[int, list[str]] = {}
        self.panel_counts: dict[int, int] = {}
        self.errors: list[str] = []
        SS_DIR.mkdir(parents=True, exist_ok=True)

    def _ss(self, name: str) -> Path:
        ts = datetime.now().strftime("%H%M%S")
        p = SS_DIR / f"{ts}_{name}.png"
        self.page.screenshot(path=str(p), full_page=True)
        print(f"  📸 {p.name}")
        return p

    def _wait_ready(self, timeout_s: int = 15):
        end = time.time() + timeout_s
        while time.time() < end:
            if self.page.locator('[data-testid="stSpinner"]').count() == 0:
                break
            time.sleep(0.5)
        time.sleep(1)

    def _wait_text(self, text: str, timeout_s: int = 30) -> bool:
        end = time.time() + timeout_s
        while time.time() < end:
            try:
                loc = self.page.get_by_text(text, exact=False)
                if loc.count() > 0 and loc.first.is_visible():
                    return True
            except Exception:
                pass
            time.sleep(1)
        return False

    def _find_button(self, pattern: str, timeout_s: int = 30):
        end = time.time() + timeout_s
        while time.time() < end:
            try:
                btn = self.page.get_by_role("button", name=re.compile(pattern, re.IGNORECASE))
                if btn.count() > 0:
                    try:
                        btn.first.scroll_into_view_if_needed(timeout=3000)
                        time.sleep(0.3)
                    except Exception:
                        pass
                    if btn.first.is_visible():
                        return btn.first
                self.page.evaluate("""
                    window.scrollTo(0, document.body.scrollHeight);
                    const m = document.querySelector('[data-testid="stAppViewContainer"]');
                    if (m) m.scrollTo(0, m.scrollHeight);
                """)
            except Exception:
                pass
            time.sleep(1)
        return None

    def _click_btn(self, pattern: str, timeout_s: int = 30, label: str = ""):
        btn = self._find_button(pattern, timeout_s)
        if not btn:
            raise RuntimeError(f"Button not found: '{label or pattern}'")
        btn.scroll_into_view_if_needed()
        time.sleep(0.3)
        btn.click()
        print(f"  🖱️ Clicked: {label or pattern}")
        time.sleep(1)

    def _wait_llm(self, step_name: str, max_s: int = LLM_TIMEOUT_S):
        print(f"  ⏳ Waiting for {step_name} (up to {max_s}s)...")
        try:
            self.page.wait_for_selector('[data-testid="stSpinner"]', state="visible", timeout=15_000)
            print(f"  ⏳ Spinner appeared")
        except Exception:
            print(f"  ⚠ Spinner may have already gone")
        deadline = time.time() + max_s
        stable = 0
        while time.time() < deadline:
            spinners = self.page.locator('[data-testid="stSpinner"]')
            progress = self.page.locator('[role="progressbar"]')
            if spinners.count() == 0 and progress.count() == 0:
                stable += 1
                if stable >= 3:
                    break
            else:
                stable = 0
            time.sleep(1)
        if stable >= 3:
            print(f"  ✅ {step_name} complete")
        else:
            print(f"  ⚠ Timeout for {step_name}")
            self.errors.append(f"Timeout: {step_name}")
        self._wait_ready()

    # ── STEP 0: Reset & Select PDF ───

    def step0_setup(self):
        print(f"\n{'='*60}\nSTEP 0: Setup — Reset & Select {PDF_NAME}\n{'='*60}")
        t0 = time.time()

        self.page.goto(BASE_URL, wait_until="networkidle")
        self._wait_ready()

        # Click Reset
        sidebar = self.page.locator('[data-testid="stSidebar"]')
        try:
            reset = sidebar.get_by_role("button", name="Reset")
            if reset.is_visible():
                reset.click()
                self._wait_ready(10)
                print("  🔄 Reset done")
        except Exception:
            print("  ⚠ No Reset button found")

        self._ss("00_after_reset")

        # Select test.pdf from dropdown
        dropdown = sidebar.locator('div[data-testid="stSelectbox"]').filter(has_text="Test PDF")
        dropdown.locator('[data-baseweb="select"]').click()
        time.sleep(1)
        self.page.get_by_role("option", name=PDF_NAME, exact=True).click()
        self._wait_ready()
        self._ss("00_selected")
        print(f"  ✅ Selected: {PDF_NAME}")
        self.timings["step0"] = round(time.time() - t0, 1)

    # ── STEP 1: PDF → PNG, select only pages 5 & 6 ───

    def step1_select_pages(self):
        print(f"\n{'='*60}\nSTEP 1: PDF → PNG + Select Pages {KEEP_PAGES}\n{'='*60}")
        t0 = time.time()

        # Click Run Step 1
        self._click_btn(r"Run Step 1.*Upload.*Convert", label="Run Step 1")
        self._wait_llm("PDF conversion", max_s=60)
        self._ss("01_converted")
        assert self._wait_text("Converted", 10), "No 'Converted' text"
        print("  ✅ PDF converted")

        # Click None to deselect all
        self._click_btn(r"^None$", timeout_s=10, label="None")
        self._wait_ready(5)
        print("  ☐ Deselected all pages")

        # Now select only pages 5 and 6
        for pn in sorted(KEEP_PAGES):
            try:
                cb = self.page.locator('[data-testid="stCheckbox"]').filter(
                    has_text=re.compile(rf"^Page {pn}$")
                )
                if cb.count() > 0:
                    inp = cb.first.locator('input[type="checkbox"]')
                    if not inp.is_checked():
                        cb.first.click()
                        self._wait_ready(5)
                        print(f"  ☑ Selected Page {pn}")
                    else:
                        print(f"  ☑ Page {pn} already selected")
            except Exception as e:
                print(f"  ⚠ Could not select Page {pn}: {e}")
                self.errors.append(f"Failed to select page {pn}: {e}")

        self._ss("01_pages_selected")

        # Click Next → Step 2
        self._click_btn(r"Next.*Step 2", label="Next → Step 2")
        self._wait_ready()
        assert self._wait_text("Run Step 2", 15) or self._wait_text("Figure Detection", 5), \
            "Step 2 view did not load"
        self._ss("01_done")
        print("  ✅ Step 1 done")
        self.timings["step1"] = round(time.time() - t0, 1)

    # ── STEP 2: Figure Detection + HITL #1 ───

    def step2_figure_detection(self):
        print(f"\n{'='*60}\nSTEP 2: Figure Detection (DI + LLM)\n{'='*60}")
        t0 = time.time()

        self._ss("02_before")
        self._click_btn(r"Run Step 2.*Figure Detection", label="Run Step 2")
        self._wait_llm("Figure Detection", max_s=300)
        self._ss("02_detected")

        # HITL #1: Confirm regions
        self._click_btn(r"Confirm Regions.*Step 3", timeout_s=30, label="Confirm Regions")
        self._wait_ready()
        self._ss("02_confirmed")

        # Navigate to Step 3
        self._click_btn(r"Next.*Step 3", timeout_s=30, label="Next → Step 3")
        self._wait_ready()
        assert self._wait_text("Step 3", 15), "Step 3 view did not load"
        self._ss("02_done")
        print("  ✅ Step 2 done + HITL #1 confirmed")
        self.timings["step2"] = round(time.time() - t0, 1)

    # ── STEP 3: Panel Names + HITL #2 ───

    def step3_panel_names(self):
        print(f"\n{'='*60}\nSTEP 3: Panel Name Extraction (LLM)\n{'='*60}")
        t0 = time.time()

        self._ss("03_before")
        self._click_btn(r"Run Step 3.*Extract Panel Names", label="Run Step 3")
        self._wait_llm("Panel Name Extraction", max_s=300)
        self._ss("03_extracted")

        # Try to capture panel names from the page
        try:
            page_text = self.page.inner_text("body")
            # Look for panel name patterns in the text
            print(f"\n  === Panel Names Detected ===")
            # Try to find panel name sections
            for pn in sorted(KEEP_PAGES):
                # Look for text like "Page 5:" or "Page 6:" followed by panel names
                pattern = rf"Page\s*{pn}.*?(?:panels?|names?)[:\s]*(.*?)(?:Page\s*\d|$)"
                m = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
                if m:
                    print(f"  Page {pn}: {m.group(1)[:200]}")
        except Exception as e:
            print(f"  ⚠ Could not extract panel names: {e}")

        # HITL #2: Confirm names
        self._click_btn(r"Confirm Names.*Step 4", timeout_s=30, label="Confirm Names")
        self._wait_ready()
        self._ss("03_confirmed")

        # Navigate to Step 4
        self._click_btn(r"Next.*Step 4", timeout_s=30, label="Next → Step 4")
        self._wait_ready()
        assert self._wait_text("Step 4", 15), "Step 4 view did not load"
        self._ss("03_done")
        print("  ✅ Step 3 done + HITL #2 confirmed")
        self.timings["step3"] = round(time.time() - t0, 1)

    # ── STEP 4: Panel Areas with LLM Vision ───

    def step4_panel_areas_llm(self):
        print(f"\n{'='*60}\nSTEP 4: Panel Areas — LLM Vision method\n{'='*60}")
        t0 = time.time()

        self._ss("04_before")

        # Select "LLM Vision (slower, more robust)" radio button
        try:
            radio = self.page.get_by_text("LLM Vision (slower, more robust)")
            if radio.count() > 0:
                radio.first.click()
                time.sleep(1)
                print("  🔘 Selected LLM Vision method")
            else:
                print("  ⚠ LLM Vision radio not found, trying alternative")
                # Try clicking the radio by label
                self.page.locator('label').filter(has_text="LLM Vision").first.click()
                time.sleep(1)
                print("  🔘 Selected LLM Vision method (alt)")
        except Exception as e:
            print(f"  ⚠ Could not select LLM Vision: {e}")
            self.errors.append(f"LLM Vision selection failed: {e}")

        self._ss("04_method_selected")

        # Click Run Step 4
        self._click_btn(r"Run Step 4.*LLM Panel Areas", label="Run Step 4: LLM")
        self._wait_llm("LLM Panel Area Detection", max_s=300)
        self._ss("04_detected")

        # HITL #3: Confirm
        self._click_btn(r"Confirm.*Done|Confirm.*Step 5", timeout_s=60, label="Confirm")
        self._wait_ready()
        self._ss("04_confirmed")

        print("  ✅ Step 4 done + HITL #3 confirmed")
        self.timings["step4"] = round(time.time() - t0, 1)

    # ── Capture final results ───

    def capture_results(self):
        print(f"\n{'='*60}\nCapturing Final Results\n{'='*60}")

        self._wait_ready()

        # Scroll through the page to capture all results
        self._ss("final_results_top")

        # Try to extract panel info from page text
        try:
            page_text = self.page.inner_text("body")
            print(f"\n  === Page Text (excerpt) ===")
            # Find panel-related info
            lines = page_text.split('\n')
            panel_lines = [l.strip() for l in lines if any(
                kw in l.lower() for kw in ['panel', 'page 5', 'page 6', 'crop', 'bay']
            ) and l.strip()]
            for line in panel_lines[:30]:
                print(f"  {line}")
        except Exception as e:
            print(f"  ⚠ Could not read page text: {e}")

        # Take screenshots scrolling down
        for i in range(3):
            self.page.evaluate(f"""
                const h = document.body.scrollHeight;
                window.scrollTo(0, h * {(i+1)/3});
            """)
            time.sleep(1)
            self._ss(f"final_scroll_{i+1}")

        # Look for panel crops tab
        try:
            crops_tab = self.page.get_by_text("Panel Crops", exact=False)
            if crops_tab.count() > 0:
                crops_tab.first.click()
                time.sleep(2)
                self._wait_ready()
                self._ss("panel_crops")
                print("  📸 Panel Crops tab captured")
        except Exception:
            print("  ⚠ Panel Crops tab not found")

        # Check checkpoints/test for saved results
        print("\n  === Checking saved checkpoint data ===")

    def run(self):
        start = time.time()
        print(f"\n{'#'*60}")
        print(f"# test.pdf Pages 5 & 6 — LLM Pipeline")
        print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}")

        try:
            self.step0_setup()
            self.step1_select_pages()
            self.step2_figure_detection()
            self.step3_panel_names()
            self.step4_panel_areas_llm()
            self.capture_results()

            elapsed = time.time() - start
            print(f"\n{'='*60}")
            print(f"ALL STEPS COMPLETE ✅  ({elapsed:.0f}s total)")
            print(f"{'='*60}")

        except Exception as e:
            self._ss("ERROR")
            elapsed = time.time() - start
            print(f"\n{'='*60}")
            print(f"FAILED ❌ ({elapsed:.0f}s): {e}")
            print(f"{'='*60}")
            self.errors.append(str(e))
            import traceback
            traceback.print_exc()

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Timings: {json.dumps(self.timings, indent=2)}")
        print(f"Errors:  {self.errors if self.errors else 'None'}")
        print(f"Screenshots saved to: {SS_DIR}")

        # Save results
        results = {
            "pdf": PDF_NAME,
            "pages": sorted(KEEP_PAGES),
            "method": "LLM Vision",
            "timings": self.timings,
            "errors": self.errors,
            "total_seconds": round(time.time() - start, 1),
            "timestamp": datetime.now().isoformat(),
        }
        rp = SS_DIR / "results.json"
        rp.write_text(json.dumps(results, indent=2))
        print(f"Results: {rp}")

        return len(self.errors) == 0


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            ignore_https_errors=True,
        )
        page = ctx.new_page()
        page.set_default_timeout(30_000)

        runner = TestPDF56Runner(page)
        ok = runner.run()

        browser.close()
        return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
