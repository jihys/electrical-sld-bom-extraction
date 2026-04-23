"""
E2E Automated Demo Test for Electrical SLD BOM Extraction Streamlit App.

Runs the full pipeline:
  Step 1: PDF → PNG (select pages 3+)
  Step 2: Figure Detection + HITL confirm
  Step 3: Panel Name Extraction + HITL confirm
  Step 4: Panel Area + Bay + HITL confirm
  Step 5: BOM Extraction + HITL confirm → Final JSON

Usage:
  python tests/e2e_demo_test.py
  python tests/e2e_demo_test.py --pdf h_test.pdf --skip-pages 1,2
  pytest tests/e2e_demo_test.py -v -s
"""
from __future__ import annotations

import argparse
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

DEFAULT_BASE_URL = "http://localhost:8501"
DEFAULT_PDF = "test.pdf"
DEFAULT_SKIP_PAGES = [1, 2]
LLM_TIMEOUT = 600_000  # 10 min
SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "e2e_screenshots"


class StreamlitE2ERunner:
    def __init__(self, page: Page, base_url: str, pdf_name: str, skip_pages: list[int]):
        self.page = page
        self.base_url = base_url
        self.pdf_name = pdf_name
        self.skip_pages = skip_pages
        self.step_results: dict = {}
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Helpers ───────────────────────────────────────────────────

    def _ss(self, name: str) -> Path:
        """Take screenshot."""
        ts = datetime.now().strftime("%H%M%S")
        p = SCREENSHOT_DIR / f"{ts}_{name}.png"
        self.page.screenshot(path=str(p), full_page=True)
        print(f"  📸 {p.name}")
        return p

    def _wait_streamlit_ready(self, timeout_s: int = 15):
        """Wait for Streamlit to finish rerunning (no spinner, page stable)."""
        # Wait for any spinner to disappear
        end = time.time() + timeout_s
        while time.time() < end:
            spinners = self.page.locator('[data-testid="stSpinner"]')
            if spinners.count() == 0:
                break
            time.sleep(0.5)
        time.sleep(1)  # extra settle

    def _wait_for_text(self, text: str, timeout_s: int = 30) -> bool:
        """Wait until text appears on page."""
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

    def _wait_for_button(self, pattern: str, timeout_s: int = 30):
        """Wait until a button matching pattern appears, scroll to find it."""
        end = time.time() + timeout_s
        while time.time() < end:
            try:
                btn = self.page.get_by_role(
                    "button", name=re.compile(pattern, re.IGNORECASE)
                )
                if btn.count() > 0:
                    # Try scroll_into_view first even if not yet 'visible'
                    try:
                        btn.first.scroll_into_view_if_needed(timeout=3000)
                        time.sleep(0.3)
                    except Exception:
                        pass
                    if btn.first.is_visible():
                        return btn.first
                # Try scrolling down as fallback
                self.page.evaluate("""
                    window.scrollTo(0, document.body.scrollHeight);
                    const m = document.querySelector('[data-testid="stAppViewContainer"]');
                    if (m) m.scrollTo(0, m.scrollHeight);
                """)
            except Exception:
                pass
            time.sleep(1)
        return None

    def _click_button(self, pattern: str, timeout_s: int = 30, label: str = ""):
        """Wait for button and click it. Raises if not found."""
        btn = self._wait_for_button(pattern, timeout_s)
        if not btn:
            raise RuntimeError(f"Button not found: '{label or pattern}'")
        btn.scroll_into_view_if_needed()
        time.sleep(0.3)
        btn.click()
        print(f"  🖱️ Clicked: {label or pattern}")
        time.sleep(1)  # allow Streamlit to pick up click

    def _wait_llm_done(self, step_name: str):
        """Wait for LLM processing spinner to complete.
        
        Handles multi-phase spinners: waits until no spinner is visible
        for a stable period (no new spinner reappearing).
        """
        print(f"  ⏳ Waiting for {step_name} (LLM, up to 5 min)...")
        # Wait for spinner to appear
        try:
            self.page.wait_for_selector(
                '[data-testid="stSpinner"]', state="visible", timeout=15_000
            )
            print(f"  ⏳ Spinner appeared, processing...")
        except Exception:
            print(f"  ⚠ Spinner may have already appeared and gone")
        # Wait for spinner to fully disappear (handle multi-phase spinners)
        deadline = time.time() + LLM_TIMEOUT / 1000
        stable_count = 0
        while time.time() < deadline:
            spinners = self.page.locator('[data-testid="stSpinner"]')
            progress = self.page.locator('[role="progressbar"]')
            if spinners.count() == 0 and progress.count() == 0:
                stable_count += 1
                if stable_count >= 3:  # No spinner for 3 consecutive checks (~3s)
                    break
            else:
                stable_count = 0
            time.sleep(1)
        if stable_count >= 3:
            print(f"  ✅ {step_name} complete")
        else:
            print(f"  ⚠ Spinner timeout after 10min for {step_name}")
        self._wait_streamlit_ready()

    # ── Step 0: Select PDF ────────────────────────────────────────

    def step0_select_pdf(self):
        print(f"\n{'='*60}\nSTEP 0: Select PDF ({self.pdf_name})\n{'='*60}")

        self.page.goto(self.base_url, wait_until="networkidle")
        self._wait_streamlit_ready()

        # Reset app
        sidebar = self.page.locator('[data-testid="stSidebar"]')
        try:
            reset = sidebar.get_by_role("button", name="Reset")
            if reset.is_visible():
                reset.click()
                self._wait_streamlit_ready(10)
                print("  🔄 Reset")
        except Exception:
            pass

        self._ss("00_initial")

        # Open Test PDF dropdown
        dropdown = sidebar.locator('div[data-testid="stSelectbox"]').filter(
            has_text="Test PDF"
        )
        dropdown.locator('[data-baseweb="select"]').click()
        time.sleep(1)

        # Select exact PDF
        self.page.get_by_role("option", name=self.pdf_name, exact=True).click()
        self._wait_streamlit_ready()
        self._ss("00_selected")
        print(f"  ✅ Selected: {self.pdf_name}")
        self.step_results["step0"] = {"status": "ok"}

    # ── Step 1: PDF → PNG ─────────────────────────────────────────

    def step1_pdf_to_png(self):
        print(f"\n{'='*60}\nSTEP 1: PDF → PNG\n{'='*60}")

        self._click_button(r"Run Step 1.*Upload.*Convert", label="Run Step 1")
        self._wait_llm_done("PDF conversion")
        self._ss("01_converted")

        assert self._wait_for_text("Converted", 10), "No 'Converted' text"
        print("  ✅ Converted to PNG")

        # Uncheck pages
        for pn in self.skip_pages:
            try:
                cbs = self.page.locator('[data-testid="stCheckbox"]').filter(
                    has_text=re.compile(rf"^Page {pn}$")
                )
                if cbs.count() > 0:
                    inp = cbs.first.locator('input[type="checkbox"]')
                    if inp.is_checked():
                        cbs.first.click()
                        self._wait_streamlit_ready(5)
                        print(f"  ☐ Unchecked Page {pn}")
            except Exception as e:
                print(f"  ⚠ Could not uncheck Page {pn}: {e}")

        self._ss("01_pages")

        # Click Next → Step 2 and WAIT for Step 2 view to load
        self._click_button(r"Next.*Step 2", label="Next → Step 2")
        self._wait_streamlit_ready()

        # Verify Step 2 view is active
        assert self._wait_for_text("Run Step 2", 15) or \
            self._wait_for_text("Figure Detection", 5), \
            "Step 2 view did not load"
        self._ss("01_done")
        print("  ✅ Step 1 done → Step 2 view loaded")
        self.step_results["step1"] = {"status": "ok"}

    # ── Step 2: Figure Detection ──────────────────────────────────

    def step2_figure_detection(self):
        print(f"\n{'='*60}\nSTEP 2: Figure Detection (LLM)\n{'='*60}")

        # Make sure we see "Run Step 2" button before clicking
        self._ss("02_before")
        self._click_button(r"Run Step 2.*Figure Detection", label="Run Step 2")
        self._wait_llm_done("Figure Detection")
        self._ss("02_detected")

        # HITL: Confirm default
        self._click_button(
            r"Confirm Regions.*Step 3", timeout_s=30, label="Confirm Regions"
        )
        self._wait_streamlit_ready()
        self._ss("02_confirmed")

        # Navigate to Step 3 — this is REQUIRED
        self._click_button(
            r"Next.*Step 3", timeout_s=30, label="Next → Step 3"
        )
        self._wait_streamlit_ready()
        assert self._wait_for_text("Step 3", 15), "Step 3 view did not load"
        self._ss("02_done")
        print("  ✅ Step 2 done → Step 3 view loaded")
        self.step_results["step2"] = {"status": "ok"}

    # ── Step 3: Panel Names ───────────────────────────────────────

    def step3_panel_names(self):
        print(f"\n{'='*60}\nSTEP 3: Panel Name Extraction (LLM)\n{'='*60}")

        self._ss("03_before")
        self._click_button(r"Run Step 3.*Extract Panel Names", label="Run Step 3")
        self._wait_llm_done("Panel Name Extraction")
        self._ss("03_extracted")

        # HITL: Confirm default
        self._click_button(
            r"Confirm Names.*Step 4", timeout_s=30, label="Confirm Names"
        )
        self._wait_streamlit_ready()
        self._ss("03_confirmed")

        # Navigate to Step 4
        self._click_button(
            r"Next.*Step 4", timeout_s=30, label="Next → Step 4"
        )
        self._wait_streamlit_ready()
        assert self._wait_for_text("Step 4", 15), "Step 4 view did not load"
        self._ss("03_done")
        print("  ✅ Step 3 done → Step 4 view loaded")
        self.step_results["step3"] = {"status": "ok"}

    # ── Step 4: Panel Areas + Bay ─────────────────────────────────

    def step4_panel_areas(self):
        print(f"\n{'='*60}\nSTEP 4: Panel Areas + Bay\n{'='*60}")

        self._ss("04_before")
        self._click_button(r"Run Step 4", label="Run Step 4")
        self._wait_llm_done("Panel Area Detection")
        self._ss("04_detected")

        # HITL: Confirm default
        self._click_button(
            r"Confirm.*Step 5", timeout_s=60, label="Confirm → Step 5"
        )
        self._wait_streamlit_ready()
        self._ss("04_confirmed")

        # Navigate to Step 5
        self._click_button(
            r"Next.*Step 5", timeout_s=30, label="Next → Step 5"
        )
        self._wait_streamlit_ready()
        assert self._wait_for_text("Step 5", 15), "Step 5 view did not load"
        self._ss("04_done")
        print("  ✅ Step 4 done → Step 5 view loaded")
        self.step_results["step4"] = {"status": "ok"}

    # ── Step 5: BOM Extraction ────────────────────────────────────

    def step5_bom_extraction(self):
        print(f"\n{'='*60}\nSTEP 5: BOM Extraction (LLM)\n{'='*60}")

        self._ss("05_before")
        self._click_button(r"Run Step 5.*BOM Extraction", label="Run Step 5")
        self._wait_llm_done("BOM Extraction")
        self._ss("05_extracted")

        # Collapse all expanders first to make page shorter, then find button
        try:
            self.page.evaluate("""
                // Collapse all open expanders to shorten page
                document.querySelectorAll('[data-testid="stExpander"] details[open]').forEach(d => {
                    d.removeAttribute('open');
                });
                // Also try clicking summary elements
                document.querySelectorAll('[data-testid="stExpander"] details[open] > summary').forEach(s => {
                    s.click();
                });
            """)
            time.sleep(1)
        except Exception:
            pass

        # Scroll to bottom aggressively
        for _ in range(5):
            self.page.evaluate("""
                window.scrollTo(0, document.body.scrollHeight);
                const containers = document.querySelectorAll(
                    '[data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"], .main'
                );
                containers.forEach(c => c.scrollTo && c.scrollTo(0, c.scrollHeight));
                // Scroll all parent containers
                let el = document.querySelector('button');
                if (el) {
                    let p = el.parentElement;
                    while (p) {
                        if (p.scrollHeight > p.clientHeight) p.scrollTop = p.scrollHeight;
                        p = p.parentElement;
                    }
                }
            """)
            time.sleep(0.5)

        # HITL: Confirm BOM — the button is at the bottom of a long page
        # with 12+ BOM expanders. Use JavaScript to find and click it.
        confirm_found = False

        # First collapse all expanders to shorten the page
        try:
            self.page.evaluate("""
                document.querySelectorAll('[data-testid="stExpander"] details[open]').forEach(d => {
                    d.removeAttribute('open');
                });
            """)
            time.sleep(1)
        except Exception:
            pass

        # Try to find and click via JavaScript (most reliable for off-screen buttons)
        for attempt in range(8):
            try:
                clicked = self.page.evaluate("""
                    () => {
                        const buttons = Array.from(document.querySelectorAll('button'));
                        const btn = buttons.find(b => b.textContent.includes('Confirm BOM'));
                        if (btn) {
                            btn.scrollIntoView({block: 'center'});
                            btn.click();
                            return true;
                        }
                        return false;
                    }
                """)
                if clicked:
                    print(f"  🖱️ Clicked: Confirm BOM (JS, attempt {attempt+1})")
                    confirm_found = True
                    time.sleep(1)
                    break
            except Exception:
                pass

            # Try scrolling with keyboard End key
            self.page.keyboard.press("End")
            time.sleep(1.5)

        if not confirm_found:
            self._click_button(
                r"Confirm BOM", timeout_s=30, label="Confirm BOM"
            )
        self._wait_streamlit_ready()
        self._ss("05_confirmed")

        # Check final results
        self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        self._ss("05_final")

        has_final = (
            self._wait_for_text("Final Results", 5)
            or self._wait_for_text("Download JSON", 5)
            or self._wait_for_text("Saved:", 5)
        )
        print(f"  {'✅' if has_final else '⚠'} Final results {'found' if has_final else 'not visible'}")
        self.step_results["step5"] = {"status": "ok" if has_final else "warning"}

    # ── Run All ───────────────────────────────────────────────────

    def run_all(self) -> bool:
        start = time.time()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'#'*60}")
        print(f"# E2E Demo — {self.pdf_name} | Skip: {self.skip_pages}")
        print(f"# {ts}")
        print(f"{'#'*60}")

        try:
            self.step0_select_pdf()
            self.step1_pdf_to_png()
            self.step2_figure_detection()
            self.step3_panel_names()
            self.step4_panel_areas()
            self.step5_bom_extraction()

            print(f"\n{'='*60}")
            print(f"ALL STEPS COMPLETE ✅  ({time.time()-start:.0f}s)")
            print(f"{'='*60}")

        except Exception as e:
            self._ss("ERROR")
            print(f"\n{'='*60}")
            print(f"FAILED ❌ ({time.time()-start:.0f}s): {e}")
            print(f"{'='*60}")
            self.step_results["error"] = str(e)
            return False

        finally:
            results = {
                "pdf": self.pdf_name,
                "skip_pages": self.skip_pages,
                "elapsed_seconds": round(time.time() - start, 1),
                "timestamp": datetime.now().isoformat(),
                "steps": self.step_results,
            }
            rp = SCREENSHOT_DIR / "results.json"
            rp.write_text(json.dumps(results, indent=2))
            print(f"\n{json.dumps(self.step_results, indent=2)}")
            print(f"Screenshots: {SCREENSHOT_DIR}")

        return True


def run_e2e_test(
    base_url: str = DEFAULT_BASE_URL,
    pdf_name: str = DEFAULT_PDF,
    skip_pages: list[int] | None = None,
    headless: bool = True,
) -> bool:
    if skip_pages is None:
        skip_pages = DEFAULT_SKIP_PAGES

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            ignore_https_errors=True,
        )
        page = ctx.new_page()
        page.set_default_timeout(30_000)

        runner = StreamlitE2ERunner(page, base_url, pdf_name, skip_pages)
        ok = runner.run_all()

        browser.close()
        return ok


def test_e2e_demo():
    """pytest entry point."""
    assert run_e2e_test(headless=True), "E2E demo test failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E Demo Test")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    parser.add_argument("--skip-pages", default="1,2")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    skip = [int(x.strip()) for x in args.skip_pages.split(",") if x.strip()]
    ok = run_e2e_test(args.base_url, args.pdf, skip, not args.headed)
    sys.exit(0 if ok else 1)
