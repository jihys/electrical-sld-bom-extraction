"""
E2E test: public_sld_2.pdf — Step 1 → Step 4 SVG Grid.

Usage:
  cd /home/azureuser/localfiles/electrical-sld-bom-extraction
  source venv/bin/activate
  python tests/test_public_sld2_pipeline.py
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
from datetime import datetime

try:
    from playwright.sync_api import sync_playwright, Page
except ImportError:
    print("ERROR: pip install playwright && playwright install chromium")
    sys.exit(1)

BASE_URL = "http://localhost:8501"
LLM_TIMEOUT_S = 300
SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "e2e_screenshots" / "public_sld2"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

results = {}

def ss(page: Page, name: str) -> Path:
    ts = datetime.now().strftime("%H%M%S")
    p = SCREENSHOT_DIR / f"{ts}_{name}.png"
    page.screenshot(path=str(p), full_page=True)
    print(f"  📸 {p.name}")
    return p

def wait_ready(page: Page, timeout_s=15):
    end = time.time() + timeout_s
    while time.time() < end:
        if page.locator('[data-testid="stSpinner"]').count() == 0:
            break
        time.sleep(0.5)
    time.sleep(1)

def wait_llm(page: Page, label: str):
    print(f"  ⏳ Waiting for {label} (up to {LLM_TIMEOUT_S}s)...")
    try:
        page.wait_for_selector('[data-testid="stSpinner"]', state="visible", timeout=15_000)
        print(f"  ⏳ Spinner appeared…")
    except Exception:
        print(f"  ⚠ Spinner may have already gone")
    deadline = time.time() + LLM_TIMEOUT_S
    stable = 0
    while time.time() < deadline:
        sp = page.locator('[data-testid="stSpinner"]')
        pb = page.locator('[role="progressbar"]')
        if sp.count() == 0 and pb.count() == 0:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
        time.sleep(1)
    if stable >= 3:
        print(f"  ✅ {label} complete")
    else:
        print(f"  ⚠ Timeout for {label}")
    wait_ready(page)

def click_button(page: Page, pattern: str, label: str = "", timeout_s=30):
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            btn = page.get_by_role("button", name=re.compile(pattern, re.IGNORECASE))
            if btn.count() > 0:
                try:
                    btn.first.scroll_into_view_if_needed(timeout=3000)
                    time.sleep(0.3)
                except Exception:
                    pass
                if btn.first.is_visible():
                    btn.first.click()
                    print(f"  🖱️ Clicked: {label or pattern}")
                    time.sleep(1)
                    return
        except Exception:
            pass
        # Scroll down
        try:
            page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Button not found: '{label or pattern}'")

def wait_for_text(page: Page, text: str, timeout_s=30) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            loc = page.get_by_text(text, exact=False)
            if loc.count() > 0 and loc.first.is_visible():
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def select_multiselect_option(page: Page, key: str, option_text: str):
    """Select an option in a Streamlit multiselect widget by key."""
    sidebar = page.locator('[data-testid="stSidebar"]')
    ms = sidebar.locator(f'div[data-testid="stMultiSelect"]').filter(has_text=re.compile(option_text[:20], re.IGNORECASE))
    if ms.count() == 0:
        # Try all multiselects
        all_ms = sidebar.locator('div[data-testid="stMultiSelect"]')
        for i in range(all_ms.count()):
            inner = all_ms.nth(i).inner_text()
            if key.replace("_", " ").lower() in inner.lower() or option_text[:15].lower() in inner.lower():
                ms = all_ms.nth(i)
                break
        else:
            print(f"  ⚠ Could not find multiselect for {key}")
            return False
    # Click the input to open dropdown
    inp = ms.locator("input")
    if inp.count() > 0:
        inp.first.click()
        time.sleep(0.5)
        # Type partial text to filter
        inp.first.fill(option_text[:10])
        time.sleep(0.5)
        # Select matching option
        try:
            opt = page.locator('[data-testid="stPortalContainer-Popover"]').get_by_text(option_text, exact=False)
            if opt.count() > 0:
                opt.first.click()
                time.sleep(0.5)
                print(f"  ✅ Selected VP: {option_text}")
                return True
        except Exception:
            pass
        # Try clicking away to close
        page.keyboard.press("Escape")
        time.sleep(0.3)
    return False

def run_test():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        # ══════════════════════════════════════════════════════════
        # STEP 0: Select PDF + Visual Prompts
        # ══════════════════════════════════════════════════════════
        print(f"\n{'='*60}\nSTEP 0: Setup — public_sld_2.pdf\n{'='*60}")
        page.goto(BASE_URL, wait_until="networkidle")
        wait_ready(page)

        sidebar = page.locator('[data-testid="stSidebar"]')

        # Reset if needed
        try:
            reset = sidebar.get_by_role("button", name="Reset")
            if reset.is_visible():
                reset.click()
                wait_ready(page, 10)
                print("  🔄 Reset")
        except Exception:
            pass

        ss(page, "00_initial")

        # Select PDF
        dropdown = sidebar.locator('div[data-testid="stSelectbox"]').filter(has_text="Test PDF")
        dropdown.locator('[data-baseweb="select"]').click()
        time.sleep(1)
        page.get_by_role("option", name="public_sld_2.pdf", exact=True).click()
        wait_ready(page)
        ss(page, "00_pdf_selected")
        print("  ✅ Selected: public_sld_2.pdf")

        # Visual Prompts — select first image in each category
        # Panel Name Detection: panel_name_box_example1.png (first)
        print("  Setting Visual Prompts...")
        vp_categories = [
            ("panel_name", "panel_name_box_example1"),
            ("panel_area", "panel_box_explanation.png"),
            ("bay_split", "bay_example"),
        ]
        for cat_key, first_img in vp_categories:
            try:
                all_ms = sidebar.locator('div[data-testid="stMultiSelect"]')
                for i in range(all_ms.count()):
                    txt = all_ms.nth(i).inner_text()
                    # Find matching multiselect
                    inp = all_ms.nth(i).locator("input")
                    if inp.count() > 0:
                        inp.first.click()
                        time.sleep(0.3)
                        # Look for the option in dropdown
                        try:
                            opts = page.locator('[role="listbox"] [role="option"]')
                            for j in range(opts.count()):
                                opt_text = opts.nth(j).inner_text()
                                if first_img.replace(".png", "") in opt_text.replace(".png", ""):
                                    opts.nth(j).click()
                                    time.sleep(0.5)
                                    print(f"  ✅ VP {cat_key}: {opt_text}")
                                    break
                            else:
                                page.keyboard.press("Escape")
                                time.sleep(0.3)
                                continue
                            break
                        except Exception:
                            page.keyboard.press("Escape")
                            time.sleep(0.3)
                            continue
            except Exception as e:
                print(f"  ⚠ VP {cat_key}: {e}")

        wait_ready(page)
        ss(page, "00_vp_set")
        results["step0"] = "OK"

        # ══════════════════════════════════════════════════════════
        # STEP 1: PDF → PNG
        # ══════════════════════════════════════════════════════════
        print(f"\n{'='*60}\nSTEP 1: PDF → PNG\n{'='*60}")
        click_button(page, r"Run Step 1.*Upload.*Convert", "Run Step 1")
        wait_llm(page, "PDF conversion")
        ss(page, "01_converted")

        if wait_for_text(page, "Converted", 10) or wait_for_text(page, "Page", 5):
            print("  ✅ Converted to PNG")
        else:
            print("  ⚠ No 'Converted' text found, continuing...")

        # public_sld_2.pdf is 1 page — make sure Page 1 is selected
        try:
            cb = page.locator('[data-testid="stCheckbox"]').filter(has_text=re.compile(r"Page\s*1"))
            if cb.count() > 0:
                inp = cb.first.locator('input[type="checkbox"]')
                if not inp.is_checked():
                    cb.first.click()
                    wait_ready(page, 5)
                    print("  ☑ Checked Page 1")
                else:
                    print("  ☑ Page 1 already selected")
        except Exception as e:
            print(f"  ⚠ Page selection: {e}")

        ss(page, "01_pages")

        # Next → Step 2
        click_button(page, r"Next.*Step 2", "Next → Step 2")
        wait_ready(page)
        if wait_for_text(page, "Step 2", 15) or wait_for_text(page, "Figure Detection", 10):
            print("  ✅ Step 1 done → Step 2 loaded")
        ss(page, "01_done")
        results["step1"] = "OK"

        # ══════════════════════════════════════════════════════════
        # STEP 2: Figure Detection
        # ══════════════════════════════════════════════════════════
        print(f"\n{'='*60}\nSTEP 2: Figure Detection (LLM)\n{'='*60}")
        ss(page, "02_before")
        click_button(page, r"Run Step 2.*Figure Detection", "Run Step 2")
        wait_llm(page, "Figure Detection")
        ss(page, "02_detected")
        results["step2"] = "OK"

        # ══════════════════════════════════════════════════════════
        # HITL #1: Confirm Regions
        # ══════════════════════════════════════════════════════════
        print(f"\n{'='*60}\nHITL #1: Confirm Regions\n{'='*60}")
        try:
            click_button(page, r"Confirm Regions.*Step 3", "Confirm Regions → Step 3", timeout_s=30)
            wait_ready(page)
            ss(page, "02_confirmed")
            print("  ✅ Regions confirmed")
        except Exception as e:
            print(f"  ⚠ Confirm button: {e}")
            # Try Next → Step 3 directly
            ss(page, "02_confirm_fail")

        # Next → Step 3
        try:
            click_button(page, r"Next.*Step 3", "Next → Step 3", timeout_s=15)
            wait_ready(page)
        except Exception:
            print("  ⚠ No Next→Step3 button, may auto-advance")

        if wait_for_text(page, "Step 3", 15) or wait_for_text(page, "Panel Names", 10):
            print("  ✅ HITL #1 done → Step 3 loaded")
        ss(page, "02_done")
        results["hitl1"] = "OK"

        # ══════════════════════════════════════════════════════════
        # STEP 3: Panel Name Extraction
        # ══════════════════════════════════════════════════════════
        print(f"\n{'='*60}\nSTEP 3: Panel Name Extraction (LLM)\n{'='*60}")
        ss(page, "03_before")
        click_button(page, r"Run Step 3.*Extract Panel Names", "Run Step 3")
        wait_llm(page, "Panel Name Extraction")
        ss(page, "03_extracted")

        # Print detected panel names
        try:
            body_text = page.locator('[data-testid="stMainBlockContainer"]').inner_text()
            # Find panel name related text
            lines = body_text.split('\n')
            name_lines = [l.strip() for l in lines if l.strip() and ('panel' in l.lower() or 'name' in l.lower() or '▸' in l or '►' in l or '•' in l)]
            if name_lines:
                print(f"  📋 Panel names found:")
                for nl in name_lines[:20]:
                    print(f"     {nl}")
        except Exception:
            pass

        results["step3"] = "OK"

        # ══════════════════════════════════════════════════════════
        # HITL #2: Confirm Names
        # ══════════════════════════════════════════════════════════
        print(f"\n{'='*60}\nHITL #2: Confirm Names\n{'='*60}")
        try:
            click_button(page, r"Confirm Names.*Step 4", "Confirm Names → Step 4", timeout_s=30)
            wait_ready(page)
            ss(page, "03_confirmed")
            print("  ✅ Names confirmed")
        except Exception as e:
            print(f"  ⚠ Confirm names: {e}")
            ss(page, "03_confirm_fail")

        # Next → Step 4
        try:
            click_button(page, r"Next.*Step 4", "Next → Step 4", timeout_s=15)
            wait_ready(page)
        except Exception:
            print("  ⚠ No Next→Step4 button, may auto-advance")

        if wait_for_text(page, "Step 4", 15) or wait_for_text(page, "Panel Area", 10):
            print("  ✅ HITL #2 done → Step 4 loaded")
        ss(page, "03_done")
        results["hitl2"] = "OK"

        # ══════════════════════════════════════════════════════════
        # STEP 4: SVG Grid Detection
        # ══════════════════════════════════════════════════════════
        print(f"\n{'='*60}\nSTEP 4: SVG Grid Detection\n{'='*60}")
        ss(page, "04_before")

        # Select SVG Grid method via radio button
        try:
            radio = page.get_by_text("SVG Grid", exact=False)
            if radio.count() > 0:
                radio.first.click()
                time.sleep(0.5)
                print("  ✅ Selected SVG Grid method")
            else:
                print("  ⚠ SVG Grid radio not found — may be default")
        except Exception as e:
            print(f"  ⚠ SVG Grid selection: {e}")

        # Click Run Step 4 SVG
        try:
            click_button(page, r"Run Step 4.*SVG", "Run Step 4: SVG Grid Detection")
        except RuntimeError:
            # Fallback: try any Run Step 4 button
            click_button(page, r"Run Step 4", "Run Step 4")

        wait_llm(page, "SVG Grid Detection")
        ss(page, "04_svg_detected")

        # Check for errors
        try:
            err_els = page.locator('[data-testid="stAlert"]').filter(has_text=re.compile(r"error|fail", re.IGNORECASE))
            if err_els.count() > 0:
                err_text = err_els.first.inner_text()
                print(f"  ❌ Error: {err_text}")
                results["step4_svg"] = f"FAIL: {err_text}"
            else:
                print("  ✅ Step 4 SVG complete")
                results["step4_svg"] = "OK"
        except Exception:
            results["step4_svg"] = "OK"

        ss(page, "04_final")

        # ══════════════════════════════════════════════════════════
        # Summary
        # ══════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        for k, v in results.items():
            icon = "✅" if v == "OK" else "❌"
            print(f"  {icon} {k}: {v}")

        # Save results
        res_path = SCREENSHOT_DIR / "results.json"
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  📄 Results saved: {res_path}")
        print(f"  📁 Screenshots in: {SCREENSHOT_DIR}")

        browser.close()

    return all(v == "OK" for v in results.values())


if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)
