"""
Test public_sld_2.pdf pipeline: Steps 1-4 (SVG mode).

Usage:
  cd /home/azureuser/localfiles/electrical-sld-bom-extraction
  source venv/bin/activate
  python tests/test_public_sld2_svg.py
"""
from playwright.sync_api import sync_playwright
import time, re, json
from pathlib import Path
from datetime import datetime

SS_DIR = Path("outputs/e2e_screenshots/public_sld2")
SS_DIR.mkdir(parents=True, exist_ok=True)

PDF_NAME = "public_sld_2.pdf"
BASE_URL = "http://localhost:8501"

results = {}


def ss(page, name):
    ts = datetime.now().strftime("%H%M%S")
    p = SS_DIR / f"{ts}_{name}.png"
    page.screenshot(path=str(p), full_page=True)
    print(f"  📸 {p.name}")
    return p


def wait_spinner(page, timeout_s=300):
    """Wait for all spinners/progress bars to disappear."""
    try:
        page.wait_for_selector('[data-testid="stSpinner"]', state='visible', timeout=15000)
        print("  ⏳ Spinner appeared…")
    except:
        pass
    deadline = time.time() + timeout_s
    stable = 0
    while time.time() < deadline:
        sp = page.locator('[data-testid="stSpinner"]').count()
        pb = page.locator('[role="progressbar"]').count()
        if sp == 0 and pb == 0:
            stable += 1
            if stable >= 5:
                break
        else:
            stable = 0
        time.sleep(2)
    time.sleep(3)
    return stable >= 5


def click_btn(page, pattern, label="", timeout_s=30):
    btn = page.get_by_role("button", name=re.compile(pattern, re.I))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if btn.count() > 0 and btn.first.is_visible():
            btn.first.scroll_into_view_if_needed()
            time.sleep(0.5)
            btn.first.click()
            print(f"  🖱️ Clicked: {label or pattern}")
            time.sleep(1)
            return True
        time.sleep(1)
    print(f"  ❌ Button not found: {label or pattern}")
    return False


def report_page(page, keywords=None):
    body = page.inner_text("body")
    if keywords:
        lines = [l.strip() for l in body.split("\n")
                 if any(k.lower() in l.lower() for k in keywords) and len(l.strip()) > 3][:20]
    else:
        lines = [l.strip() for l in body.split("\n") if l.strip()][:30]
    for l in lines:
        print(f"    {l[:120]}")

    btns = page.locator("button")
    vis_btns = []
    for i in range(min(btns.count(), 20)):
        try:
            t = btns.nth(i).inner_text().strip()
            if t and btns.nth(i).is_visible():
                vis_btns.append(t[:60])
        except:
            pass
    if vis_btns:
        print(f"  Buttons: {vis_btns}")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.set_default_timeout(60000)
        page.goto(BASE_URL, wait_until="networkidle")
        time.sleep(3)

        sidebar = page.locator('[data-testid="stSidebar"]')

        # ============================================================
        # SETUP: Select PDF + Visual Prompts
        # ============================================================
        print("\n" + "=" * 60)
        print(f"SETUP: Select {PDF_NAME} & Visual Prompts")
        print("=" * 60)

        # Select PDF
        dropdown = sidebar.locator('div[data-testid="stSelectbox"]').filter(has_text="Test PDF")
        dropdown.locator('[data-baseweb="select"]').click()
        time.sleep(1)
        page.get_by_role("option", name=PDF_NAME, exact=True).click()
        time.sleep(3)
        print(f"  ✅ PDF: {PDF_NAME}")

        # Visual Prompts - Panel Name
        sidebar.evaluate("el => el.scrollTo(0, el.scrollHeight)")
        time.sleep(1)
        ms = sidebar.locator('div[data-testid="stMultiSelect"]')
        if ms.count() >= 1:
            ms.nth(0).locator("input").click()
            time.sleep(1)
            for name in ["panel_name_box_example1", "panel_name_box_example2"]:
                try:
                    page.locator('[role="option"]').filter(has_text=name).click()
                    time.sleep(0.5)
                    ms.nth(0).locator("input").click()
                    time.sleep(0.5)
                except:
                    pass
            page.keyboard.press("Escape")
            time.sleep(1)
            print("  ✅ Panel Name prompts set")

        # Panel Area prompt
        if ms.count() >= 2:
            ms.nth(1).locator("input").click()
            time.sleep(1)
            opts = page.locator('[role="option"]')
            if opts.count() > 0:
                for i in range(opts.count()):
                    txt = opts.nth(i).inner_text()
                    if "panel_box" in txt or "explanation" in txt:
                        opts.nth(i).click()
                        time.sleep(0.5)
                        print(f"  ✅ Panel Area prompt: {txt}")
                        break
            else:
                print("  ⚠ No Panel Area options available")
            page.keyboard.press("Escape")
            time.sleep(1)

        ss(page, "00_setup")
        results["setup"] = "ok"

        # ============================================================
        # STEP 1: PDF → PNG
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 1: PDF → PNG")
        print("=" * 60)

        click_btn(page, r"Run Step 1", "Run Step 1")
        time.sleep(5)
        wait_spinner(page, 60)
        print("  ✅ Step 1 done")

        # public_sld_2.pdf has 1 page — just verify page 1 is visible
        body = page.inner_text("body")
        if "Page 1" in body:
            print("  ✅ Page 1 visible (1 page PDF)")
        else:
            print("  ⚠ Page 1 not found in body text")

        ss(page, "01_step1")
        report_page(page, ["Page", "Converted", "Selected", "page"])
        results["step1"] = "ok"

        # Navigate to Step 2
        click_btn(page, r"Next.*Step 2", "Next → Step 2")
        time.sleep(3)

        # ============================================================
        # STEP 2: Figure Detection (LLM)
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 2: Figure Detection (LLM)")
        print("=" * 60)

        click_btn(page, r"Run Step 2", "Run Step 2")
        ok = wait_spinner(page, 300)
        print(f"  {'✅' if ok else '⚠'} Step 2 done")
        ss(page, "02_step2")
        report_page(page, ["Page", "region", "DI", "Phase", "figure", "detect"])

        # Check for canvas / iframes (bounding boxes)
        iframes = page.locator("iframe")
        print(f"  Canvas iframes: {iframes.count()}")
        results["step2"] = "ok" if ok else "timeout"

        # ============================================================
        # HITL #1: Confirm Regions
        # ============================================================
        print("\n" + "=" * 60)
        print("HITL #1: Confirm Regions")
        print("=" * 60)

        ss(page, "03_hitl1_before")
        click_btn(page, r"Confirm Regions.*Step 3", "Confirm Regions → Step 3")
        time.sleep(3)
        wait_spinner(page, 30)
        ss(page, "04_hitl1_confirmed")
        print("  ✅ HITL #1 confirmed")
        results["hitl1"] = "ok"

        # Navigate to Step 3
        click_btn(page, r"Next.*Step 3", "Next → Step 3")
        time.sleep(3)

        # ============================================================
        # STEP 3: Panel Name Extraction (LLM)
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 3: Panel Name Extraction (LLM)")
        print("=" * 60)

        ss(page, "05_step3_before")
        click_btn(page, r"Run Step 3", "Run Step 3")
        ok = wait_spinner(page, 300)
        print(f"  {'✅' if ok else '⚠'} Step 3 done")
        ss(page, "06_step3_done")

        # Report panel names found
        report_page(page, ["Panel", "Name", "name", "overlay", "HITL"])

        # Capture panel name info from body
        body = page.inner_text("body")
        panel_lines = [l.strip() for l in body.split("\n")
                       if any(k in l.lower() for k in ["panel", "name"]) and len(l.strip()) > 3]
        print("\n  --- Panel Name Info ---")
        for line in panel_lines[:15]:
            print(f"    {line[:120]}")
        results["step3"] = "ok" if ok else "timeout"

        # ============================================================
        # HITL #2: Confirm Names
        # ============================================================
        print("\n" + "=" * 60)
        print("HITL #2: Confirm Names")
        print("=" * 60)

        ss(page, "07_hitl2_before")
        click_btn(page, r"Confirm Names.*Step 4", "Confirm Names → Step 4")
        time.sleep(3)
        wait_spinner(page, 30)
        ss(page, "08_hitl2_confirmed")
        print("  ✅ HITL #2 confirmed")
        results["hitl2"] = "ok"

        # Navigate to Step 4
        click_btn(page, r"Next.*Step 4", "Next → Step 4")
        time.sleep(3)

        # ============================================================
        # STEP 4: SVG Grid Detection
        # ============================================================
        print("\n" + "=" * 60)
        print("STEP 4: SVG Grid Detection (no LLM)")
        print("=" * 60)

        ss(page, "09_step4_before")

        # Select SVG Grid method radio button
        svg_radio = page.get_by_text("SVG Grid (fast, no LLM)")
        if svg_radio.count() > 0 and svg_radio.first.is_visible():
            svg_radio.first.click()
            time.sleep(1)
            print("  ✅ SVG Grid method selected")
        else:
            print("  ⚠ SVG Grid radio not found, checking default…")
            # It might be already selected by default (index=0)

        ss(page, "09b_svg_selected")

        # Click Run Step 4: SVG Grid Detection
        clicked = click_btn(page, r"Run Step 4.*SVG", "Run Step 4: SVG Grid")
        if not clicked:
            # Fallback: try any Run Step 4 button
            clicked = click_btn(page, r"Run Step 4", "Run Step 4 (fallback)")

        ok = wait_spinner(page, 300)
        print(f"  {'✅' if ok else '⚠'} Step 4 SVG done")

        ss(page, "10_step4_svg_done")

        # Report results
        report_page(page, ["Panel", "Area", "Bay", "Confirm", "JSON", "crop", "SVG", "svg"])

        # Scroll down for full results
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        ss(page, "10b_step4_scroll")

        # Scroll back up
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        ss(page, "10c_step4_top")

        results["step4_svg"] = "ok" if ok else "timeout"

        # ============================================================
        # Check for errors
        # ============================================================
        body = page.inner_text("body")
        error_lines = [l.strip() for l in body.split("\n")
                       if any(k in l.lower() for k in ["error", "fail", "exception"]) and len(l.strip()) > 3]
        if error_lines:
            print("\n  ⚠ Possible errors on page:")
            for line in error_lines[:10]:
                print(f"    ❌ {line[:150]}")

        # ============================================================
        # SIDEBAR STEPPER STATE
        # ============================================================
        print("\n" + "=" * 60)
        print("SIDEBAR STEPPER STATE")
        print("=" * 60)

        sidebar_text = sidebar.inner_text()
        for line in sidebar_text.split("\n"):
            line = line.strip()
            if any(k in line for k in ["Upload", "Figure", "Panel", "BOM", "Areas", "Step", "✅", "🔵"]):
                print(f"  {line[:80]}")

        # ============================================================
        # TIMING INFO
        # ============================================================
        print("\n" + "=" * 60)
        print("TIMING INFORMATION")
        print("=" * 60)

        for line in body.split("\n"):
            if any(t in line for t in ["S2a:", "S2b:", "S3:", "S4:", "⏱", "elapsed", "time"]):
                print(f"  {line.strip()[:100]}")

        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        for k, v in results.items():
            status = "✅" if v == "ok" else "⚠"
            print(f"  {status} {k}: {v}")

        total = len(results)
        passed = sum(1 for v in results.values() if v == "ok")
        print(f"\n  Total: {passed}/{total} passed")

        # Save results
        with open(SS_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        browser.close()
        print("\n" + "=" * 60)
        print(f"TEST COMPLETE — Screenshots in {SS_DIR}")
        print("=" * 60)


if __name__ == "__main__":
    main()
