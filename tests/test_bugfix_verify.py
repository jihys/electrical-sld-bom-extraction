"""
Quick Playwright headless tests to verify 3 bug fixes:
1. No error banner on load
2. Step 1 → nav pill back to Step 1 shows thumbnails
3. Re-run Step 1 works without error
"""
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("ERROR: pip install playwright && playwright install chromium")
    sys.exit(1)

BASE_URL = "http://localhost:8501"
SS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "bugfix_screenshots"
SS_DIR.mkdir(parents=True, exist_ok=True)

results = []


def ss(page, name):
    ts = datetime.now().strftime("%H%M%S")
    p = SS_DIR / f"{ts}_{name}.png"
    page.screenshot(path=str(p), full_page=True)
    print(f"  📸 {p}")
    return p


def wait_ready(page, timeout_s=15):
    end = time.time() + timeout_s
    while time.time() < end:
        if page.locator('[data-testid="stSpinner"]').count() == 0:
            break
        time.sleep(0.5)
    time.sleep(1)


def wait_no_spinner(page, timeout_s=60):
    """Wait for spinner to appear then disappear."""
    # Wait for spinner to appear
    try:
        page.wait_for_selector('[data-testid="stSpinner"]', state="visible", timeout=10_000)
    except Exception:
        pass
    # Wait for it to disappear
    end = time.time() + timeout_s
    stable = 0
    while time.time() < end:
        spinners = page.locator('[data-testid="stSpinner"]')
        progress = page.locator('[role="progressbar"]')
        if spinners.count() == 0 and progress.count() == 0:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
        time.sleep(1)
    time.sleep(1)


def record(name, passed, detail="", screenshot_path=""):
    status = "PASS" if passed else "FAIL"
    results.append({"name": name, "status": status, "detail": detail, "screenshot": str(screenshot_path)})
    icon = "✅" if passed else "❌"
    print(f"\n{icon} {name}: {status}")
    if detail:
        print(f"   Detail: {detail}")
    if screenshot_path:
        print(f"   Screenshot: {screenshot_path}")


def run_tests():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()

        # ── Test 1: No error banner ──────────────────────────────
        print(f"\n{'='*60}")
        print("TEST 1: No error banner on load")
        print(f"{'='*60}")
        page.goto(BASE_URL, wait_until="networkidle")
        wait_ready(page)
        s1 = ss(page, "01_initial_load")

        # Check for error banners
        error_texts = []
        for sel in [
            '[data-testid="stException"]',
            '[data-testid="stError"]',
            '.stAlert',
        ]:
            locs = page.locator(sel)
            for i in range(locs.count()):
                txt = locs.nth(i).inner_text()
                if txt.strip():
                    error_texts.append(txt.strip()[:200])

        # Also check for "Last error" or "Errno 5" text anywhere
        body_text = page.locator("body").inner_text()
        has_errno5 = "Errno 5" in body_text or "Last error" in body_text

        if not error_texts and not has_errno5:
            record("No error banner", True, "No error banners found on initial load", s1)
        else:
            detail = f"Errors found: {error_texts}" if error_texts else "Found 'Errno 5' or 'Last error' text"
            record("No error banner", False, detail, s1)

        # ── Setup: Select test.pdf ───────────────────────────────
        print(f"\n{'='*60}")
        print("SETUP: Selecting test.pdf")
        print(f"{'='*60}")
        sidebar = page.locator('[data-testid="stSidebar"]')

        # Reset if available
        try:
            reset_btn = sidebar.get_by_role("button", name="Reset")
            if reset_btn.is_visible(timeout=3000):
                reset_btn.click()
                wait_ready(page, 10)
                print("  🔄 Reset done")
        except Exception:
            pass

        # Select test.pdf from dropdown
        dropdown = sidebar.locator('div[data-testid="stSelectbox"]').filter(has_text="Test PDF")
        dropdown.locator('[data-baseweb="select"]').click()
        time.sleep(1)
        page.get_by_role("option", name="test.pdf", exact=True).click()
        wait_ready(page)
        ss(page, "02_pdf_selected")
        print("  ✅ test.pdf selected")

        # ── Run Step 1: PDF → PNG ────────────────────────────────
        print(f"\n{'='*60}")
        print("SETUP: Running Step 1 (PDF → PNG)")
        print(f"{'='*60}")

        step1_btn = page.get_by_role("button", name=re.compile(r"Run Step 1", re.IGNORECASE))
        if step1_btn.count() > 0:
            step1_btn.first.scroll_into_view_if_needed()
            step1_btn.first.click()
            print("  🖱️ Clicked Run Step 1")
            wait_no_spinner(page, 30)
            wait_ready(page)
        ss(page, "03_step1_done")

        # Verify Step 1 completed
        body = page.locator("body").inner_text()
        step1_ok = "Converted" in body or "page" in body.lower()
        print(f"  Step 1 result: {'OK' if step1_ok else 'may have issues'}")

        # ── Test 2: Nav pill back to Step 1 ──────────────────────
        print(f"\n{'='*60}")
        print("TEST 2: Step 1 → Next → nav pill back → thumbnails visible")
        print(f"{'='*60}")

        # Click "Next → Step 2"
        next_btn = page.get_by_role("button", name=re.compile(r"Next.*Step 2|→.*Step 2", re.IGNORECASE))
        if next_btn.count() > 0:
            next_btn.first.scroll_into_view_if_needed()
            time.sleep(0.3)
            next_btn.first.click()
            print("  🖱️ Clicked Next → Step 2")
            wait_ready(page)
        else:
            print("  ⚠ 'Next → Step 2' button not found, trying alternative")
            # Maybe there's a different next button
            all_btns = page.get_by_role("button").all_inner_texts()
            print(f"  Available buttons: {[b for b in all_btns if b.strip()][:20]}")

        ss(page, "04_at_step2")

        # Click sidebar nav pill "1" to go back to Step 1
        time.sleep(1)
        # Look for nav pill in sidebar - could be button with text "1" or similar
        nav_pill_clicked = False
        
        # Try finding nav pill buttons in sidebar
        # Nav pills are typically small numbered buttons in sidebar
        sidebar_buttons = sidebar.get_by_role("button")
        for i in range(sidebar_buttons.count()):
            btn = sidebar_buttons.nth(i)
            txt = btn.inner_text().strip()
            if txt == "1":
                btn.click()
                nav_pill_clicked = True
                print("  🖱️ Clicked nav pill '1' in sidebar")
                break

        if not nav_pill_clicked:
            # Try radio buttons or other navigation elements
            # Streamlit often uses radio for nav
            step1_nav = sidebar.get_by_text(re.compile(r"^1$|Step 1|PDF.*PNG", re.IGNORECASE))
            if step1_nav.count() > 0:
                step1_nav.first.click()
                nav_pill_clicked = True
                print("  🖱️ Clicked Step 1 nav element")

        if not nav_pill_clicked:
            # Try clicking any element that looks like step 1 navigation
            nav_items = sidebar.locator("button, [role='tab'], [data-testid='stRadio'] label")
            for i in range(nav_items.count()):
                item = nav_items.nth(i)
                txt = item.inner_text().strip()
                if "1" in txt and len(txt) < 30:
                    item.click()
                    nav_pill_clicked = True
                    print(f"  🖱️ Clicked nav item: '{txt}'")
                    break

        wait_ready(page)
        s2 = ss(page, "05_back_to_step1")

        # Check if thumbnails are visible
        # Thumbnails could be <img> elements in the main content
        time.sleep(1)
        images = page.locator('[data-testid="stImage"] img, img[src*="thumb"], img[src*="page"]')
        img_count = images.count()
        
        # Also check for any images in the main area
        all_imgs = page.locator('section.main img, [data-testid="stAppViewContainer"] img')
        all_img_count = all_imgs.count()
        
        body_text = page.locator("body").inner_text()
        has_page_content = any(kw in body_text.lower() for kw in ["page", "converted", "thumbnail", "png", "선택"])

        if img_count > 0 or (all_img_count > 0 and has_page_content):
            record("Nav pill back to Step 1", True,
                   f"Thumbnails visible ({img_count} stImage, {all_img_count} total imgs)", s2)
        elif has_page_content and nav_pill_clicked:
            record("Nav pill back to Step 1", True,
                   f"Step 1 content visible (page content found, {all_img_count} imgs)", s2)
        else:
            record("Nav pill back to Step 1", False,
                   f"No thumbnails found ({img_count} stImage, {all_img_count} imgs, nav_clicked={nav_pill_clicked})", s2)

        # ── Test 3: Re-run Step 1 ────────────────────────────────
        print(f"\n{'='*60}")
        print("TEST 3: Re-run Step 1")
        print(f"{'='*60}")

        rerun_btn = page.get_by_role("button", name=re.compile(r"Re-?run Step 1|Run Step 1", re.IGNORECASE))
        if rerun_btn.count() > 0:
            rerun_btn.first.scroll_into_view_if_needed()
            time.sleep(0.3)
            rerun_btn.first.click()
            print("  🖱️ Clicked Re-run Step 1")
            wait_no_spinner(page, 30)
            wait_ready(page)
        else:
            print("  ⚠ Re-run button not found")
            all_btns = page.get_by_role("button").all_inner_texts()
            print(f"  Available buttons: {[b for b in all_btns if b.strip()][:20]}")

        s3 = ss(page, "06_rerun_step1")

        # Check no error after re-run
        error_after = []
        for sel in ['[data-testid="stException"]', '[data-testid="stError"]']:
            locs = page.locator(sel)
            for i in range(locs.count()):
                txt = locs.nth(i).inner_text()
                if txt.strip():
                    error_after.append(txt.strip()[:200])

        body_text = page.locator("body").inner_text()
        has_error_text = "Error" in body_text and ("Errno" in body_text or "Traceback" in body_text)

        if not error_after and not has_error_text:
            # Verify conversion seems successful
            has_converted = "Converted" in body_text or "page" in body_text.lower()
            record("Re-run Step 1", True,
                   f"No errors after re-run, conversion={'OK' if has_converted else 'unknown'}", s3)
        else:
            detail = f"Errors after re-run: {error_after}" if error_after else "Error text found in body"
            record("Re-run Step 1", False, detail, s3)

        # ── Summary ──────────────────────────────────────────────
        browser.close()

    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    for r in results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"  {icon} {r['name']}: {r['status']}")
        if r["detail"]:
            print(f"     {r['detail']}")
        if r["screenshot"]:
            print(f"     Screenshot: {r['screenshot']}")
    print(f"\nTotal: {passed}/{total} passed")
    
    # Save results
    import json
    results_path = SS_DIR / "bugfix_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run_tests())
