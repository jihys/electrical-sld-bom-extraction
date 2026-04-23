"""Visual pipeline test: runs Steps 1-5 in a single browser session, 
takes screenshots at each stage for visual inspection."""
from playwright.sync_api import sync_playwright
import time, re
from pathlib import Path

SS_DIR = Path("outputs/e2e_screenshots")
SS_DIR.mkdir(parents=True, exist_ok=True)

def ss(page, name):
    p = SS_DIR / f"vt_{name}.png"
    page.screenshot(path=str(p), full_page=True)
    print(f"  📸 {p.name}")
    return p

def wait_spinner(page, timeout_s=300):
    """Wait for all spinners/progress bars to disappear."""
    try:
        page.wait_for_selector('[data-testid="stSpinner"]', state='visible', timeout=15000)
    except: pass
    deadline = time.time() + timeout_s
    stable = 0
    while time.time() < deadline:
        sp = page.locator('[data-testid="stSpinner"]').count()
        pb = page.locator('[role="progressbar"]').count()
        if sp == 0 and pb == 0:
            stable += 1
            if stable >= 5: break
        else: stable = 0
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
                 if any(k in l for k in keywords) and len(l.strip()) > 3][:20]
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
        except: pass
    if vis_btns:
        print(f"  Buttons: {vis_btns}")

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.set_default_timeout(60000)
        page.goto("http://localhost:8501", wait_until="networkidle")
        time.sleep(3)
        
        sidebar = page.locator('[data-testid="stSidebar"]')
        
        # ============================================================
        # SETUP: PDF + Visual Prompts
        # ============================================================
        print("\n" + "="*60)
        print("SETUP: Select PDF & Visual Prompts")
        print("="*60)
        
        dropdown = sidebar.locator('div[data-testid="stSelectbox"]').filter(has_text="Test PDF")
        dropdown.locator('[data-baseweb="select"]').click()
        time.sleep(1)
        page.get_by_role("option", name="public_sld_1.pdf", exact=True).click()
        time.sleep(3)
        print("  ✅ PDF: public_sld_1.pdf")
        
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
                except: pass
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
        
        # ============================================================
        # STEP 1: PDF → PNG
        # ============================================================
        print("\n" + "="*60)
        print("STEP 1: PDF → PNG")
        print("="*60)
        
        click_btn(page, r"Run Step 1", "Run Step 1")
        time.sleep(10)
        wait_spinner(page, 60)
        print("  ✅ Step 1 done")
        
        # Select pages 1 & 6 only
        click_btn(page, r"^None$", "None")
        time.sleep(2)
        for pn in [1, 6]:
            cb = page.locator('[data-testid="stCheckbox"]').filter(has_text=re.compile(rf"^Page {pn}$"))
            if cb.count() > 0:
                cb.first.click()
                time.sleep(1)
        print("  ✅ Pages 1, 6 selected")
        ss(page, "01_step1")
        
        # Navigate to Step 2
        click_btn(page, r"Next.*Step 2", "Next → Step 2")
        time.sleep(3)
        
        # ============================================================
        # STEP 2: Figure Detection
        # ============================================================
        print("\n" + "="*60)
        print("STEP 2: Figure Detection (LLM)")
        print("="*60)
        
        click_btn(page, r"Run Step 2", "Run Step 2")
        ok = wait_spinner(page, 300)
        print(f"  {'✅' if ok else '⚠'} Step 2 done")
        ss(page, "02_step2")
        
        # Check results
        report_page(page, ["Page", "region", "DI", "Phase"])
        
        # ============================================================
        # HITL #1: Confirm Regions
        # ============================================================
        print("\n" + "="*60)
        print("HITL #1: Confirm Regions")
        print("="*60)
        
        # Verify HITL UI elements
        iframes = page.locator("iframe")
        print(f"  Canvas iframes: {iframes.count()}")
        add_r = page.get_by_text("Add Region")
        move_r = page.get_by_text("Move / Resize")
        print(f"  Add Region: {add_r.count() > 0}, Move/Resize: {move_r.count() > 0}")
        
        ss(page, "03_hitl1")
        
        click_btn(page, r"Confirm Regions.*Step 3", "Confirm Regions → Step 3")
        time.sleep(3)
        wait_spinner(page, 30)
        
        ss(page, "04_hitl1_confirmed")
        print("  ✅ HITL #1 confirmed")
        
        # ============================================================
        # Navigate to Step 3
        # ============================================================
        click_btn(page, r"Next.*Step 3", "Next → Step 3")
        time.sleep(3)
        
        # ============================================================
        # STEP 3: Panel Names
        # ============================================================
        print("\n" + "="*60)
        print("STEP 3: Panel Name Extraction (LLM)")
        print("="*60)
        
        ss(page, "05_step3_before")
        click_btn(page, r"Run Step 3", "Run Step 3")
        ok = wait_spinner(page, 300)
        print(f"  {'✅' if ok else '⚠'} Step 3 done")
        ss(page, "06_step3_done")
        
        report_page(page, ["Panel", "Name", "name", "overlay", "HITL"])
        
        # Check for panel name list, overlay
        body = page.inner_text("body")
        has_names = "Confirm Names" in body or "Panel" in body
        print(f"  Panel names visible: {has_names}")
        
        # ============================================================
        # HITL #2: Confirm Names
        # ============================================================
        print("\n" + "="*60)
        print("HITL #2: Confirm Names")
        print("="*60)
        
        ss(page, "07_hitl2")
        
        click_btn(page, r"Confirm Names.*Step 4", "Confirm Names → Step 4")
        time.sleep(3)
        wait_spinner(page, 30)
        
        ss(page, "08_hitl2_confirmed")
        print("  ✅ HITL #2 confirmed")
        
        # Navigate to Step 4
        click_btn(page, r"Next.*Step 4", "Next → Step 4")
        time.sleep(3)
        
        # ============================================================
        # STEP 4: Panel Areas + Bay
        # ============================================================
        print("\n" + "="*60)
        print("STEP 4: Panel Areas + Bay (LLM)")
        print("="*60)
        
        ss(page, "09_step4_before")
        click_btn(page, r"Run Step 4", "Run Step 4")
        ok = wait_spinner(page, 300)
        print(f"  {'✅' if ok else '⚠'} Step 4 done")
        ss(page, "10_step4_done")
        
        report_page(page, ["Panel", "Area", "Bay", "Confirm", "JSON", "crop"])
        
        # ============================================================
        # HITL #3: Confirm All
        # ============================================================
        print("\n" + "="*60)
        print("HITL #3: Confirm All")
        print("="*60)
        
        ss(page, "11_hitl3")
        
        # Try multiple patterns for the confirm button
        confirmed = False
        for pat in [r"Confirm.*Step 5", r"Confirm All", r"Confirm.*Done"]:
            if click_btn(page, pat, f"Confirm ({pat})", timeout_s=10):
                confirmed = True
                break
        
        if confirmed:
            time.sleep(3)
            wait_spinner(page, 30)
            ss(page, "12_hitl3_confirmed")
            print("  ✅ HITL #3 confirmed")
        else:
            print("  ❌ Could not find HITL #3 confirm button")
            report_page(page, [])
        
        # Navigate to Step 5
        click_btn(page, r"Next.*Step 5", "Next → Step 5", timeout_s=10)
        time.sleep(3)
        
        # ============================================================
        # STEP 5: BOM Extraction
        # ============================================================
        print("\n" + "="*60)
        print("STEP 5: BOM Extraction (LLM)")
        print("="*60)
        
        ss(page, "13_step5_before")
        click_btn(page, r"Run Step 5", "Run Step 5")
        ok = wait_spinner(page, 300)
        print(f"  {'✅' if ok else '⚠'} Step 5 done")
        ss(page, "14_step5_done")
        
        report_page(page, ["BOM", "Panel", "component", "JSON", "Download", "Final", "Confirm"])
        
        # Scroll down for full results
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        ss(page, "15_step5_scroll")
        
        # Final confirm if needed
        for pat in [r"Confirm BOM", r"Confirm.*Done", r"Download"]:
            btn = page.get_by_role("button", name=re.compile(pat, re.I))
            if btn.count() > 0 and btn.first.is_visible():
                print(f"  Found button: {btn.first.inner_text()}")
        
        # ============================================================
        # SIDEBAR STEPPER STATE
        # ============================================================
        print("\n" + "="*60)
        print("SIDEBAR STEPPER STATE")
        print("="*60)
        
        sidebar_text = sidebar.inner_text()
        for line in sidebar_text.split("\n"):
            line = line.strip()
            if any(k in line for k in ["Upload", "Figure", "Panel", "BOM", "Areas"]):
                print(f"  {line[:80]}")
        
        # ============================================================
        # TIMING INFO
        # ============================================================
        print("\n" + "="*60)
        print("TIMING INFORMATION")
        print("="*60)
        
        body = page.inner_text("body")
        for line in body.split("\n"):
            if any(t in line for t in ["S2a:", "S2b:", "S3:", "S4:", "S5:", "⏱"]):
                print(f"  {line.strip()[:100]}")
        
        # Final full page screenshot
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        ss(page, "16_final_top")
        
        browser.close()
        print("\n" + "="*60)
        print("VISUAL PIPELINE TEST COMPLETE")
        print("="*60)

if __name__ == "__main__":
    main()
