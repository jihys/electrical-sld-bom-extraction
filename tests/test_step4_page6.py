"""Step 4 locate/verify A/B test on page 6.

Compares crop quality with/without guide images.
Uses GT from panel_names_test_image_example_GT.json.
"""
import base64, json, time, shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.config import Settings
from src.agents.llm_caller import create_llm_client
from src.cad.panel_pipeline import process_all_panels_batch
from src.cad.panel_utils import safe_name

GT_PATH = Path("/home/azureuser/localfiles/cad-image-understanding/data/panel_names_test_image_example_GT.json")
PAGE = 6
PAGE_IMG = Path("outputs/pages/page6.png")

# Name bboxes from latest checkpoint
NAME_BBOXES_P6 = {
    "LV-DTB":         [1961, 1122, 2017, 1136],
    "E-TR-02B":       [173, 427, 272, 470],
    "LV-01":          [168, 722, 215, 736],
    "LV-06":          [1607, 1121, 1652, 1136],
    "LV-02 (UPPER)":  [164, 1120, 252, 1137],
    "LV-04 (UPPER)":  [882, 1121, 970, 1137],
    "LV-03 (LOWER)":  [522, 1121, 612, 1137],
    "LV-05 (LOWER)":  [1239, 1121, 1330, 1136],
}
NAMES_P6 = list(NAME_BBOXES_P6.keys())


def evaluate_crops(results: List[dict], full_img: np.ndarray, label: str):
    """Evaluate crop quality by checking sizes and overlaps."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Located: {len(results)}/{len(NAMES_P6)} panels")

    h_full, w_full = full_img.shape[:2]
    bboxes = {}
    for r in results:
        name = r["panel_name"]
        bbox = r["bbox"]
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        area_pct = (bw * bh) / (w_full * h_full) * 100
        bboxes[name] = bbox
        excl = r.get("exclude_regions", [])
        excl_str = f"  excl={len(excl)}" if excl else ""
        print(f"  {name:20s}: [{bbox[0]:4d},{bbox[1]:4d},{bbox[2]:4d},{bbox[3]:4d}]  {bw}x{bh}  {area_pct:.1f}%{excl_str}")

    # Check overlaps between panels
    names = list(bboxes.keys())
    overlaps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = bboxes[names[i]]
            b = bboxes[names[j]]
            ox1 = max(a[0], b[0])
            oy1 = max(a[1], b[1])
            ox2 = min(a[2], b[2])
            oy2 = min(a[3], b[3])
            if ox1 < ox2 and oy1 < oy2:
                ow = ox2 - ox1
                oh = oy2 - oy1
                overlap_area = ow * oh
                a_area = (a[2]-a[0]) * (a[3]-a[1])
                b_area = (b[2]-b[0]) * (b[3]-b[1])
                min_area = min(a_area, b_area)
                overlap_pct = overlap_area / min_area * 100
                # Only report significant overlaps (>5% of smaller panel)
                if overlap_pct > 5:
                    overlaps.append((names[i], names[j], overlap_pct))
    if overlaps:
        print(f"\n  ⚠ Overlaps:")
        for n1, n2, pct in overlaps:
            print(f"    {n1} ↔ {n2}: {pct:.0f}%")
    else:
        print(f"\n  ✓ No significant overlaps")

    # Missing panels
    found = {r["panel_name"] for r in results}
    missing = [n for n in NAMES_P6 if n not in found]
    if missing:
        print(f"  ⚠ Missing: {missing}")

    return bboxes


def run_test(guide_images: List[np.ndarray], label: str, out_dir: Path):
    """Run locate+verify on page 6 with given guide images."""
    settings = Settings()
    client = create_llm_client(
        endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
    deploy = settings.azure_openai_deployment

    full_img = cv2.imread(str(PAGE_IMG))
    if full_img is None:
        print(f"ERROR: Cannot load {PAGE_IMG}")
        return []

    page_out = out_dir / "panels"
    page_debug = out_dir / "debug"
    page_out.mkdir(parents=True, exist_ok=True)
    page_debug.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = process_all_panels_batch(
        client, deploy, guide_images,
        page_num=PAGE, crop_idx=1,
        names=NAMES_P6, img=full_img,
        all_name_bboxes=NAME_BBOXES_P6,
        page_out_dir=page_out,
        page_debug_dir=page_debug,
        grid_size=120,
        verify_max_tries=3,
        verify_reasoning_effort="none",
    )
    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")

    bboxes = evaluate_crops(results, full_img, label)
    return results


def main():
    guide_ref_path = Path("data/panel_box_explanation.png")
    guide_ref1_path = Path("data/panel_box_explanation1.png")

    # ── Test A: No guide images ──────────────────────────────────
    print("\n" + "▓" * 60)
    print("  TEST A: No guide images")
    print("▓" * 60)
    out_a = Path("outputs/_step4_test/test_A_no_guide")
    if out_a.exists():
        shutil.rmtree(out_a)
    results_a = run_test([], "A: No guide", out_a)

    # ── Test B: panel_box_explanation.png (3 examples) ───────────
    print("\n" + "▓" * 60)
    print("  TEST B: panel_box_explanation.png (3 examples)")
    print("▓" * 60)
    guide_b = cv2.imread(str(guide_ref_path))
    out_b = Path("outputs/_step4_test/test_B_explanation")
    if out_b.exists():
        shutil.rmtree(out_b)
    results_b = run_test([guide_b] if guide_b is not None else [], "B: explanation.png", out_b)

    # ── Test C: panel_box_explanation1.png (2 examples, larger) ──
    print("\n" + "▓" * 60)
    print("  TEST C: panel_box_explanation1.png (2 examples)")
    print("▓" * 60)
    guide_c = cv2.imread(str(guide_ref1_path))
    out_c = Path("outputs/_step4_test/test_C_explanation1")
    if out_c.exists():
        shutil.rmtree(out_c)
    results_c = run_test([guide_c] if guide_c is not None else [], "C: explanation1.png", out_c)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  COMPARISON SUMMARY")
    print("═" * 60)
    for label, results in [("A: No guide", results_a), ("B: explanation.png", results_b), ("C: explanation1.png", results_c)]:
        found = len(results)
        overlaps = 0
        bboxes = {r["panel_name"]: r["bbox"] for r in results}
        names = list(bboxes.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = bboxes[names[i]]
                b = bboxes[names[j]]
                ox1 = max(a[0], b[0])
                oy1 = max(a[1], b[1])
                ox2 = min(a[2], b[2])
                oy2 = min(a[3], b[3])
                if ox1 < ox2 and oy1 < oy2:
                    ow = ox2 - ox1
                    oh = oy2 - oy1
                    a_area = (a[2]-a[0]) * (a[3]-a[1])
                    b_area = (b[2]-b[0]) * (b[3]-b[1])
                    if (ow * oh) / min(a_area, b_area) > 0.05:
                        overlaps += 1
        print(f"  {label:30s}: {found}/{len(NAMES_P6)} panels  {overlaps} overlaps")


if __name__ == "__main__":
    main()
