"""Quick test: improved panel name extraction on page 5."""
import base64
import time
from pathlib import Path

import cv2

from src.config import Settings
from src.agents.llm_caller import create_llm_client
from src.cad.panel_name_extractor import (
    extract_panel_names_from_tile,
    dedup_panel_names_for_page,
    verify_panel_names_with_full_page,
)

GT = [
    "NGR-E-TR-01B", "E-TR-01B",
    "E-H-02B (UPPER)", "E-H-03B (LOWER)",
    "E-H-04B (UPPER)", "E-H-05B (LOWER)",
    "E-H-06B (UPPER)", "E-H-07B (LOWER)",
    "E-H-08B", "E-H-09B", "E-H-10B", "E-H-01B",
]

def main():
    settings = Settings()
    client = create_llm_client(
        endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )
    deploy = settings.azure_openai_deployment

    # Load page 5 image
    img_path = Path("outputs/pages/page5.png")
    full_img = cv2.imread(str(img_path))
    h, w = full_img.shape[:2]
    print(f"Page 5: {w}x{h}")

    # Load visual reference
    ref_path = Path("data/panel_name_box_example2.png")
    example_img_b64 = base64.b64encode(ref_path.read_bytes()).decode() if ref_path.exists() else None
    print(f"Visual ref: {'loaded' if example_img_b64 else 'NONE'}")

    # ── Tile (1200px width, 400 overlap) ──────────────────────────────────
    tile_width, overlap = 1200, 400
    tile_bboxes = []
    x = 0
    while x < w:
        x2 = min(x + tile_width, w)
        tile_bboxes.append((x, 0, x2, h))
        if x2 >= w:
            break
        x += tile_width - overlap
    print(f"Tiles: {len(tile_bboxes)}  ({tile_width}px width, {overlap}px overlap)")
    for i, (x1, y1, x2, y2) in enumerate(tile_bboxes):
        print(f"  tile{i}: x=[{x1},{x2}] ({x2-x1}px)")

    # ── Phase 1: Per-tile extraction ──────────────────────────────────────
    t0 = time.time()
    tile_cands = []
    for ti, (tx1, ty1, tx2, ty2) in enumerate(tile_bboxes):
        tile_img = full_img[ty1:ty2, tx1:tx2]
        names, _el = extract_panel_names_from_tile(
            tile_img, client, deploy,
            example_img_b64=example_img_b64,
        )
        print(f"  tile{ti}: {names}")
        if names:
            tile_cands.extend(names)
    t_extract = time.time() - t0

    print(f"\n--- Tile candidates ({len(tile_cands)}): {tile_cands}")

    # ── Phase 2: Dedup ────────────────────────────────────────────────────
    deduped, t_dedup = dedup_panel_names_for_page(
        tile_cands, full_img, client, deploy,
    )
    print(f"--- Deduped ({len(deduped)}): {deduped}")

    # ── Phase 3: Full-page verification ───────────────────────────────────
    verified, t_verify = verify_panel_names_with_full_page(
        deduped, full_img, client, deploy,
        example_img_b64=example_img_b64,
    )
    print(f"--- Verified ({len(verified)}): {verified}")

    total_time = time.time() - t0
    print(f"\nTime: extract={t_extract:.1f}s  dedup={t_dedup:.1f}s  verify={t_verify:.1f}s  total={total_time:.1f}s")

    # ── Evaluate vs GT ────────────────────────────────────────────────────
    gt_set = {n.upper() for n in GT}
    verified_set = {n.upper() for n in verified}

    hits = gt_set & verified_set
    missed = gt_set - verified_set
    extra = verified_set - gt_set

    print(f"\n{'='*60}")
    print(f"GT: {len(GT)}  |  Extracted: {len(verified)}  |  Recall: {len(hits)}/{len(GT)} ({100*len(hits)/len(GT):.0f}%)")
    print(f"Hits:   {sorted(hits)}")
    if missed:
        print(f"Missed: {sorted(missed)}")
    if extra:
        print(f"Extra:  {sorted(extra)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
