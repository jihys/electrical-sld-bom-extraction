"""
panel_viz.py
------------
Visualisation helpers for panel-name extraction notebook.

Public API
----------
visualize_tiles(tiles_by_page, page_images, output_root, notebook_dir,
                max_pages=3)

visualize_panel_bboxes(panel_bbox_matches, page_images, output_root,
                       notebook_dir)

visualize_panel_areas(summary, notebook_dir, n_cols=5)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt


_TILE_COLORS = [
    (255,  60,  60), ( 60, 100, 255), ( 40, 200,  40),
    (255, 160,   0), (180,  60, 220), (  0, 200, 200),
]

_METHOD_COLORS = {
    "rule-single":   (  0, 200,   0),
    "rule-alphanum": (  0, 220,  80),
    "rule-token":    (  0, 180,  60),
    "rule-merge2":   (  0, 120, 255),
    "rule-merge3":   (  0,  80, 200),
    "llm":           (200,  80,   0),
    "llm-validate":  (220, 100,   0),
    "val-positional": (  0, 180, 180),   # cyan: position-based gap fill (tile validation Step 2)
}
_DEFAULT_COLOR = (150, 150, 150)


def visualize_tiles(
    tiles_by_page: Dict[int, list],
    page_images: dict,
    output_root: Path,
    notebook_dir: Path,
    max_pages: int = 3,
    display_width: int = 1400,
):
    """Draw tile bounding boxes on full-page images and show with matplotlib."""
    viz_pages = sorted(tiles_by_page.keys())[:max_pages]
    for viz_page in viz_pages:
        full_img = page_images.get(viz_page)
        tiles = tiles_by_page[viz_page]
        if full_img is None:
            print(f"page {viz_page}: no full-page image — skip visualisation")
            continue

        vis = full_img.copy()
        for i, (x1, y1, x2, y2) in enumerate(tiles):
            c = _TILE_COLORS[i % len(_TILE_COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 6)
            label = f"T{i+1}  {x2-x1}×{y2-y1}"
            cv2.putText(vis, label, (x1 + 10, y1 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3)

        scale = min(1.0, display_width / max(vis.shape[:2]))
        vis_small = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))

        plt.figure(figsize=(18, 10))
        plt.imshow(cv2.cvtColor(vis_small, cv2.COLOR_BGR2RGB))
        plt.title(f"Page {viz_page} — {len(tiles)} tiles total (scale {scale:.2f}×)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        vis_path = output_root / f"page{viz_page}_tiles.png"
        cv2.imwrite(str(vis_path), vis)
        print(f"Saved overlay: {vis_path.resolve().relative_to(notebook_dir)}")

        # Save individual tile crops
        tiles_dir = output_root / f"page{viz_page}_tile_crops"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for i, (x1, y1, x2, y2) in enumerate(tiles):
            crop = full_img[y1:y2, x1:x2]
            tile_path = tiles_dir / f"tile_{i+1:02d}_{x1}_{y1}_{x2}_{y2}.png"
            cv2.imwrite(str(tile_path), crop)
        print(f"Saved {len(tiles)} tile crops → {tiles_dir.resolve().relative_to(notebook_dir)}")


def visualize_panel_bboxes(
    panel_bbox_matches: Dict[int, Dict[str, list]],
    page_images: dict,
    output_root: Path,
    notebook_dir: Path,
    crops: Optional[list] = None,
    display_width: int = 1400,
):
    """Draw matched panel bboxes and show in notebook (method colors) + save per crop (blue).

    Notebook display : method-colored boxes on the full-page image (scaled).
    File save        : blue boxes + blue text on each image_N_M.png crop at full resolution.
                       Saved to  output_root/viz_panel_bbox/image_N_M_bbox.png

    Parameters
    ----------
    crops : list of crop dicts (from discover_crops).  If provided, per-crop images
            are saved.  Each crop dict must have keys:
              page_num, crop_idx, bbox (x0,y0,x1,y1), img_path (Path).
    """
    import numpy as np

    BLUE = (255, 0, 0)   # BGR
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    viz_dir = output_root / "viz_panel_bbox"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── Build per-page drawn list ─────────────────────────────────────────────
    for pn in sorted(panel_bbox_matches.keys()):
        full_img = page_images.get(pn)
        if full_img is None:
            print(f"page {pn}: no full-page image -- skip")
            continue

        # Collect all (x1,y1,x2,y2, name, method_color) for this page
        drawn: List[tuple] = []
        for name, ms in panel_bbox_matches[pn].items():
            hits = ms if isinstance(ms, list) else ([ms] if ms else [])
            for m in hits:
                x1, y1, x2, y2 = [int(v) for v in m["bbox"]]
                c = _METHOD_COLORS.get(m.get("method", ""), _DEFAULT_COLOR)
                drawn.append((x1, y1, x2, y2, name, c))

        # ── Notebook display: method colors, ALL names ────────────────────────
        vis = full_img.copy()
        for x1, y1, x2, y2, name, c in drawn:
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 4)
            cv2.putText(vis, name, (x1, max(y1 - 8, 20)), FONT, 0.8, c, 2)

        scale = min(1.0, display_width / max(vis.shape[:2]))
        small = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
        plt.figure(figsize=(18, 10))
        plt.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        plt.title(
            f"Page {pn} — Panel Bbox Matches  "
            f"(green=rule, blue=merge, orange=llm, cyan=val-positional)"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # ── Per-crop save: blue, full resolution ──────────────────────────────
        if crops is None:
            # Fallback: save full-page viz with blue boxes
            save_vis = full_img.copy()
            for x1, y1, x2, y2, name, _ in drawn:
                cv2.rectangle(save_vis, (x1, y1), (x2, y2), BLUE, 4)
                cv2.putText(save_vis, name, (x1, max(y1 - 8, 20)), FONT, 0.8, BLUE, 2)
            p = viz_dir / f"page{pn}_panel_bbox.png"
            cv2.imwrite(str(p), save_vis)
            print(f"Saved: {p.resolve().relative_to(notebook_dir)}")
            continue

        # crops is List[(page_num, crop_idx, png_path, bbox)]
        page_crops = [c for c in crops if c[0] == pn]
        if not page_crops:
            continue

        for crop in page_crops:
            _pn, _cidx, img_path, crop_bbox = crop
            cx0, cy0, cx1_c, cy1_c = [int(v) for v in crop_bbox]

            # Load the original crop image at full resolution
            crop_bgr = cv2.imread(str(img_path))
            if crop_bgr is None:
                print(f"  [WARN] cannot read {img_path}")
                continue

            # Draw blue boxes (convert page coords → crop-local coords)
            for px1, py1, px2, py2, name, _ in drawn:
                # Skip boxes entirely outside this crop
                if px2 < cx0 or px1 > cx1_c or py2 < cy0 or py1 > cy1_c:
                    continue
                lx1 = max(px1 - cx0, 0)
                ly1 = max(py1 - cy0, 0)
                lx2 = min(px2 - cx0, crop_bgr.shape[1])
                ly2 = min(py2 - cy0, crop_bgr.shape[0])
                cv2.rectangle(crop_bgr, (lx1, ly1), (lx2, ly2), BLUE, 3)
                label_y = max(ly1 - 8, 20)
                cv2.putText(crop_bgr, name, (lx1, label_y), FONT, 0.7, BLUE, 2)

            out_name = img_path.stem + "_bbox.png"
            out_path = viz_dir / out_name
            cv2.imwrite(str(out_path), crop_bgr)
            print(f"Saved crop viz: {out_path.resolve().relative_to(notebook_dir)}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage-by-stage pipeline comparison
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_LABEL_COLORS = {
    "1_rule_only":    (  0, 200,   0),   # green
    "2_llm_fallback": (200, 100,   0),   # orange
    "3_validation":   (  0, 140, 220),   # blue
}

_ADDED_COLOR   = (255,  60, 220)   # magenta  — newly appeared vs prev stage
_REMOVED_COLOR = (  0,  40, 200)   # deep red — gone vs prev stage (shown on prev)


def _draw_dashed_rect(img, x1, y1, x2, y2, color, dash=16, thickness=3):
    """Draw an approximated dashed rectangle (used to mark removed bboxes)."""
    pts = [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]
    for sx, sy, ex, ey in pts:
        length = max(abs(ex - sx), abs(ey - sy))
        steps = max(1, length // dash)
        for i in range(steps):
            if i % 2 == 1:
                continue
            t0, t1 = i / steps, min((i + 1) / steps, 1.0)
            p0 = (int(sx + (ex - sx) * t0), int(sy + (ey - sy) * t0))
            p1 = (int(sx + (ex - sx) * t1), int(sy + (ey - sy) * t1))
            cv2.line(img, p0, p1, color, thickness)


def _bbox_key(name: str, hit: dict) -> tuple:
    x1, y1, x2, y2 = [round(v) for v in hit["bbox"]]
    return (name, x1, y1, x2, y2)


def visualize_pipeline_stages(
    stages: dict,
    page_images: dict,
    target_pages: Optional[List[int]] = None,
    output_root: Optional[Path] = None,
    notebook_dir: Optional[Path] = None,
    display_width: int = 1600,
):
    """Compare panel bbox state across pipeline stages for selected pages.

    Parameters
    ----------
    stages       : OrderedDict {stage_label: panel_bbox_matches snapshot}
                   Keys should be like '1_rule_only', '2_llm_fallback', '3_validation'.
    page_images  : {page_num: BGR ndarray}
    target_pages : pages to visualize; None = all pages present in stages
    output_root  : optional dir to save PNG files
    notebook_dir : base for relative path display
    display_width: total figure width hint in pixels (per stage column)

    Stage diff legend
    -----------------
    - Unchanged bboxes from prev stage  → method color (same as single-view)
    - Added bboxes (new in this stage)  → magenta thick border
    - Removed bboxes (gone vs prev)     → dashed dark-red on the PREVIOUS stage column

    Each subplot shows a summary:
      matched/total | +N added / -N removed (vs prev)
    """
    import numpy as np

    stage_labels = list(stages.keys())
    n_stages = len(stage_labels)
    if n_stages == 0:
        print("No stages to visualize.")
        return

    all_pages: set = set()
    for s in stages.values():
        all_pages.update(s.keys())
    pages = sorted(target_pages or all_pages)

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for pn in pages:
        full_img = page_images.get(pn)
        if full_img is None:
            print(f"page {pn}: no image — skip")
            continue

        h, w = full_img.shape[:2]

        # Pre-compute per-stage bbox key sets for diff
        stage_keys: List[set] = []
        for lbl in stage_labels:
            sm = stages[lbl].get(pn, {})
            keys = {_bbox_key(n, h) for n, hits in sm.items() for h in hits}
            stage_keys.append(keys)

        # Build rendered images (one per stage, with diff overlay)
        rendered: List[np.ndarray] = []
        summaries: List[str] = []

        for idx, lbl in enumerate(stage_labels):
            vis = full_img.copy()
            sm = stages[lbl].get(pn, {})
            cur_keys = stage_keys[idx]
            prev_keys = stage_keys[idx - 1] if idx > 0 else set()

            added   = cur_keys - prev_keys   if idx > 0 else set()
            removed = prev_keys - cur_keys   if idx > 0 else set()

            # Draw removed bboxes as dashed dark-red (still on this canvas for context)
            for key in removed:
                _, rx1, ry1, rx2, ry2 = key
                _draw_dashed_rect(vis, rx1, ry1, rx2, ry2, _REMOVED_COLOR, dash=18, thickness=3)
                cv2.putText(vis, "DEL", (rx1, max(ry1 - 6, 20)), FONT, 0.65, _REMOVED_COLOR, 2)

            # Draw current bboxes
            for name, hits in sm.items():
                for hit in hits:
                    x1, y1, x2, y2 = [int(v) for v in hit["bbox"]]
                    key = _bbox_key(name, hit)
                    method_c = _METHOD_COLORS.get(hit.get("method", ""), _DEFAULT_COLOR)

                    if key in added:
                        # Extra thick magenta outer ring → then method color inner
                        cv2.rectangle(vis, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), _ADDED_COLOR, 6)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), method_c, 3)
                        cv2.putText(vis, f"+{name}", (x1, max(y1 - 8, 20)), FONT, 0.75, _ADDED_COLOR, 2)
                    else:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), method_c, 4)
                        cv2.putText(vis, name, (x1, max(y1 - 8, 20)), FONT, 0.75, method_c, 2)

            rendered.append(vis)

            matched = sum(1 for hits in sm.values() if hits)
            total   = len(sm)
            diff_str = f"  |  +{len(added)} / -{len(removed)}" if idx > 0 else ""
            summaries.append(f"{lbl}\npage {pn}: {matched}/{total} matched{diff_str}")

        # Plot — keep full-res images; use display_width hint for figsize
        col_inch = max(10, display_width / n_stages / 100)
        fig_h_inch = max(col_inch * h / w, 8)
        fig, axes = plt.subplots(1, n_stages, figsize=(col_inch * n_stages, fig_h_inch), dpi=150)
        if n_stages == 1:
            axes = [axes]

        for ax, img_rgb, summary in zip(axes, rendered, summaries):
            ax.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB), interpolation="lanczos")
            ax.set_title(summary, fontsize=9, loc="left")
            ax.axis("off")

        # Legend
        legend_text = (
            "Legend:  green=rule  orange=llm-fallback  blue=val-positional\n"
            "magenta border=ADDED vs prev   dashed red=REMOVED vs prev"
        )
        fig.text(0.5, 0.01, legend_text, ha="center", fontsize=9, color="#444")

        plt.suptitle(f"Page {pn} — Pipeline Stage Comparison", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])
        plt.show()

        if output_root:
            out_path = output_root / f"page{pn}_pipeline_stages.png"
            fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
            rel = out_path.resolve().relative_to(notebook_dir) if notebook_dir else out_path
            print(f"Saved: {rel}")
        plt.close(fig)


def visualize_panel_areas(summary: dict, notebook_dir: Path, n_cols: int = 5) -> None:
    """Display extracted panel crop images as a grid based on panel_areas_summary.json."""
    for page_key, items in sorted(summary.get('by_page', {}).items()):
        if not items:
            continue

        entries = []
        for it in items:
            f = it.get('file')
            if f:
                entries.append((f"{it['panel_name']}\n(crop{it['crop_idx']})", notebook_dir / f))

        print(f'\nPage {page_key} — {len(items)} panels, {len(entries)} crop(s)')
        sample = entries[:25]
        n_rows = (len(sample) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.8), squeeze=False)
        axes_flat = axes.flatten()

        for i, (title, img_path) in enumerate(sample):
            img = cv2.imread(str(img_path))
            if img is not None:
                axes_flat[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes_flat[i].set_title(title, fontsize=7)
            axes_flat[i].axis('off')

        for j in range(len(sample), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.suptitle(f'Page {page_key} — Panel Areas', fontsize=12, y=0.995)
        plt.tight_layout()
        plt.show()


def visualize_panel_areas_on_crop(
    summary: dict,
    crop_images: dict,
    display_width: int = 1400,
) -> None:
    """Draw extracted panel bboxes on the original crop image.

    Parameters
    ----------
    summary      : panel_areas_summary dict with 'by_page' → [{panel_name, bbox, crop_idx, ...}]
    crop_images  : Dict[(page_num, crop_idx)] → np.ndarray (BGR) of the original crop image.
                   Alternatively Dict[int, np.ndarray] keyed by page_num (assumes crop_idx=1).
    display_width: max display width in pixels.
    """
    import numpy as np

    # Normalize crop_images keys: accept (page, idx) or just page
    _images: Dict = {}
    for k, v in crop_images.items():
        if isinstance(k, tuple):
            _images[k] = v
        else:
            _images[(int(k), 1)] = v

    COLORS = [
        (255,  60,  60), ( 60, 100, 255), ( 40, 200,  40),
        (255, 160,   0), (180,  60, 220), (  0, 200, 200),
        (255, 255,  60), (120,  60, 180),
    ]
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for page_key, items in sorted(summary.get('by_page', {}).items()):
        if not items:
            continue

        # Group by crop_idx
        by_crop: Dict[int, list] = {}
        for it in items:
            by_crop.setdefault(it['crop_idx'], []).append(it)

        for crop_idx, panels in sorted(by_crop.items()):
            key = (int(page_key), crop_idx)
            base = _images.get(key)
            if base is None:
                print(f'Page {page_key} crop {crop_idx}: no source image — skip overlay')
                continue

            vis = base.copy()
            for i, panel in enumerate(panels):
                color = COLORS[i % len(COLORS)]
                x1, y1, x2, y2 = panel['bbox']
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                label = panel['panel_name']
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.9, 2)
                cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(vis, label, (x1 + 3, y1 - 5), FONT, 0.9, (255, 255, 255), 2)

                # Draw exclude_regions with semi-transparent fill + dashed border
                for ex in panel.get('exclude_regions', []):
                    ex1, ey1, ex2, ey2 = ex
                    # Semi-transparent white overlay on excluded area
                    overlay = vis[ey1:ey2, ex1:ex2].copy()
                    white = np.full_like(overlay, 255)
                    cv2.addWeighted(white, 0.5, overlay, 0.5, 0, overlay)
                    vis[ey1:ey2, ex1:ex2] = overlay
                    # Dashed border for excluded area
                    cv2.rectangle(vis, (ex1, ey1), (ex2, ey2), (0, 0, 200), 2, cv2.LINE_AA)
                    # "X" label
                    cv2.putText(vis, 'EXCL', (ex1 + 5, ey1 + 20), FONT, 0.6, (0, 0, 200), 2)

            scale = min(1.0, display_width / max(vis.shape[:2]))
            small = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
            plt.figure(figsize=(18, 10))
            plt.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            plt.title(
                f'Page {page_key} crop {crop_idx} — '
                f'{len(panels)} panel(s): {", ".join(p["panel_name"] for p in panels)}',
                fontsize=11,
            )
            plt.axis('off')
            plt.tight_layout()
            plt.show()


def visualize_bay_areas(bay_summary: dict, notebook_dir: Path, n_cols: int = 5) -> None:
    """Display bay-split images as a grid based on bay_areas_summary.json."""
    for page_key, items in sorted(bay_summary.get('by_page', {}).items()):
        if not items:
            continue

        entries = []
        for it in items:
            for i, bay in enumerate(it.get('bays', [])):
                f = bay.get('file')
                if f:
                    entries.append((f"{it['panel_name']}\nbay{i + 1:02d}", notebook_dir / f))

        if not entries:
            continue

        print(f'\nPage {page_key} — {len(items)} panels split, {len(entries)} bay(s)')
        sample = entries[:25]
        n_rows = (len(sample) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 2.8), squeeze=False)
        axes_flat = axes.flatten()

        for i, (title, img_path) in enumerate(sample):
            img = cv2.imread(str(img_path))
            if img is not None:
                axes_flat[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes_flat[i].set_title(title, fontsize=7)
            axes_flat[i].axis('off')

        for j in range(len(sample), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.suptitle(f'Page {page_key} — Bay Areas', fontsize=12, y=0.995)
        plt.tight_layout()
        plt.show()


def compare_panel_results(
    notebook_dir: Path,
    summary_a: dict,
    summary_b: dict,
    label_a: str = 'with_guide',
    label_b: str = 'no_guide',
    compare_pages: Optional[List[int]] = None,
) -> None:
    """Compare two run results (with guide vs without guide) side by side per panel.

    For each panel, the left column shows result A and the right column shows result B in the same row.
    If only one side has a result, the other is shown as a blank.
    """
    all_pages = set(summary_a.get('by_page', {}).keys()) | set(summary_b.get('by_page', {}).keys())
    pages = sorted(
        [int(p) for p in all_pages if (compare_pages is None or int(p) in compare_pages)]
    )

    for page_num in pages:
        page_key = str(page_num)
        items_a = {it['panel_name']: it for it in summary_a.get('by_page', {}).get(page_key, [])}
        items_b = {it['panel_name']: it for it in summary_b.get('by_page', {}).get(page_key, [])}
        all_panels = sorted(set(items_a.keys()) | set(items_b.keys()))

        if not all_panels:
            continue

        print(f'\nPage {page_num} comparison — {len(all_panels)} panels  |  {label_a}: {len(items_a)}  |  {label_b}: {len(items_b)}')

        n_rows = len(all_panels)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_rows * 3.5), squeeze=False)

        for row, panel_name in enumerate(all_panels):
            for col, (items, label) in enumerate([(items_a, label_a), (items_b, label_b)]):
                ax = axes[row][col]
                it = items.get(panel_name)
                if it and it.get('file'):
                    img_path = notebook_dir / it['file']
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        ax.set_title(f'[{label}]\n{panel_name}', fontsize=8)
                    else:
                        ax.set_title(f'[{label}]\n{panel_name}\n(file read failed)', fontsize=8)
                else:
                    ax.set_facecolor('#f0f0f0')
                    ax.text(0.5, 0.5, f'no result\n({panel_name})',
                            ha='center', va='center', fontsize=9, color='gray',
                            transform=ax.transAxes)
                    ax.set_title(f'[{label}]\n{panel_name}', fontsize=8)
                ax.axis('off')

        plt.suptitle(f'Page {page_num} — {label_a}  vs  {label_b}', fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.show()
