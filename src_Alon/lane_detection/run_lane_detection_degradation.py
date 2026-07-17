"""
Robustness sweep for lane detection (Part 2/3 + 'per SNR' requirement) --
fully in-memory, no augmented dataset is ever written to disk. Replaces the
old disk-based create_augmented_images.py + compere_gt_to_aug.py pair (see
TASKS.md Phase 3) with the same in-memory pattern Tasks 2/3 already use.

GT note (see TASKS.md 3.2): real BDD100K lane-marking annotations are a
SEPARATE label release from the det_v2 detection labels we have locally (the
det_v2 files only contain box2d object categories -- no "lane" category at
all). Until that separate lane-label file is sourced, this driver measures
robustness the same way the legacy compere_gt_to_aug.py did: augmented/
enhanced detection vs. the CLEAN-image detection on the same frame
(pseudo-GT, same methodology already used for Task 2). This is a known,
documented limitation, not an oversight -- see README section 11.

1. Baseline: run process_image() on the clean images (also the pseudo-GT
   reference for the following steps).
2. For each of the 3 distortions and each of NUM_LEVELS severity levels:
   apply the distortion in RAM, compute its achieved SNR, re-run the lane
   pipeline, and diff the detected line endpoints against the clean-image
   baseline lines.
3. Enhancement (Part 3): same augmented frame additionally passed through
   enhancements.ENHANCEMENTS[aug_name], re-run, diffed the same way.
4. Plot lane-offset-error / survival-rate vs SNR per distortion, plus a
   distorted-vs-enhanced-vs-clean comparison.

Usage (run from inside src_Alon/lane_detection/):
    python run_lane_detection_degradation.py
    python run_lane_detection_degradation.py --clean_dir <custom_path> --out_dir <custom_path>
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
import enhancements

sys.path.insert(0, str(config.PROJECT_ROOT / "Amit_project"))
from augmentation_levels import make_augmentation_fns, compute_snr_db

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lane_detection_pipeline import process_image

MAX_ERROR_PX = config.MAX_ERROR_PX  # "survival" tolerance, same as legacy threshold_survival.py


def list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)


def line_diff(line_a, line_b):
    """Pixel offset at the bottom (y1) endpoint -- same definition the
    legacy compere_gt_to_aug.py used."""
    if line_a is None or line_b is None:
        return None
    ax1, _, _, _ = line_a
    bx1, _, _, _ = line_b
    return abs(ax1 - bx1)


def _save_visualization(result, vis_dir: Path, filename: str):
    if result is None:
        return
    vis_dir.mkdir(parents=True, exist_ok=True)
    stem = os.path.splitext(filename)[0]
    cv2.imwrite(str(vis_dir / f"lanes_{stem}.png"), result["overlay_img"])


def run_baseline(clean_dir, vis_root: Path, num_visualize=5):
    image_paths = list_images(clean_dir)
    print(f"[baseline] {len(image_paths)} clean images")

    baseline_results = {}
    rows = []
    for i, path in enumerate(image_paths):
        result = process_image(str(path), show=False)
        if result is None:
            continue
        baseline_results[path.name] = result
        rows.append({
            "image": path.name,
            "left_detected": result["left_lane"] is not None,
            "right_detected": result["right_lane"] is not None,
        })
        if i < num_visualize:
            _save_visualization(result, vis_root / "baseline_visualizations", path.name)
    return image_paths, baseline_results, pd.DataFrame(rows)


def run_degraded_in_memory(image_paths, baseline_results, vis_root: Path, num_levels):
    """Returns (distorted_df, enhanced_df) -- one row per (image,
    augmentation, level), diffing detected lane-line endpoints against the
    clean-image baseline (pseudo-GT, see module docstring)."""
    fns = make_augmentation_fns(num_levels)

    print(f"[degraded] caching {len(image_paths)} clean images in memory")
    clean_cache = {p: cv2.imread(str(p)) for p in image_paths}

    distorted_rows, enhanced_rows = [], []
    for aug_name, apply_fn in fns.items():
        restore_fn = enhancements.ENHANCEMENTS[aug_name]
        for level_idx in range(num_levels):
            level = level_idx + 1
            print(f"[degraded] {aug_name} level_{level} ({len(image_paths)} images)")
            for i, path in enumerate(image_paths):
                filename = path.name
                baseline = baseline_results.get(filename)
                clean_img = clean_cache[path]
                if baseline is None or clean_img is None:
                    continue

                aug_img = apply_fn(clean_img, level_idx)
                snr = compute_snr_db(clean_img, aug_img)

                aug_result = process_image(str(path), show=False, image_override=aug_img)
                distorted_rows.append(_diff_row(filename, aug_name, level, snr, baseline, aug_result))
                if i == 0:
                    _save_visualization(aug_result, vis_root / "degraded_visualizations" / aug_name / f"level_{level}", filename)

                enh_img = restore_fn(aug_img)
                enh_result = process_image(str(path), show=False, image_override=enh_img)
                enhanced_rows.append(_diff_row(filename, aug_name, level, snr, baseline, enh_result))
                if i == 0:
                    _save_visualization(enh_result, vis_root / "enhanced_visualizations" / aug_name / f"level_{level}", filename)

    return pd.DataFrame(distorted_rows), pd.DataFrame(enhanced_rows)


def _diff_row(filename, aug_name, level, snr, baseline, result):
    if result is None:
        left_diff = right_diff = None
        left_lost = baseline["left_lane"] is not None
        right_lost = baseline["right_lane"] is not None
    else:
        left_diff = line_diff(baseline["left_lane"], result["left_lane"])
        right_diff = line_diff(baseline["right_lane"], result["right_lane"])
        left_lost = baseline["left_lane"] is not None and result["left_lane"] is None
        right_lost = baseline["right_lane"] is not None and result["right_lane"] is None
    return {
        "image": filename,
        "augmentation": aug_name,
        "level": level,
        "snr_db": snr,
        "left_diff_px": left_diff,
        "right_diff_px": right_diff,
        "left_lost": left_lost,
        "right_lost": right_lost,
    }


def survival_rate(df: pd.DataFrame, max_error=MAX_ERROR_PX) -> float:
    """Fraction of (image, side) pairs where the line was neither lost nor
    displaced more than max_error px -- same definition as the legacy
    threshold_survival.py."""
    total = survived = 0
    for _, r in df.iterrows():
        for side in ("left", "right"):
            diff = r[f"{side}_diff_px"]
            lost = r[f"{side}_lost"]
            if pd.isna(diff) and not lost:
                continue
            total += 1
            if not lost and not pd.isna(diff) and diff <= max_error:
                survived += 1
    return survived / total if total > 0 else np.nan


def plot_metric_vs_snr(level_summary, baseline_value, metric, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for aug_name, group in level_summary.groupby("augmentation"):
        group = group.sort_values("snr_db")
        ax.plot(group["snr_db"], group[metric], marker="o", label=aug_name)
    ax.axhline(baseline_value, linestyle="--", color="red", label=f"clean baseline {baseline_value:.2f}")
    ax.set_xlabel("SNR (dB)")
    ax.invert_xaxis()
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs SNR -- degradation sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_enhancement_comparison(distorted_df, enhanced_df, baseline_value, metric, ylabel, out_path):
    dist_means = distorted_df.groupby("augmentation")[metric].mean()
    enh_means = enhanced_df.groupby("augmentation")[metric].mean()
    augs = list(dist_means.index)

    x = range(len(augs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], [dist_means[a] for a in augs], width, label="distorted")
    ax.bar([i + width / 2 for i in x], [enh_means.get(a, float("nan")) for a in augs], width, label="enhanced")
    ax.axhline(baseline_value, linestyle="--", color="red", label=f"clean baseline {baseline_value:.2f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(augs)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} -- distorted vs. enhanced (mean over all levels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Task 1 - Lane detection robustness sweep vs SNR (in-memory)")
    parser.add_argument("--clean_dir", default=str(config.TASK1_CLEAN_DIR))
    parser.add_argument("--csv_dir", default=str(config.TASK1_CSV_DIR))
    parser.add_argument("--graph_dir", default=str(config.TASK1_GRAPH_DIR))
    parser.add_argument("--vis_dir", default=str(config.TASK1_VIS_DIR))
    parser.add_argument("--num_levels", type=int, default=9)
    args = parser.parse_args()

    csv_dir, graph_dir, vis_dir = Path(args.csv_dir), Path(args.graph_dir), Path(args.vis_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths, baseline_results, baseline_df = run_baseline(args.clean_dir, vis_dir)
    baseline_df.to_csv(csv_dir / "baseline_per_image.csv", index=False)

    distorted_df, enhanced_df = run_degraded_in_memory(image_paths, baseline_results, vis_dir, args.num_levels)
    distorted_df.to_csv(csv_dir / "degraded_per_image.csv", index=False)
    enhanced_df.to_csv(csv_dir / "enhanced_per_image.csv", index=False)

    rows = []
    for (aug, level), group in distorted_df.groupby(["augmentation", "level"]):
        rows.append({
            "augmentation": aug, "level": level,
            "snr_db": group["snr_db"].mean(),
            "survival_rate": survival_rate(group),
            "mean_left_diff_px": group["left_diff_px"].mean(),
            "mean_right_diff_px": group["right_diff_px"].mean(),
        })
    level_summary = pd.DataFrame(rows)
    level_summary.to_csv(csv_dir / "level_summary.csv", index=False)

    print("\n=== Level summary (mean per augmentation/level, distorted) ===")
    print(level_summary.to_string(index=False))

    plot_metric_vs_snr(level_summary, 1.0, "survival_rate",
                        "Lane-line survival rate (vs. clean baseline)",
                        graph_dir / "lane_offset_vs_snr.png")

    plot_enhancement_comparison(
        distorted_df.groupby("augmentation").apply(lambda g: pd.Series({"survival_rate": survival_rate(g)})).reset_index(),
        enhanced_df.groupby("augmentation").apply(lambda g: pd.Series({"survival_rate": survival_rate(g)})).reset_index(),
        1.0, "survival_rate", "Lane-line survival rate (vs. clean baseline)",
        graph_dir / "lane_offset_per_distortion.png")

    print(f"\nResults written to {csv_dir.resolve()} and {graph_dir.resolve()}")


if __name__ == "__main__":
    main()
