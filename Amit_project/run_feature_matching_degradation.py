"""
Robustness sweep for feature matching (Part 2 + 'per SNR' requirement) --
fully in-memory, no augmented dataset is ever written to disk.

1. Baseline: run process_pair() on the clean frame pairs (real feature
   matching quality -- this doubles as the pseudo-GT for the following
   steps, per the course project's stated methodology).
2. For each of the 3 distortions (motion_blur, low_light, rain) and each
   of NUM_LEVELS severity levels: apply the distortion to each frame in
   RAM, compute its achieved SNR on the spot, and re-run the exact same
   matching pipeline (reused from feature_matching_bdd100k.py) on the
   in-memory arrays. Only a handful of before/after visualizations are
   saved to disk (one per augmentation/level, for the report).
3. Enhancement (Part 3): for the same augmented frame, also apply the
   matching restoration function from enhancements.py and re-run the
   pipeline again -- lets us compare distorted vs. enhanced vs. clean
   baseline.
4. Plot match-accuracy (inlier_ratio, match_ratio) vs SNR per distortion,
   with the clean baseline as a reference line, plus a distorted-vs-
   enhanced-vs-clean comparison bar chart per distortion.

Usage examples
--------------
Baseline (clean) run, both experiments:
    python run_feature_matching_degradation.py
    python run_feature_matching_degradation.py --clean_dir <custom_path> --out_dir <custom_path>
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import enhancements

import argparse
import os

import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_matching_bdd100k import process_pair, collect_pairs_from_dataset
from augmentation_levels import make_augmentation_fns, compute_snr_db, AUGMENTATIONS


def run_baseline(clean_dir, vis_root, method, n_features, ratio_thresh, ransac_thresh):
    pairs = collect_pairs_from_dataset(clean_dir, num_pairs=100000, mode="sequence", frame_gap=1)
    print(f"[baseline] {len(pairs)} clean frame pairs")

    rows = []
    baseline_min_kp = {}
    vis_dir = os.path.join(vis_root, "baseline_visualizations")
    for i, (p1, p2) in enumerate(pairs):
        m = process_pair(p1, p2, method=method, n_features=n_features,
                          ratio_thresh=ratio_thresh, ransac_thresh=ransac_thresh,
                          output_dir=vis_dir, save_visualization=(i < 5))
        if m:
            rows.append(m)
            baseline_min_kp[(m["img1"], m["img2"])] = min(m["n_keypoints_img1"], m["n_keypoints_img2"])
    return pairs, pd.DataFrame(rows), baseline_min_kp


def run_degraded_in_memory(pairs, baseline_min_kp, vis_root, method, n_features, ratio_thresh,
                            ransac_thresh, num_levels):
    """Returns (distorted_df, enhanced_df) -- same schema, one row per
    (pair, augmentation, level); enhanced_df is the same augmented frame
    additionally passed through enhancements.ENHANCEMENTS[aug_name].

    Each row's "match_ratio" (from process_pair) is normalized by the
    CURRENT image's own keypoint count, which collapses under severe
    distortion/inflates under contrast-boosting enhancement -- see
    TASKS.md's Phase 2 addendum. "match_ratio_vs_baseline" instead
    normalizes by the CLEAN baseline's (fixed) keypoint count for that same
    pair, so it behaves monotonically with severity and fairly rewards
    enhancement that recovers real matches."""
    fns = make_augmentation_fns(num_levels)

    unique_paths = sorted({p for pair in pairs for p in pair})
    clean_cache = {p: cv2.imread(p, cv2.IMREAD_COLOR) for p in unique_paths}
    print(f"[degraded] cached {len(clean_cache)} clean frames in memory")

    distorted_rows, enhanced_rows = [], []
    for aug_name, apply_fn in fns.items():
        restore_fn = enhancements.ENHANCEMENTS[aug_name]
        for level_idx in range(num_levels):
            level = level_idx + 1
            vis_dir = os.path.join(vis_root, "degraded_visualizations", aug_name, f"level_{level}")
            enh_vis_dir = os.path.join(vis_root, "enhanced_visualizations", aug_name, f"level_{level}")
            print(f"[degraded] {aug_name} level_{level} ({len(pairs)} pairs)")
            for i, (p1, p2) in enumerate(pairs):
                c1, c2 = clean_cache[p1], clean_cache[p2]
                if c1 is None or c2 is None:
                    continue
                a1 = apply_fn(c1, level_idx)
                a2 = apply_fn(c2, level_idx)
                snr = (compute_snr_db(c1, a1) + compute_snr_db(c2, a2)) / 2.0

                ref_kp = baseline_min_kp.get((os.path.basename(p1), os.path.basename(p2)))

                m = process_pair(p1, p2, method=method, n_features=n_features,
                                  ratio_thresh=ratio_thresh, ransac_thresh=ransac_thresh,
                                  output_dir=vis_dir, save_visualization=(i == 0),
                                  img1_override=a1, img2_override=a2)
                if m:
                    m["augmentation"] = aug_name
                    m["level"] = level
                    m["snr_db"] = snr
                    m["match_ratio_vs_baseline"] = (m["n_good_matches"] / ref_kp) if ref_kp else float("nan")
                    distorted_rows.append(m)

                e1, e2 = restore_fn(a1), restore_fn(a2)
                me = process_pair(p1, p2, method=method, n_features=n_features,
                                   ratio_thresh=ratio_thresh, ransac_thresh=ransac_thresh,
                                   output_dir=enh_vis_dir, save_visualization=(i == 0),
                                   img1_override=e1, img2_override=e2)
                if me:
                    me["augmentation"] = aug_name
                    me["level"] = level
                    me["snr_db"] = snr
                    me["match_ratio_vs_baseline"] = (me["n_good_matches"] / ref_kp) if ref_kp else float("nan")
                    enhanced_rows.append(me)
    return pd.DataFrame(distorted_rows), pd.DataFrame(enhanced_rows)


def plot_metric_vs_snr(level_summary, baseline_value, metric, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for aug_name, group in level_summary.groupby("augmentation"):
        group = group.sort_values("snr_db")
        ax.plot(group["snr_db"], group[metric], marker="o", label=aug_name)
    ax.axhline(baseline_value, linestyle="--", color="red", label=f"clean baseline {baseline_value:.3f}")
    ax.set_xlabel("SNR (dB)")
    ax.invert_xaxis()
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs SNR -- degradation sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_enhancement_comparison(distorted_df, enhanced_df, baseline_value, metric, ylabel, out_path):
    """Distorted vs. enhanced vs. clean-baseline bar chart, one bar pair per
    distortion (mean over all levels/pairs) -- course brief's own example
    style (slide 31)."""
    dist_means = distorted_df.groupby("augmentation")[metric].mean()
    enh_means = enhanced_df.groupby("augmentation")[metric].mean()
    augs = list(dist_means.index)

    x = range(len(augs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], [dist_means[a] for a in augs], width, label="distorted")
    ax.bar([i + width / 2 for i in x], [enh_means.get(a, float("nan")) for a in augs], width, label="enhanced")
    ax.axhline(baseline_value, linestyle="--", color="red", label=f"clean baseline {baseline_value:.3f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(augs)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} -- distorted vs. enhanced (mean over all levels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Task 2 - Feature matching evaluation")
    parser.add_argument("--clean_dir", default=str(config.TASK2_CLEAN_DIR))
    parser.add_argument("--out_dir", default=None,
                         help="Legacy: if given, ALL output (CSVs/plots/visualizations) goes into "
                              "this single flat folder (old behavior). If omitted (default), output "
                              "goes to the unified outputs/{csv_results,graph_results,visualizations}/task2/ layout.")
    parser.add_argument("--method", default="orb", choices=["orb", "sift", "akaze"])
    parser.add_argument("--n_features", type=int, default=2000)
    parser.add_argument("--ratio_thresh", type=float, default=0.75)
    parser.add_argument("--ransac_thresh", type=float, default=5.0)
    parser.add_argument("--num_levels", type=int, default=9)
    args = parser.parse_args()

    if args.out_dir is not None:
        csv_dir = graph_dir = vis_dir = Path(args.out_dir)
    else:
        csv_dir, graph_dir, vis_dir = config.TASK2_CSV_DIR, config.TASK2_GRAPH_DIR, config.TASK2_VIS_DIR
    for d in (csv_dir, graph_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    pairs, baseline_df, baseline_min_kp = run_baseline(
        args.clean_dir, str(vis_dir), args.method,
        args.n_features, args.ratio_thresh, args.ransac_thresh)
    baseline_df.to_csv(csv_dir / "baseline_per_pair.csv", index=False)

    distorted_df, enhanced_df = run_degraded_in_memory(
        pairs, baseline_min_kp, str(vis_dir), args.method, args.n_features, args.ratio_thresh,
        args.ransac_thresh, args.num_levels)
    distorted_df.to_csv(csv_dir / "degraded_per_pair.csv", index=False)
    enhanced_df.to_csv(csv_dir / "enhanced_per_pair.csv", index=False)

    level_summary = distorted_df.groupby(["augmentation", "level"]).agg(
        inlier_ratio=("inlier_ratio", "mean"),
        match_ratio=("match_ratio", "mean"),
        match_ratio_vs_baseline=("match_ratio_vs_baseline", "mean"),
        n_good_matches=("n_good_matches", "mean"),
        snr_db=("snr_db", "mean"),
    ).reset_index()
    level_summary.to_csv(csv_dir / "level_summary.csv", index=False)

    baseline_inlier = baseline_df["inlier_ratio"].mean()
    # baseline's own match_ratio_vs_baseline is 1.0 by definition (n_good_matches
    # measured against its own keypoint count, which IS the reference) -- use
    # its raw match_ratio as the reference line instead, for a meaningful number.
    baseline_match = baseline_df["match_ratio"].mean()

    plot_metric_vs_snr(level_summary, baseline_inlier, "inlier_ratio",
                        "Inlier ratio (match accuracy)",
                        graph_dir / "inlier_ratio_vs_snr.png")
    plot_metric_vs_snr(level_summary, baseline_match, "match_ratio_vs_baseline",
                        "Match ratio (good matches / clean-baseline keypoints)",
                        graph_dir / "match_ratio_vs_snr.png")

    plot_enhancement_comparison(distorted_df, enhanced_df, baseline_inlier, "inlier_ratio",
                                 "Inlier ratio (match accuracy)",
                                 graph_dir / "inlier_ratio_per_distortion.png")
    plot_enhancement_comparison(distorted_df, enhanced_df, baseline_match, "match_ratio_vs_baseline",
                                 "Match ratio (good matches / clean-baseline keypoints)",
                                 graph_dir / "match_ratio_per_distortion.png")

    print("\n=== Baseline (clean) ===")
    print(f"  inlier_ratio={baseline_inlier:.3f}  match_ratio={baseline_match:.3f}")
    print("\n=== Level summary (mean per augmentation/level, distorted) ===")
    print(level_summary.to_string(index=False))
    print("\n=== Enhanced vs distorted (mean over all levels) ===")
    print("distorted match_ratio_vs_baseline:", distorted_df.groupby("augmentation")["match_ratio_vs_baseline"].mean().to_dict())
    print("enhanced  match_ratio_vs_baseline:", enhanced_df.groupby("augmentation")["match_ratio_vs_baseline"].mean().to_dict())
    print("distorted inlier_ratio:", distorted_df.groupby("augmentation")["inlier_ratio"].mean().to_dict())
    print("enhanced  inlier_ratio:", enhanced_df.groupby("augmentation")["inlier_ratio"].mean().to_dict())
    print(f"\nCSVs written to {csv_dir.resolve()}")
    print(f"Plots written to {graph_dir.resolve()}")
    print(f"Visualizations written to {vis_dir.resolve()}")


if __name__ == "__main__":
    main()
