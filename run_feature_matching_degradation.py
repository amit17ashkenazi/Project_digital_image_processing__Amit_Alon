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
3. Plot match-accuracy (inlier_ratio, match_ratio) vs SNR per distortion,
   with the clean baseline as a reference line.

Usage:
    python run_feature_matching_degradation.py \
        --clean_dir "C:\\Users\\amit\\Desktop\\data1\\2_data_feature_matching_high_level" \
        --out_dir results/task2_feature_matching
"""
from __future__ import annotations

import argparse
import os

import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_matching_bdd100k import process_pair, collect_pairs_from_dataset
from augmentation_levels import make_augmentation_fns, compute_snr_db, AUGMENTATIONS


def run_baseline(clean_dir, out_dir, method, n_features, ratio_thresh, ransac_thresh):
    pairs = collect_pairs_from_dataset(clean_dir, num_pairs=100000, mode="sequence", frame_gap=1)
    print(f"[baseline] {len(pairs)} clean frame pairs")

    rows = []
    vis_dir = os.path.join(out_dir, "baseline_visualizations")
    for i, (p1, p2) in enumerate(pairs):
        m = process_pair(p1, p2, method=method, n_features=n_features,
                          ratio_thresh=ratio_thresh, ransac_thresh=ransac_thresh,
                          output_dir=vis_dir, save_visualization=(i < 5))
        if m:
            rows.append(m)
    return pairs, pd.DataFrame(rows)


def run_degraded_in_memory(pairs, out_dir, method, n_features, ratio_thresh, ransac_thresh, num_levels):
    fns = make_augmentation_fns(num_levels)

    # Load every unique clean frame once, keep in RAM (frames are reused
    # across neighboring pairs).
    unique_paths = sorted({p for pair in pairs for p in pair})
    clean_cache = {p: cv2.imread(p, cv2.IMREAD_COLOR) for p in unique_paths}
    print(f"[degraded] cached {len(clean_cache)} clean frames in memory")

    rows = []
    for aug_name, apply_fn in fns.items():
        for level_idx in range(num_levels):
            level = level_idx + 1
            vis_dir = os.path.join(out_dir, "degraded_visualizations", aug_name, f"level_{level}")
            print(f"[degraded] {aug_name} level_{level} ({len(pairs)} pairs)")
            for i, (p1, p2) in enumerate(pairs):
                c1, c2 = clean_cache[p1], clean_cache[p2]
                if c1 is None or c2 is None:
                    continue
                a1 = apply_fn(c1, level_idx)
                a2 = apply_fn(c2, level_idx)
                snr = (compute_snr_db(c1, a1) + compute_snr_db(c2, a2)) / 2.0

                m = process_pair(p1, p2, method=method, n_features=n_features,
                                  ratio_thresh=ratio_thresh, ransac_thresh=ransac_thresh,
                                  output_dir=vis_dir, save_visualization=(i == 0),
                                  img1_override=a1, img2_override=a2)
                if m:
                    m["augmentation"] = aug_name
                    m["level"] = level
                    m["snr_db"] = snr
                    rows.append(m)
    return pd.DataFrame(rows)


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


def main():
    parser = argparse.ArgumentParser(description="Feature matching robustness sweep vs SNR (in-memory)")
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--method", default="orb", choices=["orb", "sift", "akaze"])
    parser.add_argument("--n_features", type=int, default=2000)
    parser.add_argument("--ratio_thresh", type=float, default=0.75)
    parser.add_argument("--ransac_thresh", type=float, default=5.0)
    parser.add_argument("--num_levels", type=int, default=9)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pairs, baseline_df = run_baseline(args.clean_dir, args.out_dir, args.method,
                                       args.n_features, args.ratio_thresh, args.ransac_thresh)
    baseline_df.to_csv(os.path.join(args.out_dir, "baseline_per_pair.csv"), index=False)

    degraded_df = run_degraded_in_memory(pairs, args.out_dir, args.method,
                                          args.n_features, args.ratio_thresh, args.ransac_thresh,
                                          args.num_levels)
    degraded_df.to_csv(os.path.join(args.out_dir, "degraded_per_pair.csv"), index=False)

    level_summary = degraded_df.groupby(["augmentation", "level"]).agg(
        inlier_ratio=("inlier_ratio", "mean"),
        match_ratio=("match_ratio", "mean"),
        n_good_matches=("n_good_matches", "mean"),
        snr_db=("snr_db", "mean"),
    ).reset_index()
    level_summary.to_csv(os.path.join(args.out_dir, "level_summary.csv"), index=False)

    baseline_inlier = baseline_df["inlier_ratio"].mean()
    baseline_match = baseline_df["match_ratio"].mean()

    plot_metric_vs_snr(level_summary, baseline_inlier, "inlier_ratio",
                        "Inlier ratio (match accuracy)",
                        os.path.join(args.out_dir, "inlier_ratio_vs_snr.png"))
    plot_metric_vs_snr(level_summary, baseline_match, "match_ratio",
                        "Match ratio (good matches / min keypoints)",
                        os.path.join(args.out_dir, "match_ratio_vs_snr.png"))

    print("\n=== Baseline (clean) ===")
    print(f"  inlier_ratio={baseline_inlier:.3f}  match_ratio={baseline_match:.3f}")
    print("\n=== Level summary (mean per augmentation/level) ===")
    print(level_summary.to_string(index=False))
    print(f"\nResults written to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
