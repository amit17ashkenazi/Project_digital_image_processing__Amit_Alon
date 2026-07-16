"""
Robustness sweep for vehicle detection (Part 2 + 'per SNR' requirement) --
fully in-memory, no augmented dataset is ever written to disk.

1. Baseline: run process_image() on the clean images (reused from
   vehicle_detection_bdd100k.py). Since no BDD100K box2d labels ship with
   this local image subset, these clean-image detections double as the
   pseudo-GT for the following steps, per the course project's stated
   methodology ("baseline vs GT, use as GT for following steps").
2. For each of the 3 distortions (motion_blur, low_light, rain) and each
   of NUM_LEVELS severity levels: apply the distortion in RAM, compute its
   achieved SNR on the spot, run the same detector on the in-memory array,
   and IoU-match the result against the baseline boxes for that image
   (greedy, highest-IoU-first). Only a handful of before/after
   visualizations are saved to disk (for the report).
3. Plot detection recall / mean IoU vs SNR per distortion, with the clean
   baseline (recall=1.0) as a reference line.

Usage (run from inside Amit_project/):
    python run_vehicle_detection_degradation.py
    python run_vehicle_detection_degradation.py --clean_dir <custom_path> --out_dir <custom_path>
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vehicle_detection_bdd100k import load_model, process_image, collect_images_from_dataset
from augmentation_levels import make_augmentation_fns, compute_snr_db, AUGMENTATIONS


def iou(box_a, box_b) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(ref_boxes, cand_boxes, iou_thresh):
    candidates = []
    for i, rb in enumerate(ref_boxes):
        for j, cb in enumerate(cand_boxes):
            v = iou(rb, cb)
            if v >= iou_thresh:
                candidates.append((i, j, v))
    candidates.sort(key=lambda x: -x[2])

    used_ref, used_cand, matches = set(), set(), []
    for i, j, v in candidates:
        if i in used_ref or j in used_cand:
            continue
        used_ref.add(i)
        used_cand.add(j)
        matches.append((i, j, v))
    return matches


def boxes_xyxy(detections):
    return [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections]


def run_baseline(model, clean_dir, out_dir, conf, iou_thresh):
    image_paths = collect_images_from_dataset(clean_dir, num_images=100000)
    print(f"[baseline] {len(image_paths)} clean images")

    baseline_boxes = {}
    vis_dir = os.path.join(out_dir, "baseline_visualizations")
    for i, path in enumerate(image_paths):
        _, detections = process_image(model, path, conf_thresh=conf, iou_thresh=iou_thresh,
                                       output_dir=vis_dir, save_visualization=(i < 5))
        baseline_boxes[os.path.basename(path)] = detections
    return image_paths, baseline_boxes


def run_degraded_in_memory(model, image_paths, baseline_boxes, out_dir, conf, iou_thresh,
                            match_iou_thresh, num_levels):
    fns = make_augmentation_fns(num_levels)

    print(f"[degraded] caching {len(image_paths)} clean images in memory")
    clean_cache = {p: cv2.imread(p) for p in image_paths}

    rows = []
    for aug_name, apply_fn in fns.items():
        for level_idx in range(num_levels):
            level = level_idx + 1
            vis_dir = os.path.join(out_dir, "degraded_visualizations", aug_name, f"level_{level}")
            print(f"[degraded] {aug_name} level_{level} ({len(image_paths)} images)")
            for i, path in enumerate(image_paths):
                filename = os.path.basename(path)
                clean_img = clean_cache[path]
                if clean_img is None:
                    continue
                aug_img = apply_fn(clean_img, level_idx)
                snr = compute_snr_db(clean_img, aug_img)

                _, detections = process_image(model, path, conf_thresh=conf, iou_thresh=iou_thresh,
                                               output_dir=vis_dir, save_visualization=(i == 0),
                                               image_override=aug_img)

                ref_boxes = boxes_xyxy(baseline_boxes.get(filename, []))
                cand_boxes = boxes_xyxy(detections)
                matches = greedy_match(ref_boxes, cand_boxes, match_iou_thresh)

                ref_count, cand_count, matched_count = len(ref_boxes), len(cand_boxes), len(matches)
                matched_recall = (matched_count / ref_count) if ref_count > 0 else np.nan
                retention_ratio = (cand_count / ref_count) if ref_count > 0 else np.nan
                mean_iou_matched = float(np.mean([v for _, _, v in matches])) if matches else np.nan

                rows.append({
                    "image": filename,
                    "augmentation": aug_name,
                    "level": level,
                    "snr_db": snr,
                    "ref_count": ref_count,
                    "cand_count": cand_count,
                    "matched_count": matched_count,
                    "matched_recall": matched_recall,
                    "retention_ratio": retention_ratio,
                    "mean_iou_matched": mean_iou_matched,
                })
    return pd.DataFrame(rows)


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


def main():
    parser = argparse.ArgumentParser(description="Vehicle detection robustness sweep vs SNR (in-memory)")
    parser.add_argument("--clean_dir", default=str(config.TASK3_CLEAN_DIR))
    parser.add_argument("--out_dir", default=str(config.TASK3_RESULTS_DIR))
    parser.add_argument("--model", default=str(config.YOLO_WEIGHTS_PATH))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold used by the detector")
    parser.add_argument("--match_iou_thresh", type=float, default=0.5, help="IoU threshold for matching a detection to a baseline box")
    parser.add_argument("--num_levels", type=int, default=9)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.model)

    image_paths, baseline_boxes = run_baseline(model, args.clean_dir, args.out_dir, args.conf, args.iou)

    degraded_df = run_degraded_in_memory(model, image_paths, baseline_boxes, args.out_dir,
                                          args.conf, args.iou, args.match_iou_thresh, args.num_levels)
    degraded_df.to_csv(os.path.join(args.out_dir, "degraded_per_image.csv"), index=False)

    level_summary = degraded_df.groupby(["augmentation", "level"]).agg(
        matched_recall=("matched_recall", "mean"),
        retention_ratio=("retention_ratio", "mean"),
        mean_iou_matched=("mean_iou_matched", "mean"),
        snr_db=("snr_db", "mean"),
    ).reset_index()
    level_summary.to_csv(os.path.join(args.out_dir, "level_summary.csv"), index=False)

    plot_metric_vs_snr(level_summary, 1.0, "matched_recall",
                        "Detection recall vs clean baseline",
                        os.path.join(args.out_dir, "recall_vs_snr.png"))
    plot_metric_vs_snr(level_summary, 1.0, "mean_iou_matched",
                        "Mean IoU of matched detections",
                        os.path.join(args.out_dir, "iou_vs_snr.png"))

    print("\n=== Level summary (mean per augmentation/level) ===")
    print(level_summary.to_string(index=False))
    print(f"\nResults written to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()