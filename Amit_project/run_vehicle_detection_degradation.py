"""
Robustness sweep for vehicle detection (Part 2 + 'per SNR'/'per class'
requirements) -- fully in-memory, no augmented dataset is ever written to
disk.

GT methodology (real, not pseudo -- see TASKS.md Phase 1): reference boxes
come from config.TASK3_GT_CSV (real BDD100K box2d annotations, extracted by
src_Alon/helper_files/extract_task3_gt.py), not from the clean-image
detection run. The clean-image detection run is still performed, but now
purely as "baseline performance vs. real GT" (course brief Part 1), saved to
baseline_vs_gt_metrics.csv.

1. Baseline: run process_image() on the clean images, IoU-match (class-aware)
   against real GT -> baseline_vs_gt_metrics.csv. Mean baseline recall is
   used as the reference line in the SNR plots (no longer a hardcoded 1.0).
2. For each of the 3 distortions and each of NUM_LEVELS severity levels:
   apply the distortion in RAM, compute its achieved SNR, run the same
   detector on the in-memory array, and IoU-match (class-aware) against real
   GT. Metrics are reported both per-class and aggregated ("all").
3. Plot detection recall / mean IoU vs SNR per distortion, per class, with
   the real baseline as a reference line.

Usage (run from inside Amit_project/):
    python run_vehicle_detection_degradation.py
    python run_vehicle_detection_degradation.py --clean_dir <custom_path> --out_dir <custom_path>
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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vehicle_detection_bdd100k import load_model, process_image, collect_images_from_dataset
from augmentation_levels import make_augmentation_fns, compute_snr_db, AUGMENTATIONS

VEHICLE_CLASSES = sorted(set(config.BDD100K_TO_COCO_VEHICLE_CLASS.values()))


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


def greedy_match(ref_boxes, ref_classes, cand_boxes, cand_classes, iou_thresh):
    """Class-aware greedy IoU matching: a candidate box can only match a
    reference box of the SAME class. Highest-IoU-first, one-to-one."""
    candidates = []
    for i, rb in enumerate(ref_boxes):
        for j, cb in enumerate(cand_boxes):
            if ref_classes[i] != cand_classes[j]:
                continue
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


def boxes_xyxy(items):
    return [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in items]


def boxes_classes(items):
    return [d["class"] for d in items]


def load_gt_boxes(gt_csv: Path) -> dict[str, list[dict]]:
    """{image_name: [{"class":.., "x1":.., "y1":.., "x2":.., "y2":..}, ...]}"""
    df = pd.read_csv(gt_csv)
    gt_boxes: dict[str, list[dict]] = {}
    for image, group in df.groupby("image"):
        gt_boxes[image] = group[["class", "x1", "y1", "x2", "y2"]].to_dict("records")
    return gt_boxes


def per_class_and_overall_rows(ref_items, cand_items, match_iou_thresh, extra_fields):
    """Given GT (ref) and detection (cand) items for ONE image/condition,
    returns a list of row dicts: one per vehicle class present in ref/cand,
    plus one with class="all" aggregating across all classes."""
    ref_boxes, ref_classes = boxes_xyxy(ref_items), boxes_classes(ref_items)
    cand_boxes, cand_classes = boxes_xyxy(cand_items), boxes_classes(cand_items)
    matches = greedy_match(ref_boxes, ref_classes, cand_boxes, cand_classes, match_iou_thresh)

    rows = []
    classes_present = sorted(set(ref_classes) | set(cand_classes)) or []
    for cls in classes_present + ["all"]:
        if cls == "all":
            idxs_ref = range(len(ref_boxes))
            idxs_cand = range(len(cand_boxes))
            cls_matches = matches
        else:
            idxs_ref = [i for i, c in enumerate(ref_classes) if c == cls]
            idxs_cand = [j for j, c in enumerate(cand_classes) if c == cls]
            cls_matches = [(i, j, v) for i, j, v in matches if ref_classes[i] == cls]

        ref_count, cand_count, matched_count = len(idxs_ref), len(idxs_cand), len(cls_matches)
        matched_recall = (matched_count / ref_count) if ref_count > 0 else np.nan
        retention_ratio = (cand_count / ref_count) if ref_count > 0 else np.nan
        mean_iou_matched = float(np.mean([v for _, _, v in cls_matches])) if cls_matches else np.nan

        rows.append({
            **extra_fields,
            "class": cls,
            "ref_count": ref_count,
            "cand_count": cand_count,
            "matched_count": matched_count,
            "matched_recall": matched_recall,
            "retention_ratio": retention_ratio,
            "mean_iou_matched": mean_iou_matched,
        })
    return rows


def run_baseline_vs_gt(model, clean_dir, gt_boxes, vis_root, conf, iou_thresh, match_iou_thresh):
    """Baseline performance vs. real GT (course brief Part 1) -- no longer
    the source of reference boxes for the degradation sweep, just a measured
    result in its own right."""
    image_paths = collect_images_from_dataset(clean_dir, num_images=100000)
    print(f"[baseline] {len(image_paths)} clean images (scored against real GT)")

    vis_dir = os.path.join(vis_root, "baseline_visualizations")
    rows = []
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        _, detections = process_image(model, path, conf_thresh=conf, iou_thresh=iou_thresh,
                                       output_dir=vis_dir, save_visualization=(i < 5))
        ref_items = gt_boxes.get(filename, [])
        rows.extend(per_class_and_overall_rows(
            ref_items, detections, match_iou_thresh, {"image": filename}))
    return image_paths, pd.DataFrame(rows)


def run_degraded_in_memory(model, image_paths, gt_boxes, vis_root, conf, iou_thresh,
                            match_iou_thresh, num_levels):
    """Returns (distorted_df, enhanced_df) -- same schema, one row per
    (image, class, augmentation, level); enhanced_df is the same augmented
    image additionally passed through enhancements.ENHANCEMENTS[aug_name]."""
    fns = make_augmentation_fns(num_levels)

    print(f"[degraded] caching {len(image_paths)} clean images in memory")
    clean_cache = {p: cv2.imread(p) for p in image_paths}

    distorted_rows, enhanced_rows = [], []
    for aug_name, apply_fn in fns.items():
        restore_fn = enhancements.ENHANCEMENTS[aug_name]
        for level_idx in range(num_levels):
            level = level_idx + 1
            vis_dir = os.path.join(vis_root, "degraded_visualizations", aug_name, f"level_{level}")
            enh_vis_dir = os.path.join(vis_root, "enhanced_visualizations", aug_name, f"level_{level}")
            print(f"[degraded] {aug_name} level_{level} ({len(image_paths)} images)")
            for i, path in enumerate(image_paths):
                filename = os.path.basename(path)
                clean_img = clean_cache[path]
                if clean_img is None:
                    continue
                aug_img = apply_fn(clean_img, level_idx)
                snr = compute_snr_db(clean_img, aug_img)
                ref_items = gt_boxes.get(filename, [])

                _, detections = process_image(model, path, conf_thresh=conf, iou_thresh=iou_thresh,
                                               output_dir=vis_dir, save_visualization=(i == 0),
                                               image_override=aug_img)
                distorted_rows.extend(per_class_and_overall_rows(
                    ref_items, detections, match_iou_thresh,
                    {"image": filename, "augmentation": aug_name, "level": level, "snr_db": snr}))

                enh_img = restore_fn(aug_img)
                _, enh_detections = process_image(model, path, conf_thresh=conf, iou_thresh=iou_thresh,
                                                    output_dir=enh_vis_dir, save_visualization=(i == 0),
                                                    image_override=enh_img)
                enhanced_rows.extend(per_class_and_overall_rows(
                    ref_items, enh_detections, match_iou_thresh,
                    {"image": filename, "augmentation": aug_name, "level": level, "snr_db": snr}))
    return pd.DataFrame(distorted_rows), pd.DataFrame(enhanced_rows)


def plot_per_class_metric_on_clean(baseline_df, metric, ylabel, out_path):
    """Per-class bar chart on CLEAN (baseline) images only, sorted descending,
    with the overall (class="all") mean as a dashed reference line -- same
    style as the course brief's own example (slide 16, "Per-class IoU on
    clean images")."""
    per_class = baseline_df[baseline_df["class"] != "all"].groupby("class")[metric].mean().dropna()
    per_class = per_class.sort_values(ascending=False)
    overall = baseline_df[baseline_df["class"] == "all"][metric].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(per_class.index, per_class.values)
    ax.axhline(overall, linestyle="--", color="red", label=f"mean (all classes) = {overall:.3f}")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Per-class {ylabel} on clean images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    """Distorted vs. enhanced vs. clean-baseline bar chart (aggregate,
    class="all"), one bar pair per distortion (mean over all levels/images)
    -- course brief's own example style (slide 31)."""
    dist_all = distorted_df[distorted_df["class"] == "all"]
    enh_all = enhanced_df[enhanced_df["class"] == "all"]
    dist_means = dist_all.groupby("augmentation")[metric].mean()
    enh_means = enh_all.groupby("augmentation")[metric].mean()
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
    parser = argparse.ArgumentParser(description="Vehicle detection robustness sweep vs SNR, per class (real GT)")
    parser.add_argument("--clean_dir", default=str(config.TASK3_CLEAN_DIR))
    parser.add_argument("--out_dir", default=None,
                         help="Legacy: if given, ALL output (CSVs/plots/visualizations) goes into "
                              "this single flat folder (old behavior). If omitted (default), output "
                              "goes to the unified outputs/{csv_results,graph_results,visualizations}/task3/ layout.")
    parser.add_argument("--gt_csv", default=str(config.TASK3_GT_CSV))
    parser.add_argument("--model", default=str(config.YOLO_WEIGHTS_PATH))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold used by the detector")
    parser.add_argument("--match_iou_thresh", type=float, default=0.5, help="IoU threshold for matching a detection to a GT box")
    parser.add_argument("--num_levels", type=int, default=9)
    args = parser.parse_args()

    if args.out_dir is not None:
        csv_dir = graph_dir = vis_dir = Path(args.out_dir)
    else:
        csv_dir, graph_dir, vis_dir = config.TASK3_CSV_DIR, config.TASK3_GRAPH_DIR, config.TASK3_VIS_DIR
    for d in (csv_dir, graph_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)

    gt_csv = Path(args.gt_csv)
    if not gt_csv.exists():
        raise FileNotFoundError(
            f"GT CSV not found: {gt_csv}. Run src_Alon/helper_files/extract_task3_gt.py first."
        )
    gt_boxes = load_gt_boxes(gt_csv)

    image_paths, baseline_df = run_baseline_vs_gt(
        model, args.clean_dir, gt_boxes, str(vis_dir), args.conf, args.iou, args.match_iou_thresh)
    baseline_df.to_csv(csv_dir / "baseline_vs_gt_metrics.csv", index=False)

    baseline_all = baseline_df[baseline_df["class"] == "all"]
    baseline_recall = baseline_all["matched_recall"].mean()
    baseline_iou = baseline_all["mean_iou_matched"].mean()
    print(f"\n[baseline vs GT] recall={baseline_recall:.3f}  mean_iou={baseline_iou:.3f}")

    degraded_df, enhanced_df = run_degraded_in_memory(
        model, image_paths, gt_boxes, str(vis_dir), args.conf, args.iou,
        args.match_iou_thresh, args.num_levels)
    degraded_df.to_csv(csv_dir / "degraded_per_image.csv", index=False)
    enhanced_df.to_csv(csv_dir / "enhanced_per_image.csv", index=False)

    level_summary_all = degraded_df[degraded_df["class"] == "all"].groupby(["augmentation", "level"]).agg(
        matched_recall=("matched_recall", "mean"),
        retention_ratio=("retention_ratio", "mean"),
        mean_iou_matched=("mean_iou_matched", "mean"),
        snr_db=("snr_db", "mean"),
    ).reset_index()
    level_summary_all.to_csv(csv_dir / "level_summary.csv", index=False)

    level_summary_per_class = degraded_df[degraded_df["class"] != "all"].groupby(
        ["augmentation", "level", "class"]).agg(
        matched_recall=("matched_recall", "mean"),
        retention_ratio=("retention_ratio", "mean"),
        mean_iou_matched=("mean_iou_matched", "mean"),
        snr_db=("snr_db", "mean"),
    ).reset_index()
    level_summary_per_class.to_csv(csv_dir / "level_summary_per_class.csv", index=False)

    plot_metric_vs_snr(level_summary_all, baseline_recall, "matched_recall",
                        "Detection recall vs real GT",
                        graph_dir / "recall_vs_snr.png")
    plot_metric_vs_snr(level_summary_all, baseline_iou, "mean_iou_matched",
                        "Mean IoU of matched detections",
                        graph_dir / "iou_vs_snr.png")
    plot_per_class_metric_on_clean(baseline_df, "mean_iou_matched",
                                    "IoU (mean, matched detections only)",
                                    graph_dir / "per_class_iou_clean.png")
    plot_per_class_metric_on_clean(baseline_df, "matched_recall",
                                    "Recall (matched GT boxes / total GT boxes)",
                                    graph_dir / "per_class_recall_clean.png")

    plot_enhancement_comparison(degraded_df, enhanced_df, baseline_recall, "matched_recall",
                                 "Detection recall vs real GT",
                                 graph_dir / "recall_per_distortion.png")
    plot_enhancement_comparison(degraded_df, enhanced_df, baseline_iou, "mean_iou_matched",
                                 "Mean IoU of matched detections",
                                 graph_dir / "iou_per_distortion.png")

    for cls in VEHICLE_CLASSES:
        cls_summary = level_summary_per_class[level_summary_per_class["class"] == cls]
        # A class can show up here with rows that are all-NaN recall/IoU if
        # the detector produced a false-positive of that class somewhere but
        # GT has zero real instances of it (cand_count > 0, ref_count == 0
        # everywhere) -- that's not a meaningful per-class result, skip it
        # the same as if there were no rows at all.
        if cls_summary.empty or not cls_summary["matched_recall"].notna().any():
            print(f"[skip] no GT instances of class '{cls}' in this image set -- no per-class plot")
            continue
        cls_baseline = baseline_df[baseline_df["class"] == cls]
        cls_baseline_recall = cls_baseline["matched_recall"].mean() if not cls_baseline.empty else np.nan
        cls_baseline_iou = cls_baseline["mean_iou_matched"].mean() if not cls_baseline.empty else np.nan
        plot_metric_vs_snr(cls_summary, cls_baseline_recall, "matched_recall",
                            f"Detection recall vs real GT ({cls})",
                            graph_dir / f"recall_vs_snr_{cls}.png")
        plot_metric_vs_snr(cls_summary, cls_baseline_iou, "mean_iou_matched",
                            f"Mean IoU of matched detections ({cls})",
                            graph_dir / f"iou_vs_snr_{cls}.png")

    print("\n=== Level summary (mean per augmentation/level, all classes) ===")
    print(level_summary_all.to_string(index=False))
    print("\n=== Level summary (mean per augmentation/level/class) ===")
    print(level_summary_per_class.to_string(index=False))
    print(f"\nCSVs written to {csv_dir.resolve()}")
    print(f"Plots written to {graph_dir.resolve()}")
    print(f"Visualizations written to {vis_dir.resolve()}")


if __name__ == "__main__":
    main()
