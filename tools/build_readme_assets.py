"""
Fills docs/readme_assets/ with everything the README needs, using ONLY
already-completed experiment output:
- outputs/csv_results/task{1,2,3}/*.csv
- outputs/graph_results/task{1,2,3}/*.png
- outputs/visualizations/task{1,2,3}/{baseline,degraded,enhanced}_visualizations/
- data/clean_images/*
- outputs/graph_results/task3/finetuned_vs_pretrained_*.png
- Amit_project/runs/detect/train-2/*.png (optional fine-tuning training curves)

Does NOT re-run any task/distortion/enhancement pipeline and does NOT call
any detector/matcher for evaluation -- the only "generation" here is (a)
applying the existing, already-tuned augmentation functions to one sample
image to build the 9-level distortion showcase grids (a visualization aid,
not a re-run of any experiment), and (b) arranging already-produced PNGs
into composite grids, or re-plotting directly from already-computed CSV
columns (never recomputing a metric).

Usage:
    python tools/build_readme_assets.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

sys.path.insert(0, str(config.PROJECT_ROOT / "Amit_project"))
from augmentation_levels import make_augmentation_fns

import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ASSETS_ROOT = config.PROJECT_ROOT / "docs" / "readme_assets"
OUTPUTS = config.PROJECT_ROOT / "outputs"
DATA = config.PROJECT_ROOT / "data" / "clean_images"

# Row order for every 3x3 comparison grid, per spec
DISTORTIONS = ["motion_blur", "low_light", "rain"]
# Column levels for the 3x3 comparison grids, per spec (NOT low/med/high --
# literally levels 1, 5, 9)
SEVERITY_LEVELS = [1, 5, 9]
SEVERITY_LABELS = [f"Level {lvl}" for lvl in SEVERITY_LEVELS]
NUM_LEVELS_FULL = 9  # for the 9-image distortion showcase grids

# Fixed representative samples (same ones used throughout this project)
TASK1_TASK3_SAMPLE = "0049e5b8-725e21a0"
TASK2_SAMPLE_PAIR = ("00e9be89-00000100", "00e9be89-00000105")


def reset_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"[WARN] missing source, skipped: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def img_or_placeholder(ax, path: Path, title: str):
    """Draws the image if it exists, else a gray 'missing' placeholder --
    keeps grid layout intact even for known gaps (e.g. task2/low_light at
    levels 8-9, where feature matching found nothing at that severity)."""
    if path.exists():
        ax.imshow(mpimg.imread(path))
    else:
        ax.text(0.5, 0.5, "no result\n(failed at\nthis severity)",
                ha="center", va="center", fontsize=9, color="gray")
        ax.set_facecolor("#eeeeee")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# --------------------------------------------------------------------------
# dataset/
# --------------------------------------------------------------------------
def build_dataset_examples():
    d1 = ASSETS_ROOT / "dataset" / "task1_task3_examples"
    d2 = ASSETS_ROOT / "dataset" / "task2_examples"
    reset_dir(d1)
    reset_dir(d2)

    for name in ["0a0a0b1a-7c39d841", "0af6039a-f3c91319", "0b2f5d0d-8fddc1fc"]:
        safe_copy(DATA / "1_data_lane_detection_low_level" / f"{name}.jpg", d1 / f"{name}.jpg")

    for name in ["00e9be89-00001020", "00e9be89-00001025", "00e9be89-00001030"]:
        safe_copy(DATA / "2_data_feature_matching_high_level" / f"{name}.jpg", d2 / f"{name}.jpg")

    print("dataset/ done")


# --------------------------------------------------------------------------
# baseline/
# --------------------------------------------------------------------------
def build_baseline():
    # Task 1: 3 clean lane-detection results
    d1 = ASSETS_ROOT / "baseline" / "task1"
    reset_dir(d1)
    for name in ["00beeb02-50440dcd", "00d4b6b7-7d0a60bf", TASK1_TASK3_SAMPLE]:
        safe_copy(OUTPUTS / "visualizations/task1/baseline_visualizations" / f"lanes_{name}.png",
                  d1 / f"lanes_{name}.png")

    # Task 2: representative clean match visualization + a baseline-metrics
    # bar chart (direct re-plot of baseline_per_pair.csv columns, not a re-run).
    d2 = ASSETS_ROOT / "baseline" / "task2"
    reset_dir(d2)
    p1, p2 = TASK2_SAMPLE_PAIR
    safe_copy(OUTPUTS / "visualizations/task2/baseline_visualizations" / f"match_{p1}__{p2}.png",
              d2 / f"match_{p1}__{p2}.png")

    # match_ratio (raw) is deliberately excluded here -- match_ratio_vs_baseline
    # is the project's single chosen Task 2 metric (2026-07-17, see TASKS.md).
    baseline2_csv = OUTPUTS / "csv_results/task2/baseline_per_pair.csv"
    if baseline2_csv.exists():
        df = pd.read_csv(baseline2_csv)
        mean_val = df["match_ratio"].mean()  # baseline's own match_ratio == match_ratio_vs_baseline by definition
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(["match_ratio_vs_baseline"], [mean_val], color="#DD8452")
        ax.set_ylabel("value")
        ax.set_title("Task 2 -- baseline metric (clean images)")
        fig.tight_layout()
        fig.savefig(d2 / "baseline_metrics.png", dpi=150)
        plt.close(fig)
    else:
        print(f"[WARN] missing {baseline2_csv}, skipped baseline metrics chart")

    # Task 3: representative clean detection + the (recall-only) per-class-on-clean chart
    d3 = ASSETS_ROOT / "baseline" / "task3"
    reset_dir(d3)
    safe_copy(OUTPUTS / "visualizations/task3/baseline_visualizations" / f"detect_{TASK1_TASK3_SAMPLE}.png",
              d3 / f"detect_{TASK1_TASK3_SAMPLE}.png")
    safe_copy(OUTPUTS / "graph_results/task3/per_class_recall_clean.png", d3 / "per_class_recall_clean.png")

    print("baseline/ done")


# --------------------------------------------------------------------------
# distortions/  (9-image showcase grid per distortion, built from scratch --
# this is a visualization aid, not an experiment: apply the existing,
# already-tuned augmentation functions to one fixed sample image)
# --------------------------------------------------------------------------
def build_distortions():
    sample_path = config.TASK3_CLEAN_DIR / f"{TASK1_TASK3_SAMPLE}.jpg"
    img = cv2.imread(str(sample_path))
    if img is None:
        print(f"[WARN] missing {sample_path}, skipped distortion showcase grids")
        return

    fns = make_augmentation_fns(NUM_LEVELS_FULL)
    for aug_name, apply_fn in fns.items():
        d = ASSETS_ROOT / "distortions" / aug_name
        reset_dir(d)
        fig, axes = plt.subplots(3, 3, figsize=(12, 7.5))
        for level_idx in range(NUM_LEVELS_FULL):
            level = level_idx + 1
            distorted = apply_fn(img, level_idx)
            rgb = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)
            ax = axes[level_idx // 3, level_idx % 3]
            ax.imshow(rgb)
            ax.set_title(f"Level {level}", fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{aug_name} -- levels 1-9 (mild to extreme)")
        fig.tight_layout()
        fig.savefig(d / "grid_levels_1_to_9.png", dpi=120)
        plt.close(fig)
    print("distortions/ done")


# --------------------------------------------------------------------------
# distortion_results/  (3x3 grid: rows=distortion [motion_blur/low_light/rain],
# cols=Level 1/5/9)
# --------------------------------------------------------------------------
def build_comparison_grid(task_vis_dir: Path, stage: str, filename_template: str, out_path: Path, title: str):
    """stage: 'degraded' or 'enhanced' -- picks which visualization subfolder to pull from."""
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    for row, aug in enumerate(DISTORTIONS):
        for col, level in enumerate(SEVERITY_LEVELS):
            path = task_vis_dir / f"{stage}_visualizations" / aug / f"level_{level}" / filename_template
            img_or_placeholder(axes[row, col], path, f"{aug} / {SEVERITY_LABELS[col]}")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def build_distortion_results():
    p1, p2 = TASK2_SAMPLE_PAIR

    # Task 1
    d1 = ASSETS_ROOT / "distortion_results" / "task1"
    reset_dir(d1)
    build_comparison_grid(
        OUTPUTS / "visualizations/task1", "degraded", f"lanes_{TASK1_TASK3_SAMPLE}.png",
        d1 / "grid_by_distortion_and_severity.png", "Task 1 -- lane detection under distortions")
    safe_copy(OUTPUTS / "graph_results/task1/lane_offset_vs_snr.png", d1 / "metric_vs_snr.png")

    # Task 2
    d2 = ASSETS_ROOT / "distortion_results" / "task2"
    reset_dir(d2)
    build_comparison_grid(
        OUTPUTS / "visualizations/task2", "degraded", f"match_{p1}__{p2}.png",
        d2 / "grid_by_distortion_and_severity.png", "Task 2 -- feature matching under distortions")
    safe_copy(OUTPUTS / "graph_results/task2/match_ratio_vs_snr.png", d2 / "match_ratio_vs_snr.png")

    # Task 3
    d3 = ASSETS_ROOT / "distortion_results" / "task3"
    reset_dir(d3)
    build_comparison_grid(
        OUTPUTS / "visualizations/task3", "degraded", f"detect_{TASK1_TASK3_SAMPLE}.png",
        d3 / "grid_by_distortion_and_severity.png", "Task 3 -- vehicle detection under distortions")
    safe_copy(OUTPUTS / "graph_results/task3/recall_vs_snr.png", d3 / "recall_vs_snr.png")

    print("distortion_results/ done")


# --------------------------------------------------------------------------
# enhancements/
# --------------------------------------------------------------------------
def build_before_after(distorted_path: Path, enhanced_path: Path, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    img_or_placeholder(axes[0], distorted_path, "distorted")
    img_or_placeholder(axes[1], enhanced_path, "enhanced")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def build_enhancements():
    rep_level = 5  # representative moderate severity for the single before/after pair
    p1, p2 = TASK2_SAMPLE_PAIR

    # Task 1
    d1 = ASSETS_ROOT / "enhancements" / "task1"
    reset_dir(d1)
    base1 = OUTPUTS / "visualizations/task1"
    for aug in DISTORTIONS:
        build_before_after(
            base1 / "degraded_visualizations" / aug / f"level_{rep_level}" / f"lanes_{TASK1_TASK3_SAMPLE}.png",
            base1 / "enhanced_visualizations" / aug / f"level_{rep_level}" / f"lanes_{TASK1_TASK3_SAMPLE}.png",
            d1 / f"before_after_{aug}.png", f"Task 1 -- {aug}, level {rep_level}: distorted vs. enhanced")
    build_comparison_grid(base1, "enhanced", f"lanes_{TASK1_TASK3_SAMPLE}.png",
                           d1 / "grid_enhanced_by_distortion_and_severity.png",
                           "Task 1 -- enhanced lane detection under distortions")
    safe_copy(OUTPUTS / "graph_results/task1/lane_offset_per_distortion.png", d1 / "metric_per_distortion.png")

    # Task 2
    d2 = ASSETS_ROOT / "enhancements" / "task2"
    reset_dir(d2)
    base2 = OUTPUTS / "visualizations/task2"
    for aug in DISTORTIONS:
        build_before_after(
            base2 / "degraded_visualizations" / aug / f"level_{rep_level}" / f"match_{p1}__{p2}.png",
            base2 / "enhanced_visualizations" / aug / f"level_{rep_level}" / f"match_{p1}__{p2}.png",
            d2 / f"before_after_{aug}.png", f"Task 2 -- {aug}, level {rep_level}: distorted vs. enhanced")
    build_comparison_grid(base2, "enhanced", f"match_{p1}__{p2}.png",
                           d2 / "grid_enhanced_by_distortion_and_severity.png",
                           "Task 2 -- enhanced feature matching under distortions")
    safe_copy(OUTPUTS / "graph_results/task2/match_ratio_per_distortion.png", d2 / "match_ratio_per_distortion.png")

    # Task 3
    d3 = ASSETS_ROOT / "enhancements" / "task3"
    reset_dir(d3)
    base3 = OUTPUTS / "visualizations/task3"
    for aug in DISTORTIONS:
        build_before_after(
            base3 / "degraded_visualizations" / aug / f"level_{rep_level}" / f"detect_{TASK1_TASK3_SAMPLE}.png",
            base3 / "enhanced_visualizations" / aug / f"level_{rep_level}" / f"detect_{TASK1_TASK3_SAMPLE}.png",
            d3 / f"before_after_{aug}.png", f"Task 3 -- {aug}, level {rep_level}: distorted vs. enhanced")
    build_comparison_grid(base3, "enhanced", f"detect_{TASK1_TASK3_SAMPLE}.png",
                           d3 / "grid_enhanced_by_distortion_and_severity.png",
                           "Task 3 -- enhanced vehicle detection under distortions")
    safe_copy(OUTPUTS / "graph_results/task3/recall_per_distortion.png", d3 / "recall_per_distortion.png")

    print("enhancements/ done")


# --------------------------------------------------------------------------
# finetuning/
# --------------------------------------------------------------------------
def build_finetuning():
    d = ASSETS_ROOT / "finetuning"
    reset_dir(d)
    safe_copy(OUTPUTS / "graph_results/task3/finetuned_vs_pretrained_recall.png",
              d / "finetuned_vs_pretrained_recall.png")

    # Optional training visualizations, per spec ("unless training
    # visualizations exist") -- these do exist (ultralytics writes them
    # automatically during training), so include the most informative ones.
    train_dir = config.PROJECT_ROOT / "Amit_project" / "runs" / "detect" / "train-2"
    safe_copy(train_dir / "results.png", d / "training_curves.png")
    safe_copy(train_dir / "confusion_matrix.png", d / "training_confusion_matrix.png")

    print("finetuning/ done")


def main():
    ASSETS_ROOT.mkdir(parents=True, exist_ok=True)
    build_dataset_examples()
    build_baseline()
    build_distortions()
    build_distortion_results()
    build_enhancements()
    build_finetuning()
    print(f"\nAll README assets written under {ASSETS_ROOT.resolve()}")


if __name__ == "__main__":
    main()
