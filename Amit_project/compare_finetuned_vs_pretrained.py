"""
Part 4 -- final comparison: pretrained+distorted vs. pretrained+enhanced vs.
fine-tuned+distorted, recall and IoU vs SNR (aggregate, class="all").

Usage:
    python compare_finetuned_vs_pretrained.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all_class_summary(csv_dir: Path, class_col_file="degraded_per_image.csv"):
    df = pd.read_csv(csv_dir / class_col_file)
    df_all = df[df["class"] == "all"]
    return df_all.groupby(["augmentation", "level"]).agg(
        matched_recall=("matched_recall", "mean"),
        mean_iou_matched=("mean_iou_matched", "mean"),
        snr_db=("snr_db", "mean"),
    ).reset_index()


def plot_three_way(pretrained_summary, enhanced_summary, finetuned_summary, metric, ylabel, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    aug_names = sorted(pretrained_summary["augmentation"].unique())
    for ax, aug in zip(axes, aug_names):
        for label, summary, style in (
            ("pretrained + distorted", pretrained_summary, "o-"),
            ("pretrained + enhanced", enhanced_summary, "s--"),
            ("fine-tuned + distorted", finetuned_summary, "^-"),
        ):
            group = summary[summary["augmentation"] == aug].sort_values("snr_db")
            if group.empty:
                continue
            ax.plot(group["snr_db"], group[metric], style, label=label)
        ax.invert_xaxis()
        ax.set_title(aug)
        ax.set_xlabel("SNR (dB)")
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    fig.suptitle(f"{ylabel} -- pretrained vs. fine-tuned (Part 4)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare pretrained vs. fine-tuned YOLOv8n (Task 3, Part 4)")
    parser.add_argument("--pretrained_csv_dir", default=str(config.TASK3_CSV_DIR))
    parser.add_argument("--finetuned_csv_dir", default=str(config.PROJECT_ROOT / "outputs_finetuned" / "task3"))
    parser.add_argument("--out_dir", default=str(config.TASK3_GRAPH_DIR))
    args = parser.parse_args()

    pretrained_csv_dir = Path(args.pretrained_csv_dir)
    finetuned_csv_dir = Path(args.finetuned_csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrained_distorted = load_all_class_summary(pretrained_csv_dir, "degraded_per_image.csv")
    pretrained_enhanced = load_all_class_summary(pretrained_csv_dir, "enhanced_per_image.csv")
    finetuned_distorted = load_all_class_summary(finetuned_csv_dir, "degraded_per_image.csv")

    plot_three_way(pretrained_distorted, pretrained_enhanced, finetuned_distorted,
                    "matched_recall", "Detection recall vs real GT",
                    out_dir / "finetuned_vs_pretrained_recall.png")
    plot_three_way(pretrained_distorted, pretrained_enhanced, finetuned_distorted,
                    "mean_iou_matched", "Mean IoU of matched detections",
                    out_dir / "finetuned_vs_pretrained_iou.png")

    print("=== Mean recall across all levels ===")
    print("pretrained + distorted:", pretrained_distorted.groupby("augmentation")["matched_recall"].mean().to_dict())
    print("pretrained + enhanced :", pretrained_enhanced.groupby("augmentation")["matched_recall"].mean().to_dict())
    print("fine-tuned + distorted:", finetuned_distorted.groupby("augmentation")["matched_recall"].mean().to_dict())
    print(f"\nPlots written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
