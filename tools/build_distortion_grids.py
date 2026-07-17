"""
Builds a 3x3 grid image per distortion type (levels 1-9, applied to one
fixed representative clean image), for the README's Distortion section.
Raw distorted frames only -- no detection/method overlay, since this is
meant to just show what each distortion/severity looks like.

Usage:
    python tools/build_distortion_grids.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

sys.path.insert(0, str(config.PROJECT_ROOT / "Amit_project"))
from augmentation_levels import make_augmentation_fns

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLE_IMAGE = config.TASK3_CLEAN_DIR / "0049e5b8-725e21a0.jpg"
OUT_DIR = config.PROJECT_ROOT / "docs" / "readme_assets"
NUM_LEVELS = 9


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(SAMPLE_IMAGE))
    if img is None:
        raise FileNotFoundError(SAMPLE_IMAGE)

    fns = make_augmentation_fns(NUM_LEVELS)
    for aug_name, apply_fn in fns.items():
        fig, axes = plt.subplots(3, 3, figsize=(12, 7.5))
        for level_idx in range(NUM_LEVELS):
            level = level_idx + 1
            distorted = apply_fn(img, level_idx)
            rgb = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)
            ax = axes[level_idx // 3, level_idx % 3]
            ax.imshow(rgb)
            ax.set_title(f"level {level}", fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{aug_name} -- levels 1-9 (mild to extreme)")
        fig.tight_layout()
        out_path = OUT_DIR / f"distortion_grid_{aug_name}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
