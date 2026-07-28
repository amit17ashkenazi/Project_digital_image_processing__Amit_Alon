"""
Part 4 -- builds the YOLO fine-tuning dataset (Task 3).

Adjusted from the original TASKS.md plan: uses REAL BDD100K GT
(config.TASK3_GT_CSV, from src_Alon/helper_files/extract_task3_gt.py)
instead of clean-image pseudo-labels, since we have it (Phase 1) and it's
strictly more reliable than the pretrained detector's own (imperfect,
~0.23 recall) clean-image output.

For each image: assigns one of the 3 distortions at a fixed moderate
severity level (round-robin, so the training set has roughly equal
representation of all 3 -- deterministic, not random, for reproducibility),
applies it, and writes the distorted image + its (unchanged, since
distortion doesn't move box locations) YOLO-format label to
data/finetune_dataset/{images,labels}/{train,val}/. A fixed 80/20 train/val
split (seeded shuffle).

Usage:
    python build_finetune_dataset.py
    python build_finetune_dataset.py --level 5 --val_ratio 0.2
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import argparse
import random

import cv2
import pandas as pd

from augmentation_levels import make_augmentation_fns

CLASS_NAMES = ["car", "truck", "bus", "motorcycle", "bicycle"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def box2d_to_yolo_line(class_id: int, x1, y1, x2, y2, img_w: int, img_h: int) -> str:
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    parser = argparse.ArgumentParser(description="Build the Task 3 fine-tuning dataset (distorted images + real-GT labels)")
    parser.add_argument("--clean_dir", default=str(config.TASK3_CLEAN_DIR))
    parser.add_argument("--gt_csv", default=str(config.TASK3_GT_CSV))
    parser.add_argument("--out_root", default=str(config.PROJECT_ROOT / "data" / "finetune_dataset"))
    parser.add_argument("--level", type=int, default=5, help="Fixed moderate severity level (1-9) applied to every training image")
    parser.add_argument("--num_levels", type=int, default=9, help="Must match the level ladders used everywhere else")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gt_csv = Path(args.gt_csv)
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT CSV not found: {gt_csv}. Run src_Alon/helper_files/extract_task3_gt.py first.")
    gt_df = pd.read_csv(gt_csv)
    gt_by_image = {img: g for img, g in gt_df.groupby("image")}

    clean_dir = Path(args.clean_dir)
    image_paths = sorted(p for p in clean_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"{len(image_paths)} clean images found")

    random.seed(args.seed)
    shuffled = image_paths[:]
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * args.val_ratio)
    val_set = set(p.name for p in shuffled[:n_val])

    fns = make_augmentation_fns(args.num_levels)
    aug_names = list(fns.keys())
    level_idx = args.level - 1

    out_root = Path(args.out_root)
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_written = {"train": 0, "val": 0}
    n_boxes_dropped_bad_class = 0
    for i, path in enumerate(image_paths):
        split = "val" if path.name in val_set else "train"
        aug_name = aug_names[i % len(aug_names)]
        apply_fn = fns[aug_name]

        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARN] could not read {path}, skipping")
            continue
        h, w = img.shape[:2]
        distorted = apply_fn(img, level_idx)

        img_out = out_root / "images" / split / path.name
        cv2.imwrite(str(img_out), distorted)

        lines = []
        for _, row in gt_by_image.get(path.name, pd.DataFrame()).iterrows():
            if row["class"] not in CLASS_TO_ID:
                n_boxes_dropped_bad_class += 1
                continue
            class_id = CLASS_TO_ID[row["class"]]
            lines.append(box2d_to_yolo_line(class_id, row["x1"], row["y1"], row["x2"], row["y2"], w, h))

        label_out = out_root / "labels" / split / (path.stem + ".txt")
        label_out.write_text("\n".join(lines), encoding="utf-8")
        n_written[split] += 1

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "path: " + str(out_root.resolve()).replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n",
        encoding="utf-8",
    )

    print(f"train: {n_written['train']} images, val: {n_written['val']} images")
    print(f"Each image distorted at a fixed level {args.level}/{args.num_levels}, "
          f"round-robin across {aug_names}")
    if n_boxes_dropped_bad_class:
        print(f"[WARN] dropped {n_boxes_dropped_bad_class} GT boxes with an unrecognized class")
    print(f"data.yaml written to {data_yaml.resolve()}")


if __name__ == "__main__":
    main()
