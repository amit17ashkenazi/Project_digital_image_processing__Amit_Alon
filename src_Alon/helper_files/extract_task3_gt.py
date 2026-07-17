"""
Extracts real BDD100K box2d annotations (v2 detection labels, det_v2_{train,
val}_release.json) for every image in the Task 3 clean-image folder, remaps
categories to the COCO/YOLO vehicle class names the detector already reports,
and writes one row per box to config.TASK3_GT_CSV.

Uses plain json.load() rather than ijson (ijson isn't installed and pip is
blocked by this machine's network/SSL setup) -- the train file is ~352MB but
loads in well under 15s, so streaming wasn't worth the extra complexity here.

Usage:
    python src_Alon/helper_files/extract_task3_gt.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

import csv
import json


def list_image_names(folder: Path) -> set[str]:
    exts = {".jpg", ".jpeg", ".png"}
    return {p.name for p in folder.iterdir() if p.suffix.lower() in exts}


def extract_boxes_for_split(json_path: Path, needed_names: set[str]) -> dict[str, list[dict]]:
    """Returns {image_name: [{"class": ..., "x1":.., "y1":.., "x2":.., "y2":..}, ...]}
    for every needed_names image found in this split, vehicle classes only."""
    print(f"Loading {json_path.name} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} labeled images in this split")

    per_image: dict[str, list[dict]] = {}
    for item in data:
        name = item.get("name")
        if name not in needed_names:
            continue
        boxes = []
        for lab in item.get("labels", []):
            category = lab.get("category")
            coco_class = config.BDD100K_TO_COCO_VEHICLE_CLASS.get(category)
            if coco_class is None:
                continue
            box2d = lab.get("box2d")
            if not box2d:
                continue
            boxes.append({
                "class": coco_class,
                "x1": box2d["x1"], "y1": box2d["y1"],
                "x2": box2d["x2"], "y2": box2d["y2"],
            })
        per_image[name] = boxes
    return per_image


def main():
    needed_names = list_image_names(config.TASK3_CLEAN_DIR)
    print(f"Task 3 clean-image folder: {len(needed_names)} images")

    found: dict[str, list[dict]] = {}
    remaining = set(needed_names)
    for json_path in (config.BDD100K_V2_TRAIN_JSON, config.BDD100K_V2_VAL_JSON):
        if not remaining:
            break
        per_image = extract_boxes_for_split(json_path, remaining)
        found.update(per_image)
        remaining -= per_image.keys()

    missing = needed_names - found.keys()
    if missing:
        print(f"[WARN] {len(missing)}/{len(needed_names)} images have NO matching "
              f"GT annotation entry (not in either det_v2 split). Examples: "
              f"{sorted(missing)[:5]}")
    else:
        print(f"All {len(needed_names)} images matched to a GT entry.")

    rows = []
    n_images_with_boxes = 0
    for image_name, boxes in found.items():
        if boxes:
            n_images_with_boxes += 1
        for b in boxes:
            rows.append({"image": image_name, **b})

    config.TASK3_GT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(config.TASK3_GT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "class", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} vehicle GT boxes across {n_images_with_boxes}/{len(needed_names)} "
          f"images (some images legitimately have zero vehicles).")
    print(f"Saved to {config.TASK3_GT_CSV}")


if __name__ == "__main__":
    main()
