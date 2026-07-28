import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

# lane_detection_pipeline.py lives in a sibling folder, so add it too
sys.path.insert(0, str(config.PROJECT_ROOT / "src_Alon" / "lane_detection"))

import os
import csv
import numpy as np

from lane_detection_pipeline import process_image

# =========================
# PATHS (now from config)
# =========================
clean_folder = config.TASK1_CLEAN_DIR
augmented_root = config.AUGMENTED_ROOT
snr_log_csv = config.SNR_LOG_CSV
output_csv = config.LANE_COMPARISON_CSV

AUGMENTATIONS = config.AUGMENTATIONS
NUM_LEVELS = config.NUM_LEVELS


def line_diff(line_a, line_b):
    if line_a is None or line_b is None:
        return None, None
    ax1, _, ax2, _ = line_a
    bx1, _, bx2, _ = line_b
    return abs(ax1 - bx1), abs(ax2 - bx2)


# =========================
# Load per-image achieved SNR (filename, augmentation, level) -> snr
# =========================
print("Loading achieved SNR log...")
snr_lookup = {}
with open(snr_log_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["filename"], row["augmentation"], int(row["level"]))
        snr_lookup[key] = float(row["achieved_snr_db"])

# =========================
# Step 1: Run baseline (clean) detection
# =========================
clean_filenames = sorted(
    f for f in os.listdir(clean_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

print(f"Running baseline detection on {len(clean_filenames)} clean images...")

baseline_results = {}
for i, filename in enumerate(clean_filenames, start=1):
    path = os.path.join(clean_folder, filename)
    result = process_image(path, show=False)
    if result is not None:
        baseline_results[filename] = result
    if i % 50 == 0 or i == len(clean_filenames):
        print(f"  Baseline: {i}/{len(clean_filenames)}")

print(f"Baseline complete: {len(baseline_results)} images successfully processed\n")

# =========================
# Step 2: Run detection on every augmented version and compare
# =========================
rows = []

for aug_name in AUGMENTATIONS:
    for level in range(1, NUM_LEVELS + 1):
        folder = os.path.join(augmented_root, aug_name, f"level_{level}")
        print(f"Processing {aug_name}/level_{level}...")

        for filename in clean_filenames:
            if filename not in baseline_results:
                continue

            aug_path = os.path.join(folder, filename)
            if not os.path.exists(aug_path):
                continue

            baseline = baseline_results[filename]
            aug_result = process_image(aug_path, show=False)

            if aug_result is None:
                left_bottom_diff = left_top_diff = None
                right_bottom_diff = right_top_diff = None
                left_lost = baseline["left_lane"] is not None
                right_lost = baseline["right_lane"] is not None
            else:
                left_bottom_diff, left_top_diff = line_diff(
                    baseline["left_lane"], aug_result["left_lane"]
                )
                right_bottom_diff, right_top_diff = line_diff(
                    baseline["right_lane"], aug_result["right_lane"]
                )
                left_lost = baseline["left_lane"] is not None and aug_result["left_lane"] is None
                right_lost = baseline["right_lane"] is not None and aug_result["right_lane"] is None

            achieved_snr = snr_lookup.get((filename, aug_name, level))

            rows.append({
                "filename": filename,
                "augmentation": aug_name,
                "level": level,
                "achieved_snr_db": achieved_snr,
                "left_bottom_diff_px": left_bottom_diff,
                "left_top_diff_px": left_top_diff,
                "right_bottom_diff_px": right_bottom_diff,
                "right_top_diff_px": right_top_diff,
                "left_lost": left_lost,
                "right_lost": right_lost,
            })

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} rows to {output_csv}")