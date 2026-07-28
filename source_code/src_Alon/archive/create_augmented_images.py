import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

import os
import csv
import cv2
import numpy as np

from augmentations import apply_motion_blur, apply_low_light, apply_rain

# =========================
# PATHS (now from config)
# =========================
input_folder = config.TASK1_CLEAN_DIR
output_root = config.AUGMENTED_ROOT
snr_log_csv = config.SNR_LOG_CSV

NUM_LEVELS = config.NUM_LEVELS


# =========================
# SNR calculation (exact MSE-based formula)
# =========================
def compute_snr_db(clean_image, augmented_image):
    clean_image = clean_image.astype(np.float64)
    augmented_image = augmented_image.astype(np.float64)

    signal_power = np.mean(clean_image ** 2)
    noise_power = np.mean((clean_image - augmented_image) ** 2)

    if noise_power <= 1e-10:
        return 100.0  # effectively noiseless

    return 10.0 * np.log10(signal_power / noise_power)


# =========================
# Parameter sweeps -- fixed ladders, no per-image tuning
# =========================
BLUR_KERNEL_SIZES = [3, 5, 7, 9, 11, 13, 15, 17, 19]
RAIN_DENSITIES = np.linspace(0.01, 0.45, NUM_LEVELS)
LOWLIGHT_BRIGHTNESS_FACTORS = np.linspace(0.9, 0.1, NUM_LEVELS)


def motion_blur_apply(image, level_idx):
    kernel_size = BLUR_KERNEL_SIZES[level_idx]
    return apply_motion_blur(image, kernel_size)

def low_light_apply(image, level_idx):
    brightness_factor = LOWLIGHT_BRIGHTNESS_FACTORS[level_idx]
    return apply_low_light(image, brightness_factor, gamma=1.5)


def rain_apply(image, level_idx):
    density = RAIN_DENSITIES[level_idx]
    return apply_rain(
        image, density=density, length=25, thickness=3,
        angle=-70, blur_ksize=5, seed=42
    )


AUGMENTATIONS = {
    "motion_blur": motion_blur_apply,
    "low_light": low_light_apply,
    "rain": rain_apply,
}

# =========================
# Folder setup: one subfolder per intensity level (1-9)
# =========================
for aug_name in AUGMENTATIONS:
    for level_idx in range(NUM_LEVELS):
        folder_name = f"level_{level_idx + 1}"
        os.makedirs(os.path.join(output_root, aug_name, folder_name), exist_ok=True)

# =========================
# Process every image
# =========================
filenames = sorted(
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

print(f"Found {len(filenames)} images to augment")

log_rows = []

for i, filename in enumerate(filenames, start=1):
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"  Skipping unreadable image: {filename}")
        continue

    for aug_name, apply_fn in AUGMENTATIONS.items():
        for level_idx in range(NUM_LEVELS):
            aug_img = apply_fn(image, level_idx)
            achieved_snr = compute_snr_db(image, aug_img)

            folder_name = f"level_{level_idx + 1}"
            out_path = os.path.join(output_root, aug_name, folder_name, filename)
            cv2.imwrite(out_path, aug_img)

            log_rows.append({
                "filename": filename,
                "augmentation": aug_name,
                "level": level_idx + 1,
                "achieved_snr_db": round(achieved_snr, 2),
            })

    if i % 25 == 0 or i == len(filenames):
        print(f"  Processed {i}/{len(filenames)}")

# =========================
# Save SNR log
# =========================
with open(snr_log_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
    writer.writeheader()
    writer.writerows(log_rows)

print(f"\nDone! Augmented images saved under '{output_root}'")
print(f"SNR log saved to '{snr_log_csv}'")

# =========================
# Summary: average measured SNR per level, per augmentation
# =========================
print("\nAverage measured SNR per level (sanity check):")
print(f"{'Augmentation':<15}{'Level':<8}{'Avg SNR (dB)':<15}")
for aug_name in AUGMENTATIONS:
    for level_idx in range(NUM_LEVELS):
        level = level_idx + 1
        matches = [r["achieved_snr_db"] for r in log_rows
                   if r["augmentation"] == aug_name and r["level"] == level]
        avg = np.mean(matches) if matches else float("nan")
        print(f"{aug_name:<15}{level:<8}{avg:<15.2f}")