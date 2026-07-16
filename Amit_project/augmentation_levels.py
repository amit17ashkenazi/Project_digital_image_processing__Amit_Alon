"""
Shared, disk-free helpers for the degradation sweeps: builds the same
9-level motion_blur/low_light/rain ladders used across the project, and
computes achieved SNR (dB) in-memory (no files written).
"""
import numpy as np

from augmentations_AMIT import apply_motion_blur, apply_low_light, apply_rain

AUGMENTATIONS = ["motion_blur", "low_light", "rain"]


def compute_snr_db(clean_image, augmented_image) -> float:
    clean = clean_image.astype(np.float64)
    aug = augmented_image.astype(np.float64)
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - aug) ** 2)
    if noise_power <= 1e-10:
        return 100.0
    return 10.0 * np.log10(signal_power / noise_power)


def build_level_params(num_levels: int):
    blur_kernel_sizes = [int(k) for k in np.linspace(3, 25, num_levels)]
    blur_kernel_sizes = [k + 1 if k % 2 == 0 else k for k in blur_kernel_sizes]
    rain_densities = np.linspace(0.01, 0.45, num_levels)
    lowlight_brightness_factors = np.linspace(0.9, 0.1, num_levels)
    return blur_kernel_sizes, rain_densities, lowlight_brightness_factors


def make_augmentation_fns(num_levels: int):
    blur_kernel_sizes, rain_densities, lowlight_factors = build_level_params(num_levels)

    def motion_blur_apply(image, level_idx):
        return apply_motion_blur(image, blur_kernel_sizes[level_idx])

    def low_light_apply(image, level_idx):
        return apply_low_light(image, lowlight_factors[level_idx], gamma=1.5)

    def rain_apply(image, level_idx):
        return apply_rain(
            image, density=rain_densities[level_idx], length=25, thickness=3,
            angle=-70, blur_ksize=5, seed=42,
        )

    return {
        "motion_blur": motion_blur_apply,
        "low_light": low_light_apply,
        "rain": rain_apply,
    }
