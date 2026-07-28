"""
Core distortion functions (motion blur, low light, rain) shared by the
augmented-dataset generator and any script that needs to apply a single
distortion to one image. Pure functions only -- no I/O, no CLI.

Copied/adapted from the lane-detection augmentations module so the same
distortion definitions are reused consistently across all three tasks
(lane detection, feature matching, vehicle detection).
"""
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# MOTION BLUR
# ---------------------------------------------------------------------------

def apply_motion_blur(image, kernel_size):
    """Applies a single pass of linear (horizontal) motion blur."""
    kernel_size = max(1, int(kernel_size))
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)


# ---------------------------------------------------------------------------
# LOW LIGHT
# ---------------------------------------------------------------------------

def apply_low_light(image, brightness_factor, gamma=1.5):
    """Darkens the V channel (HSV) and applies a gamma curve to mimic
    underexposed / night footage. brightness_factor in (0, 1]; lower = darker."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    darkened = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    normalized = darkened.astype(np.float32) / 255.0
    gamma_corrected = np.power(normalized, gamma) * 255.0
    return gamma_corrected.astype(np.uint8)


# ---------------------------------------------------------------------------
# RAIN
# ---------------------------------------------------------------------------

def apply_rain(image, density=0.02, length=20, thickness=1, angle=-70, blur_ksize=5, seed=None):
    """Overlays synthetic rain streaks (random seeded noise stretched along
    `angle`) and slightly desaturates/dims the image for an overcast look."""
    if seed is not None:
        np.random.seed(seed)

    height, width = image.shape[:2]

    rain_layer = np.zeros((height, width), dtype=np.uint8)
    num_drops = int(density * height * width)
    xs = np.random.randint(0, width, num_drops)
    ys = np.random.randint(0, height, num_drops)
    rain_layer[ys, xs] = 255

    kernel_size = max(length, 3)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    angle_rad = np.deg2rad(angle)
    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * np.sin(angle_rad))
        y = int(center + offset * np.cos(angle_rad))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    if kernel.sum() > 0:
        kernel = kernel / kernel.sum()

    streaks = cv2.filter2D(rain_layer, -1, kernel)
    if thickness > 1:
        dilate_kernel = np.ones((thickness, thickness), np.uint8)
        streaks = cv2.dilate(streaks, dilate_kernel)
    streaks = cv2.GaussianBlur(streaks, (blur_ksize, blur_ksize), 0)

    streaks_bgr = cv2.cvtColor(streaks, cv2.COLOR_GRAY2BGR)
    rainy_image = cv2.addWeighted(image, 1.0, streaks_bgr, 0.85, 0)

    hsv = cv2.cvtColor(rainy_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * 0.8
    hsv[:, :, 2] = hsv[:, :, 2] * 0.9
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
