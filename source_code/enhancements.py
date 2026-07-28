"""
Part 3 -- pre-processing enhancements for each distortion type. Each function
takes a BGR image (same convention as augmentations.py / cv2.imread) and
returns a restored BGR image. Fixed hyperparameters (level-independent), same
as a real deployed enhancer would use without knowing the severity in advance.
"""
import cv2
import numpy as np


def restore_motion_blur(img_bgr):
    """Unsharp-mask deblurring: re-boosts high-frequency edges the blur kernel suppressed.

    Softened weights (2026-07-16, TASKS.md Phase 2 follow-up): the original
    1.5/-0.5 weighting boosted ORB keypoint counts so much (mostly spurious
    high-frequency noise, not genuine structure) that match_ratio -- which
    normalizes by keypoint count -- got WORSE after "enhancement" despite the
    absolute number of good matches improving. 1.2/-0.2 recovers most of the
    same edge detail with much less keypoint inflation."""
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img_bgr, 1.2, blurred, -0.2, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def restore_low_light(img_bgr):
    """Gamma lift + CLAHE on the L channel -- standard low-light enhancement recipe.

    Softened CLAHE (2026-07-16, TASKS.md Phase 2 follow-up): clipLimit=4.0 on
    8x8 tiles over-amplified local contrast, again inflating ORB keypoint
    counts (1432->1986 mean, measured on Task 2's full run) far more than it
    helped real matches -- same match_ratio side effect as motion_blur above.
    clipLimit=2.0 on larger 16x16 tiles keeps most of the visibility recovery
    with a gentler local-contrast boost."""
    gamma = 0.45
    lut = ((np.arange(256) / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
    lifted = cv2.LUT(img_bgr, lut)
    lab = cv2.cvtColor(lifted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def restore_rain(img_bgr):
    """Edge-preserving bilateral filter to smooth out thin bright rain streaks."""
    return cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)


ENHANCEMENTS = {
    "motion_blur": restore_motion_blur,
    "low_light": restore_low_light,
    "rain": restore_rain,
}
