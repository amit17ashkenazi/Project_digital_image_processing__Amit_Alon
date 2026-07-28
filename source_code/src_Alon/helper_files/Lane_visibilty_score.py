import cv2
import numpy as np

def lane_visibility_score(img):
    """
    Returns a score between 0 and 1 for lane detection suitability.
    Higher = better image for lane detection.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # =========================
    # 1. Brightness score
    # =========================
    brightness = np.mean(gray) / 255.0  # normalize to [0,1]

    # Penalize too dark images
    brightness_score = min(brightness / 0.5, 1.0)

    # =========================
    # 2. Edge density score
    # =========================
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    edge_score = min(edge_density / 0.02, 1.0)

    # =========================
    # 3. Contrast score
    # =========================
    contrast = np.std(gray) / 128.0
    contrast_score = min(contrast, 1.0)

    # =========================
    # Final weighted score
    # =========================
    score = (
        0.3 * brightness_score +
        0.4 * edge_score +
        0.3 * contrast_score
    )

    return score