import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Shared between apply_roi() and merge_lines() so the ROI shape and the
# lane-line projection always agree on where the "top of the road" is.
ROI_TOP_RATIO = 0.6


def preprocess_image(image, blur_ksize=(5, 5)):
    """
    Performs the first two preprocessing steps:
    1. Convert to grayscale.
    2. Apply Gaussian blur.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    return gray, blurred


def detect_edges(image, low_threshold=50, high_threshold=150):
    """
    Applies the Canny edge detector.
    """
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


def apply_roi(image):
    """
    Applies a wider trapezoidal Region of Interest mask.
    Wide enough at the top and bottom to keep both lane lines even on
    curves or wider-angle shots, while still excluding sky/horizon.
    """
    height, width = image.shape


    
    polygon = np.array([[
        (int(0.0 * width), height),                       # Bottom left
        (int(0.30 * width), int(ROI_TOP_RATIO * height)),   # Top left
        (int(0.70 * width), int(ROI_TOP_RATIO * height)),   # Top right
        (int(1.0 * width), height)                        # Bottom right
    ]], dtype=np.int32)
   
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def apply_morphology(edge_image, kernel_size=5):
    """
    Morphological closing to connect broken lane edges.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel)
    return cleaned


def detect_lines(edge_image):
    """
    Probabilistic Hough Transform for line detection.
    """
    lines = cv2.HoughLinesP(
        edge_image,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=25,
        maxLineGap=60
    )
    return lines


def filter_lines(lines, image_width):
    """
    Filters out lines that are too horizontal or vertical based on slope,
    and also rejects lines that fall on the wrong side of image center
    (a left-leaning line sitting in the right half of the image is almost
    always noise, and vice versa).
    """
    left_lines = []
    right_lines = []

    if lines is None:
        return [], []

    mid_x = image_width / 2

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # Tightened slope range: excludes near-horizontal and near-vertical noise
        if not (0.50 <= abs(slope) <= 1.8):
            continue

        avg_x = (x1 + x2) / 2

        if slope < 0:
            # Left lane markings should sit left of (or near) center
            if avg_x > mid_x * 1.1:
                continue
            left_lines.append((x1, y1, x2, y2))
        else:
            # Right lane markings should sit right of (or near) center
            if avg_x < mid_x * 0.9:
                continue
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines


def reject_outliers(lines):
    """
    Removes segments whose slope deviates too far from the median slope
    of the group. This stops a handful of noisy Hough segments from
    dragging the final weighted-regression line off course.
    """
    if len(lines) < 3:
        return lines

    slopes = []
    for x1, y1, x2, y2 in lines:
        slopes.append((y2 - y1) / (x2 - x1))

    median_slope = np.median(slopes)
    deviations = [abs(s - median_slope) for s in slopes]
    mad = np.median(deviations) if np.median(deviations) > 1e-3 else 1e-3

    filtered = []
    for line, s, d in zip(lines, slopes, deviations):
        # Robust z-score style check (median absolute deviation)
        if d / mad <= 3.5:
            filtered.append(line)

    # Never return an empty list if everything got filtered (fallback to original)
    return filtered if filtered else lines


def merge_lines(lines, image_height):
    """
    Merge multiple line segments into a single line using weighted linear
    regression (fit x as a function of y, since lane lines are closer to
    vertical). Longer segments are trusted more and get a higher weight,
    and outlier segments are dropped before fitting.
    The projection upper limit matches the top of the ROI to prevent crossing.
    """
    if len(lines) == 0:
        return None

    lines = reject_outliers(lines)

    x_coords = []
    y_coords = []
    weights = []

    for x1, y1, x2, y2 in lines:
        length = np.hypot(x2 - x1, y2 - y1)
        x_coords += [x1, x2]
        y_coords += [y1, y2]
        weights += [length, length]

    poly = np.polyfit(y_coords, x_coords, 1, w=weights)
    slope, intercept = poly

    y1 = image_height
    y2 = int(image_height * ROI_TOP_RATIO)  # Matches the top of the ROI

    x1 = int(slope * y1 + intercept)
    x2 = int(slope * y2 + intercept)

    return (x1, y1, x2, y2)


def display_results(original, gray, blurred, edges, roi, morph, lane_img=None):
    images = []
    titles = []

    images.append(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    titles.append("Original")

    images.append(gray)
    titles.append("Grayscale")

    images.append(blurred)
    titles.append("Blurred")

    images.append(edges)
    titles.append("Canny Edges")

    images.append(roi)
    titles.append("ROI")

    images.append(morph)
    titles.append("Morphology")

    if lane_img is not None:
        images.append(cv2.cvtColor(lane_img, cv2.COLOR_BGR2RGB))
        titles.append("Lanes Overlaid on Original")

    plt.figure(figsize=(20, 5))

    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)

        if len(images[i].shape) == 3:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap="gray")

        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def process_image(image_path, show=True):
    """
    Runs the full lane-detection pipeline on a single image.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    show : bool
        If True, displays the step-by-step matplotlib figure for this image.

    Returns
    -------
    dict with keys:
        'image_path', 'overlay_img', 'left_lane', 'right_lane',
        'left_count', 'right_count'
        or None if the image could not be read.
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: image not found -> {image_path}")
        return None

    height, width = image.shape[:2]

    # Step 1–2
    gray, blurred = preprocess_image(image, blur_ksize=(5, 5))

    # Step 3 (higher thresholds = fewer, more confident edges - less noise)
    edges = detect_edges(blurred, 60, 180)

    # Step 4 (ROI applied right after edge detection, before morphology,
    # so morphological closing never operates on out-of-lane clutter)
    roi = apply_roi(edges)

    # Step 6 (morphology, now applied on the already-masked edges; a bigger
    # kernel closes larger gaps in broken/dashed lane markings)
    morph = apply_morphology(roi, kernel_size=7)

    # Step 5 (Hough)
    lines = detect_lines(morph)

    # Step 6 (filter) - now also checks left/right position, not just slope
    left_lines, right_lines = filter_lines(lines, width)

    # Merge (weighted regression + outlier rejection happens inside)
    left_lane = merge_lines(left_lines, height)
    right_lane = merge_lines(right_lines, height)

    # Sanity check: left lane must sit left of right lane at the bottom of
    # the image. If not, the detection is unreliable for this frame and we
    # drop it rather than draw a crossed/garbage line.
    if left_lane is not None and right_lane is not None:
        left_bottom_x = left_lane[0]
        right_bottom_x = right_lane[0]
        if left_bottom_x >= right_bottom_x:
            left_lane = None
            right_lane = None

    # Draw final lanes
    lane_img = np.zeros_like(image)

    if left_lane is not None:
        cv2.line(lane_img, (left_lane[0], left_lane[1]),
                 (left_lane[2], left_lane[3]), (0, 255, 0), 5)

    if right_lane is not None:
        cv2.line(lane_img, (right_lane[0], right_lane[1]),
                 (right_lane[2], right_lane[3]), (0, 255, 0), 5)

    # Overlay the green lane lines directly on top of the original image
    overlay_img = cv2.addWeighted(image, 1.0, lane_img, 1.0, 0)

    print(f"[{os.path.basename(image_path)}] "
          f"Left lines: {len(left_lines)}, Right lines: {len(right_lines)}")

    if show:
        display_results(
            image,
            gray,
            blurred,
            edges,
            roi,
            morph,
            overlay_img
        )

    return {
        "image_path": image_path,
        "overlay_img": overlay_img,
        "left_lane": left_lane,
        "right_lane": right_lane,
        "left_count": len(left_lines),
        "right_count": len(right_lines),
    }


def main():
    image_path = r"C:\Users\alonk\GitHub\Project_digital_image_processing__Amit_Alon\data\2_data_feature_matching_high_level\00e9be89-00001050.jpg"
    process_image(image_path, show=True)


if __name__ == "__main__":
    main()