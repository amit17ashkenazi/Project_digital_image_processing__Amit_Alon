import zipfile
import json
import os
import shutil
import tempfile

import numpy as np

# Save your lane-detection code from the previous message as lane_detection_pipeline.py
# in this same folder, so this import works.
from lane_detection_pipeline import process_image, ROI_TOP_RATIO

# =========================
# PATHS
# =========================
zip_path = "data/bdd100k.zip"
output_folder = "matched_images"
json_name = "bdd100k_labels_images_train.json"

MATCH_TOLERANCE_PX = 45   # how close (pixels) your detected line must be to the GT line
MAX_IMAGES = 300
IMG_WIDTH, IMG_HEIGHT = 1280, 720  # standard BDD100K frame size

os.makedirs(output_folder, exist_ok=True)


def fit_line_from_points(xs, ys, weights, height):
    """Same weighted-regression approach as your merge_lines(): fit x as a function of y."""
    poly = np.polyfit(ys, xs, 1, w=weights)
    slope, intercept = poly
    y1 = height
    y2 = int(height * ROI_TOP_RATIO)
    x1 = slope * y1 + intercept
    x2 = slope * y2 + intercept
    return (x1, y1, x2, y2)


def get_gt_lane_lines(labels, img_width, img_height):
    """
    Pull the ground-truth 'ego lane' boundaries out of BDD100K's lane labels:
    the annotated marking closest to center on the left, and the one closest
    to center on the right -- the same role as your left_lane / right_lane.
    """
    mid_x = img_width / 2
    candidates = []  # (side, bottom_x, fitted_line)

    for lab in labels:
        if lab.get("category") != "lane":
            continue

        lane_attr = lab.get("attributes", {})
        direction = lane_attr.get("laneDirection")
        lane_types = lane_attr.get("laneTypes") or lane_attr.get("laneType")

        if direction and direction != "parallel":
            continue
        if lane_types and "crosswalk" in str(lane_types).lower():
            continue

        poly = lab.get("poly2d")
        if not poly:
            continue

        # poly2d can be a list of segment dicts with "vertices", depending
        # on label-format version -- gather every vertex across segments.
        xs, ys = [], []
        for seg in poly:
            vertices = seg.get("vertices", []) if isinstance(seg, dict) else seg
            for p in vertices:
                xs.append(p[0])
                ys.append(p[1])

        if len(xs) < 2:
            continue

        weights = [1.0] * len(xs)
        try:
            line = fit_line_from_points(xs, ys, weights, img_height)
        except Exception:
            continue

        bottom_x = line[0]
        side = "left" if bottom_x < mid_x else "right"
        candidates.append((side, bottom_x, line))

    left_candidates = [c for c in candidates if c[0] == "left"]
    right_candidates = [c for c in candidates if c[0] == "right"]

    left_gt = max(left_candidates, key=lambda c: c[1]) if left_candidates else None
    right_gt = min(right_candidates, key=lambda c: c[1]) if right_candidates else None

    return (left_gt[2] if left_gt else None,
            right_gt[2] if right_gt else None)


def lines_match(detected, ground_truth, tolerance=MATCH_TOLERANCE_PX):
    """Compare two (x1,y1,x2,y2) lines at both y-levels (bottom and ROI-top)."""
    if detected is None or ground_truth is None:
        return False
    dx1, _, dx2, _ = detected
    gx1, _, gx2, _ = ground_truth
    return abs(dx1 - gx1) <= tolerance and abs(dx2 - gx2) <= tolerance


# =========================
# Main pipeline
# =========================
with zipfile.ZipFile(zip_path, "r") as zip_ref:

    json_path_inside_zip = None
    for path in zip_ref.namelist():
        if json_name in path:
            json_path_inside_zip = path
            break

    if json_path_inside_zip is None:
        raise Exception("JSON file not found in ZIP")

    with zip_ref.open(json_path_inside_zip) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} labeled images")

    train_paths = {
        os.path.basename(p): p
        for p in zip_ref.namelist()
        if "/train/" in p
    }

    matched_images = []

    with tempfile.TemporaryDirectory() as tmp_dir:

        for item in data:
            if len(matched_images) >= MAX_IMAGES:
                break

            attr = item.get("attributes", {})
            if not (
                attr.get("scene") == "highway"
                and attr.get("weather") == "clear"
                and attr.get("timeofday") == "daytime"
            ):
                continue

            name = item["name"]
            if name not in train_paths:
                continue

            tmp_path = os.path.join(tmp_dir, name)
            with zip_ref.open(train_paths[name]) as source, open(tmp_path, "wb") as target:
                shutil.copyfileobj(source, target)

            result = process_image(tmp_path, show=False)
            os.remove(tmp_path)

            if result is None:
                continue

            gt_left, gt_right = get_gt_lane_lines(item.get("labels", []), IMG_WIDTH, IMG_HEIGHT)

            left_ok = lines_match(result["left_lane"], gt_left)
            right_ok = lines_match(result["right_lane"], gt_right)

            if left_ok and right_ok:
                matched_images.append(name)
                print(f"MATCH ({len(matched_images)}/{MAX_IMAGES}): {name}")

        print(f"\nTotal matched: {len(matched_images)}")

        for name in matched_images:
            with zip_ref.open(train_paths[name]) as source, \
                 open(os.path.join(output_folder, name), "wb") as target:
                shutil.copyfileobj(source, target)

print(f"Done! {len(matched_images)} images copied into '{output_folder}'")