"""
Feature Matching on BDD100K Images
==================================

This script performs:
1. Loads image pairs (either from a dataset directory or two specific image files)
2. Extracts image features (default: ORB, optional: SIFT)
3. Matches feature descriptors (BFMatcher + Lowe's Ratio Test)
4. Filters inlier matches using RANSAC + Homography
5. Saves:
   - A visualization image (PNG) showing the feature matches for each image pair
   - Numerical metrics printed to the console
   - A cumulative CSV file containing the results for all processed image pairs

Usage:
    python feature_matching_bdd100k.py --dataset_dir /path/to/bdd100k/images \
                                       --output_dir ./output \
                                       --num_pairs 10 \
                                       --method orb

Or on two specific images:
    python feature_matching_bdd100k.py --img1 /path/a.jpg --img2 /path/b.jpg --output_dir ./output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os
import re
import cv2
import csv
import glob
import time
import random
import argparse
import numpy as np


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def get_detector(method="orb", n_features=2000):
    method = method.lower()
    if method == "orb":
        return cv2.ORB_create(nfeatures=n_features)
    elif method == "sift":
        return cv2.SIFT_create(nfeatures=n_features)
    elif method == "akaze":
        return cv2.AKAZE_create()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'orb', 'sift', or 'akaze'.")


def get_matcher(method="orb"):
    method = method.lower()
    if method == "orb":
        # ORB descriptors are binary -> Hamming distance
        return cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        # SIFT/AKAZE descriptors are float -> L2 distance
        return cv2.BFMatcher(cv2.NORM_L2)


# --------------------------------------------------------------------------
# Process a single image pair
# --------------------------------------------------------------------------
def process_pair(img1_path, img2_path, method="orb", n_features=2000,
                  ratio_thresh=0.75, ransac_thresh=5.0, output_dir="./output",
                  save_visualization=True, img1_override=None, img2_override=None):
    """img1_path/img2_path are used for naming/metrics either way. If
    img1_override/img2_override (BGR arrays) are given, they are matched
    instead of re-reading from disk -- lets callers feed an in-memory
    distorted image without ever writing it to disk."""

    img1 = img1_override if img1_override is not None else cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = img2_override if img2_override is not None else cv2.imread(img2_path, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        print(f"[WARN] Could not read: {img1_path} or {img2_path}")
        return None

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    detector = get_detector(method, n_features)
    matcher = get_matcher(method)

    t0 = time.time()
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    detect_time = time.time() - t0

    n_kp1, n_kp2 = len(kp1), len(kp2)

    if des1 is None or des2 is None or n_kp1 == 0 or n_kp2 == 0:
        print(f"[WARN] No descriptors found for pair: {img1_path}, {img2_path}")
        return None

    # KNN Matching + Lowe's Ratio Test
    t0 = time.time()
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m_n in knn_matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    match_time = time.time() - t0

    n_good_matches = len(good_matches)

    # RANSAC + Homography -> count inliers
    n_inliers = 0
    inlier_ratio = 0.0
    if n_good_matches >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if mask is not None:
            n_inliers = int(mask.sum())
            inlier_ratio = n_inliers / n_good_matches

    # Metrics
    match_ratio = n_good_matches / min(n_kp1, n_kp2) if min(n_kp1, n_kp2) > 0 else 0.0
    avg_distance = float(np.mean([m.distance for m in good_matches])) if good_matches else 0.0

    metrics = {
        "img1": os.path.basename(img1_path),
        "img2": os.path.basename(img2_path),
        "method": method,
        "n_keypoints_img1": n_kp1,
        "n_keypoints_img2": n_kp2,
        "n_good_matches": n_good_matches,
        "n_inliers": n_inliers,
        "inlier_ratio": round(inlier_ratio, 4),
        "match_ratio": round(match_ratio, 4),
        "avg_match_distance": round(avg_distance, 4),
        "detect_time_sec": round(detect_time, 4),
        "match_time_sec": round(match_time, 4),
    }

    # Visualization -- only draw the good matches (up to 100 for clarity)
    vis_path = None
    if save_visualization:
        draw_matches = sorted(good_matches, key=lambda m: m.distance)[:100]
        vis = cv2.drawMatches(
            img1, kp1, img2, kp2, draw_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        os.makedirs(output_dir, exist_ok=True)
        pair_name = f"{os.path.splitext(metrics['img1'])[0]}__{os.path.splitext(metrics['img2'])[0]}"
        vis_path = os.path.join(output_dir, f"match_{pair_name}.png")
        cv2.imwrite(vis_path, vis)
    metrics["visualization_path"] = vis_path

    return metrics


FRAME_NAME_RE = re.compile(r"^(?P<video_id>.+)-(?P<frame>\d+)$")


def parse_frame_name(filepath):
    """Parses a filename in the format <video_id>-<frame_number>.jpg (as used
    by BDD100K sequences). Example: a91b7555-00001220.jpg -> video_id='a91b7555',
    frame_number=1220"""
    base = os.path.splitext(os.path.basename(filepath))[0]
    m = FRAME_NAME_RE.match(base)
    if not m:
        return None, None
    return m.group("video_id"), int(m.group("frame"))


# --------------------------------------------------------------------------
# Collect image pairs from a dataset directory
# --------------------------------------------------------------------------
def collect_pairs_from_dataset(dataset_dir, num_pairs=10, mode="consecutive",
                                frame_gap=1, seed=42):
    """
    Searches for images (jpg/png) in the directory (including subfolders) and
    builds pairs.

    mode="sequence": (recommended!) Parses each filename into
        (video_id, frame_number) using the format <video_id>-<frame_number>.jpg
        (as in BDD100K: a91b7555-00001220.jpg), groups images by video_id, sorts
        by frame number within each video, and builds pairs of real consecutive
        frames from the same sequence, frame_gap apart (there's significant
        visual overlap between them -- that's what produces meaningful feature
        matching).
    mode="consecutive": Pairs of images consecutive in alphabetical order within
        the folder (naive -- doesn't check they're actually from the same video).
    mode="random": Random pairs from all images found (for "false positive" testing).
    """
    exts = ("*.jpg", "*.jpeg", "*.png")
    all_images = []
    for ext in exts:
        all_images.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
    all_images = sorted(all_images)

    if len(all_images) < 2:
        raise FileNotFoundError(f"Fewer than two images found in directory: {dataset_dir}")

    pairs = []

    if mode == "sequence":
        sequences = {}
        for path in all_images:
            video_id, frame_num = parse_frame_name(path)
            if video_id is None:
                continue
            sequences.setdefault(video_id, []).append((frame_num, path))

        sequences = {vid: sorted(frames) for vid, frames in sequences.items() if len(frames) >= 2}

        if not sequences:
            raise ValueError(
                "No frame sequences matching the format '<video_id>-<frame_number>.jpg' "
                "were found. Check the filenames in the directory, or use "
                "mode='consecutive'/'random'."
            )

        for vid, frames in sequences.items():
            for i in range(len(frames) - frame_gap):
                _, path_a = frames[i]
                _, path_b = frames[i + frame_gap]
                pairs.append((path_a, path_b))
                if len(pairs) >= num_pairs:
                    return pairs

        print(f"[INFO] Found {len(sequences)} distinct video sequences, "
              f"{len(pairs)} consecutive frame pairs total.")

    elif mode == "consecutive":
        for i in range(min(num_pairs, len(all_images) - 1)):
            pairs.append((all_images[i], all_images[i + 1]))
    else:  # random
        random.seed(seed)
        for _ in range(num_pairs):
            a, b = random.sample(all_images, 2)
            pairs.append((a, b))

    return pairs


# --------------------------------------------------------------------------
# Save cumulative CSV
# --------------------------------------------------------------------------
def save_csv(results, csv_path):
    if not results:
        print("[WARN] No results to save.")
        return

    fieldnames = list(results[0].keys())
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[OK] Saved {len(results)} rows to: {csv_path}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Feature Matching on BDD100K images")
    parser.add_argument("--dataset_dir", type=str,
                         default=str(config.TASK2_CLEAN_DIR),
                         help="Path to the BDD100K image directory")
    parser.add_argument("--img1", type=str, default=None, help="Path to first image (single-pair run)")
    parser.add_argument("--img2", type=str, default=None, help="Path to second image (single-pair run)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--csv_name", type=str, default="feature_matching_results.csv")
    parser.add_argument("--method", type=str, default="orb", choices=["orb", "sift", "akaze"])
    parser.add_argument("--n_features", type=int, default=2000)
    parser.add_argument("--ratio_thresh", type=float, default=0.75)
    parser.add_argument("--ransac_thresh", type=float, default=5.0)
    parser.add_argument("--num_pairs", type=int, default=300,
                         help="How many image pairs to process from the dataset (high default "
                              "to cover both long and short sequences)")
    parser.add_argument("--pair_mode", type=str, default="sequence",
                         choices=["sequence", "consecutive", "random"],
                         help="'sequence' (recommended): real consecutive frame pairs from the "
                              "same video, per <video_id>-<frame_number>.jpg. 'consecutive': by "
                              "file order in the folder. 'random': random pairs.")
    parser.add_argument("--frame_gap", type=int, default=1,
                         help="Distance (in frames) between the two images in a pair, when using "
                              "pair_mode='sequence'. E.g. gap=5 between a91b7555-00001220 and "
                              "a91b7555-00001225.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.csv_name)

    # Mode 1: single image pair
    if args.img1 and args.img2:
        pairs = [(args.img1, args.img2)]
    # Mode 2: full dataset directory
    elif args.dataset_dir:
        pairs = collect_pairs_from_dataset(args.dataset_dir, args.num_pairs, args.pair_mode,
                                            frame_gap=args.frame_gap)
    else:
        raise ValueError("You must provide either --dataset_dir or (--img1 and --img2)")

    all_results = []
    for img1_path, img2_path in pairs:
        print(f"\n[INFO] Processing pair: {os.path.basename(img1_path)} <-> {os.path.basename(img2_path)}")
        metrics = process_pair(
            img1_path, img2_path,
            method=args.method,
            n_features=args.n_features,
            ratio_thresh=args.ratio_thresh,
            ransac_thresh=args.ransac_thresh,
            output_dir=args.output_dir,
        )
        if metrics:
            print("  --- Metrics ---")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
            all_results.append(metrics)

    save_csv(all_results, csv_path)


if __name__ == "__main__":
    main()