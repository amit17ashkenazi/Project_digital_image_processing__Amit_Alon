"""
Feature Matching על תמונות מ-BDD100K
=====================================

מה הסקריפט עושה:
1. טוען זוגות תמונות (מתוך תיקיית ה-dataset או שני קבצים ספציפיים)
2. מחלץ Features (ברירת מחדל: ORB, אופציה: SIFT)
3. מבצע Matching בין ה-Descriptors (BFMatcher + Ratio Test של Lowe)
4. מסנן Inliers באמצעות RANSAC + Homography
5. שומר:
   - תמונת ויזואליזציה (PNG) של ההתאמות עבור כל זוג
   - מטריקות מספריות מודפסות למסך
   - קובץ CSV מצטבר עם התוצאות לכל הזוגות שעובדו

איך מריצים:
    python feature_matching_bdd100k.py --dataset_dir /path/to/bdd100k/images \
                                        --output_dir ./output \
                                        --num_pairs 10 \
                                        --method orb

או על שני קבצים ספציפיים:
    python feature_matching_bdd100k.py --img1 /path/a.jpg --img2 /path/b.jpg --output_dir ./output
"""

import os
import cv2
import csv
import glob
import time
import random
import argparse
import numpy as np


# --------------------------------------------------------------------------
# חילוץ Features
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
# עיבוד זוג תמונות
# --------------------------------------------------------------------------
def process_pair(img1_path, img2_path, method="orb", n_features=2000,
                  ratio_thresh=0.75, ransac_thresh=5.0, output_dir="./output"):

    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

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

    # RANSAC + Homography -> ספירת Inliers
    n_inliers = 0
    inlier_ratio = 0.0
    if n_good_matches >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if mask is not None:
            n_inliers = int(mask.sum())
            inlier_ratio = n_inliers / n_good_matches

    # מטריקות
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

    # ויזואליזציה - נציג רק את ה-Good Matches (או עד 100 מהם לצורך בהירות)
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


# --------------------------------------------------------------------------
# איסוף זוגות תמונות מתוך תיקיית ה-Dataset
# --------------------------------------------------------------------------
def collect_pairs_from_dataset(dataset_dir, num_pairs=10, mode="consecutive", seed=42):
    """
    מחפש תמונות (jpg/png) בתיקייה (כולל תת-תיקיות) ובונה זוגות.
    mode="consecutive": זוגות של תמונות עוקבות (שימושי לרצפים/וידאו-פריימים ב-BDD100K)
    mode="random": זוגות אקראיים מתוך כל התמונות שנמצאו
    """
    exts = ("*.jpg", "*.jpeg", "*.png")
    all_images = []
    for ext in exts:
        all_images.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
    all_images = sorted(all_images)

    if len(all_images) < 2:
        raise FileNotFoundError(f"נמצאו פחות משתי תמונות בתיקייה: {dataset_dir}")

    pairs = []
    if mode == "consecutive":
        for i in range(min(num_pairs, len(all_images) - 1)):
            pairs.append((all_images[i], all_images[i + 1]))
    else:
        random.seed(seed)
        for _ in range(num_pairs):
            a, b = random.sample(all_images, 2)
            pairs.append((a, b))

    return pairs


# --------------------------------------------------------------------------
# שמירת CSV מצטבר
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

    print(f"[OK] נשמרו {len(results)} שורות לקובץ: {csv_path}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Feature Matching על תמונות מ-BDD100K")
    parser.add_argument("--dataset_dir", type=str, default=None,
                         help="נתיב לתיקיית התמונות של BDD100K")
    parser.add_argument("--img1", type=str, default=None, help="נתיב לתמונה ראשונה (ריצה על זוג בודד)")
    parser.add_argument("--img2", type=str, default=None, help="נתיב לתמונה שנייה (ריצה על זוג בודד)")
    parser.add_argument("--output_dir", type=str, default="./output", help="תיקיית פלט")
    parser.add_argument("--csv_name", type=str, default="feature_matching_results.csv")
    parser.add_argument("--method", type=str, default="orb", choices=["orb", "sift", "akaze"])
    parser.add_argument("--n_features", type=int, default=2000)
    parser.add_argument("--ratio_thresh", type=float, default=0.75)
    parser.add_argument("--ransac_thresh", type=float, default=5.0)
    parser.add_argument("--num_pairs", type=int, default=10, help="כמה זוגות תמונות לעבד מתוך ה-dataset")
    parser.add_argument("--pair_mode", type=str, default="consecutive", choices=["consecutive", "random"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, args.csv_name)

    # מצב 1: זוג תמונות בודד
    if args.img1 and args.img2:
        pairs = [(args.img1, args.img2)]
    # מצב 2: תיקיית dataset שלמה
    elif args.dataset_dir:
        pairs = collect_pairs_from_dataset(args.dataset_dir, args.num_pairs, args.pair_mode)
    else:
        raise ValueError("יש לספק either --dataset_dir או (--img1 וגם --img2)")

    all_results = []
    for img1_path, img2_path in pairs:
        print(f"\n[INFO] מעבד זוג: {os.path.basename(img1_path)} <-> {os.path.basename(img2_path)}")
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
