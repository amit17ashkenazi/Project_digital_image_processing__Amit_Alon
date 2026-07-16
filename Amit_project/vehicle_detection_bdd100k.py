"""
Vehicle Detection on BDD100K Images using YOLOv8
================================================

This script performs:
1. Loads images (either from a dataset directory or specific image files)
2. Runs a YOLOv8 model (pretrained on the COCO dataset) for vehicle detection
3. Filters only vehicle-related classes:
   car, truck, bus, motorcycle, bicycle
4. Saves:
   - A visualization image with bounding boxes, class labels, and confidence scores
   - Numerical metrics (number of detected vehicles, class distribution, average confidence, and inference time)
   - A cumulative CSV file containing all detections (bounding box, class, and confidence) for each processed image

Installation (one-time):
    pip install ultralytics

Usage:
    python vehicle_detection_bdd100k.py --dataset_dir /path/to/bdd100k/images \
                                        --output_dir ./output \
                                        --num_images 20 \
                                        --model yolov8n.pt \
                                        --conf 0.25

Or on specific images:
    python vehicle_detection_bdd100k.py --images img1.jpg img2.jpg --output_dir ./output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import os
import cv2
import csv
import glob
import time
import argparse
import numpy as np
from ultralytics import YOLO


# --------------------------------------------------------------------------
# Vehicle class mapping from COCO (the base model detects 80 categories)
# --------------------------------------------------------------------------
VEHICLE_CLASSES = {
    "car": (0, 165, 255),        # orange
    "truck": (0, 0, 255),        # red
    "bus": (255, 0, 0),          # blue
    "motorcycle": (0, 255, 255), # yellow
    "bicycle": (0, 255, 0),      # green
}


def load_model(model_path="yolov8n.pt"):
    print(f"[INFO] Loading model: {model_path} ...")
    model = YOLO(model_path)
    return model


# --------------------------------------------------------------------------
# Process a single image
# --------------------------------------------------------------------------
def process_image(model, image_path, conf_thresh=0.25, iou_thresh=0.45, output_dir="./output",
                   save_visualization=True, image_override=None):
    """image_path is used for naming/metrics either way. If image_override
    (a BGR array) is given, it is detected on instead of re-reading from
    disk -- lets callers feed an in-memory distorted image without ever
    writing it to disk."""
    img = image_override if image_override is not None else cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read: {image_path}")
        return None, []

    t0 = time.time()
    results = model.predict(img, conf=conf_thresh, iou=iou_thresh, verbose=False)
    infer_time = time.time() - t0

    result = results[0]
    names = result.names  # class_id -> class_name

    detections = []
    vis = img.copy() if save_visualization else None

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        if cls_name not in VEHICLE_CLASSES:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        detections.append({
            "image": os.path.basename(image_path),
            "class": cls_name,
            "confidence": round(conf, 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": w, "height": h,
            "box_area": w * h,
        })

        # Draw bounding box + label
        if save_visualization:
            color = VEHICLE_CLASSES[cls_name]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

    vis_path = None
    if save_visualization:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(output_dir, f"detect_{base_name}.png")
        cv2.imwrite(vis_path, vis)

    # Summary metrics for this image
    class_counts = {}
    for d in detections:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

    confidences = [d["confidence"] for d in detections]
    summary = {
        "image": os.path.basename(image_path),
        "total_vehicles": len(detections),
        "n_cars": class_counts.get("car", 0),
        "n_trucks": class_counts.get("truck", 0),
        "n_buses": class_counts.get("bus", 0),
        "n_motorcycles": class_counts.get("motorcycle", 0),
        "n_bicycles": class_counts.get("bicycle", 0),
        "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "min_confidence": round(float(np.min(confidences)), 4) if confidences else 0.0,
        "max_confidence": round(float(np.max(confidences)), 4) if confidences else 0.0,
        "inference_time_sec": round(infer_time, 4),
        "visualization_path": vis_path,
    }

    print(f"  [{summary['image']}] found {summary['total_vehicles']} vehicles "
          f"(cars={summary['n_cars']}, trucks={summary['n_trucks']}, "
          f"buses={summary['n_buses']}, motorcycles={summary['n_motorcycles']}, "
          f"bicycles={summary['n_bicycles']}) | time={summary['inference_time_sec']}s")

    return summary, detections


# --------------------------------------------------------------------------
# Collect images from a dataset directory
# --------------------------------------------------------------------------
def collect_images_from_dataset(dataset_dir, num_images=20):
    exts = ("*.jpg", "*.jpeg", "*.png")
    all_images = []
    for ext in exts:
        all_images.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
    all_images = sorted(all_images)

    if not all_images:
        raise FileNotFoundError(f"No images found in directory: {dataset_dir}")

    return all_images[:num_images]


# --------------------------------------------------------------------------
# Save cumulative CSV
# --------------------------------------------------------------------------
def save_csv(rows, csv_path, fieldnames):
    if not rows:
        print(f"[WARN] No data to save for {csv_path}")
        return
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[OK] Saved {len(rows)} rows to: {csv_path}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection on BDD100K images with YOLOv8")
    parser.add_argument("--dataset_dir", type=str, default=str(config.TASK3_CLEAN_DIR),
                         help="Path to the BDD100K image directory")
    parser.add_argument("--images", type=str, nargs="+", default=None, help="List of specific image paths")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--summary_csv_name", type=str, default="vehicle_detection_summary.csv")
    parser.add_argument("--detections_csv_name", type=str, default="vehicle_detection_details.csv")
    parser.add_argument("--model", type=str, default=str(config.YOLO_WEIGHTS_PATH),
                         help="Model weights: yolov8n.pt (fast) / yolov8s.pt / yolov8m.pt (more accurate)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--num_images", type=int, default=20, help="How many images to process from the dataset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv_path = os.path.join(args.output_dir, args.summary_csv_name)
    details_csv_path = os.path.join(args.output_dir, args.detections_csv_name)

    if args.images:
        image_paths = args.images
    elif args.dataset_dir:
        image_paths = collect_images_from_dataset(args.dataset_dir, args.num_images)
    else:
        raise ValueError("You must provide either --dataset_dir or --images")

    model = load_model(args.model)

    all_summaries = []
    all_detections = []

    print(f"\n[INFO] Processing {len(image_paths)} images...\n")
    for img_path in image_paths:
        summary, detections = process_image(
            model, img_path,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            output_dir=args.output_dir,
        )
        if summary:
            all_summaries.append(summary)
            all_detections.extend(detections)

    # Save two CSV files: per-image summary + per-detection detail
    summary_fields = ["image", "total_vehicles", "n_cars", "n_trucks", "n_buses",
                       "n_motorcycles", "n_bicycles", "avg_confidence", "min_confidence",
                       "max_confidence", "inference_time_sec", "visualization_path"]
    details_fields = ["image", "class", "confidence", "x1", "y1", "x2", "y2",
                       "width", "height", "box_area"]

    save_csv(all_summaries, summary_csv_path, summary_fields)
    save_csv(all_detections, details_csv_path, details_fields)

    # Overall summary metrics across the full set
    if all_summaries:
        total_vehicles = sum(s["total_vehicles"] for s in all_summaries)
        avg_per_image = total_vehicles / len(all_summaries)
        print(f"\n=== Overall Summary ===")
        print(f"Total images processed: {len(all_summaries)}")
        print(f"Total vehicles detected: {total_vehicles}")
        print(f"Average vehicles per image: {avg_per_image:.2f}")


if __name__ == "__main__":
    main()