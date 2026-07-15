"""
Vehicle Detection על תמונות מ-BDD100K באמצעות YOLOv8
======================================================

מה הסקריפט עושה:
1. טוען תמונות (מתיקיית dataset או קבצים ספציפיים)
2. מריץ מודל YOLOv8 (pretrained על COCO) לזיהוי רכבים
3. מסנן רק את הקטגוריות הרלוונטיות לרכבים:
   car, truck, bus, motorcycle, bicycle
4. שומר:
   - תמונה ויזואלית עם Bounding Boxes מצוירים + label + confidence
   - מטריקות מספריות (מס' רכבים, פילוח לפי סוג, confidence ממוצע, זמן ריצה)
   - קובץ CSV מצטבר עם כל הזיהויים (bbox, class, confidence) לכל תמונה

התקנה (חד פעמי):
    pip install ultralytics

איך מריצים:
    python vehicle_detection_bdd100k.py --dataset_dir /path/to/bdd100k/images \
                                          --output_dir ./output \
                                          --num_images 20 \
                                          --model yolov8n.pt \
                                          --conf 0.25

או על תמונות ספציפיות:
    python vehicle_detection_bdd100k.py --images img1.jpg img2.jpg --output_dir ./output
"""

import os
import cv2
import csv
import glob
import time
import argparse
import numpy as np
from ultralytics import YOLO


# --------------------------------------------------------------------------
# מיפוי קטגוריות רכבים מתוך COCO (המודל המקורי מזהה 80 קטגוריות)
# --------------------------------------------------------------------------
VEHICLE_CLASSES = {
    "car": (0, 165, 255),        # כתום
    "truck": (0, 0, 255),        # אדום
    "bus": (255, 0, 0),          # כחול
    "motorcycle": (0, 255, 255), # צהוב
    "bicycle": (0, 255, 0),      # ירוק
}


def load_model(model_path="yolov8n.pt"):
    print(f"[INFO] טוען מודל: {model_path} ...")
    model = YOLO(model_path)
    return model


# --------------------------------------------------------------------------
# עיבוד תמונה בודדת
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

        # ציור Bounding Box + Label
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

    # מטריקות מסכמות לתמונה הזו
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

    print(f"  [{summary['image']}] נמצאו {summary['total_vehicles']} רכבים "
          f"(cars={summary['n_cars']}, trucks={summary['n_trucks']}, "
          f"buses={summary['n_buses']}, motorcycles={summary['n_motorcycles']}, "
          f"bicycles={summary['n_bicycles']}) | זמן={summary['inference_time_sec']}s")

    return summary, detections


# --------------------------------------------------------------------------
# איסוף תמונות מתוך תיקיית dataset
# --------------------------------------------------------------------------
def collect_images_from_dataset(dataset_dir, num_images=20):
    exts = ("*.jpg", "*.jpeg", "*.png")
    all_images = []
    for ext in exts:
        all_images.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
    all_images = sorted(all_images)

    if not all_images:
        raise FileNotFoundError(f"לא נמצאו תמונות בתיקייה: {dataset_dir}")

    return all_images[:num_images]


# --------------------------------------------------------------------------
# שמירת CSV מצטבר
# --------------------------------------------------------------------------
def save_csv(rows, csv_path, fieldnames):
    if not rows:
        print(f"[WARN] אין נתונים לשמירה עבור {csv_path}")
        return
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[OK] נשמרו {len(rows)} שורות לקובץ: {csv_path}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Vehicle Detection על תמונות מ-BDD100K עם YOLOv8")
    parser.add_argument("--dataset_dir", type=str, default=None, help="נתיב לתיקיית תמונות BDD100K")
    parser.add_argument("--images", type=str, nargs="+", default=None, help="רשימת נתיבי תמונות ספציפיים")
    parser.add_argument("--output_dir", type=str, default="./output", help="תיקיית פלט")
    parser.add_argument("--summary_csv_name", type=str, default="vehicle_detection_summary.csv")
    parser.add_argument("--detections_csv_name", type=str, default="vehicle_detection_details.csv")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="משקלות מודל: yolov8n.pt (מהיר) / yolov8s.pt / yolov8m.pt (מדויק יותר)")
    parser.add_argument("--conf", type=float, default=0.25, help="סף confidence לזיהוי")
    parser.add_argument("--iou", type=float, default=0.45, help="סף IoU ל-NMS")
    parser.add_argument("--num_images", type=int, default=20, help="כמה תמונות לעבד מתוך ה-dataset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv_path = os.path.join(args.output_dir, args.summary_csv_name)
    details_csv_path = os.path.join(args.output_dir, args.detections_csv_name)

    if args.images:
        image_paths = args.images
    elif args.dataset_dir:
        image_paths = collect_images_from_dataset(args.dataset_dir, args.num_images)
    else:
        raise ValueError("יש לספק either --dataset_dir או --images")

    model = load_model(args.model)

    all_summaries = []
    all_detections = []

    print(f"\n[INFO] מעבד {len(image_paths)} תמונות...\n")
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

    # שמירת שני קבצי CSV: סיכום לכל תמונה + פירוט לכל זיהוי בודד
    summary_fields = ["image", "total_vehicles", "n_cars", "n_trucks", "n_buses",
                       "n_motorcycles", "n_bicycles", "avg_confidence", "min_confidence",
                       "max_confidence", "inference_time_sec", "visualization_path"]
    details_fields = ["image", "class", "confidence", "x1", "y1", "x2", "y2",
                       "width", "height", "box_area"]

    save_csv(all_summaries, summary_csv_path, summary_fields)
    save_csv(all_detections, details_csv_path, details_fields)

    # מטריקות כלליות על כל הסט
    if all_summaries:
        total_vehicles = sum(s["total_vehicles"] for s in all_summaries)
        avg_per_image = total_vehicles / len(all_summaries)
        print(f"\n=== סיכום כללי ===")
        print(f"סה\"כ תמונות שעובדו: {len(all_summaries)}")
        print(f"סה\"כ רכבים שזוהו: {total_vehicles}")
        print(f"ממוצע רכבים לתמונה: {avg_per_image:.2f}")


if __name__ == "__main__":
    main()
