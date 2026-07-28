"""
Part 4 -- fine-tunes yolov8n.pt on the distorted training set built by
build_finetune_dataset.py (Task 3, real-GT labels, fixed moderate-severity
distortion). Mirrors the course brief's own minimal example (slide 34:
epochs=3, batch=2, device="cpu"), scaled up modestly since our dataset
(240 train images) is a bit larger than the slide's toy example.

Usage:
    python finetune_vehicle_detection.py
    python finetune_vehicle_detection.py --epochs 20 --batch 8
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

import argparse
import shutil

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on distorted Task 3 images")
    parser.add_argument("--data_yaml", default=str(config.PROJECT_ROOT / "data" / "finetune_dataset" / "data.yaml"))
    parser.add_argument("--base_weights", default=str(config.YOLO_WEIGHTS_PATH))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_weights", default=str(Path(__file__).resolve().parent / "yolov8n_finetuned.pt"))
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found. Run build_finetune_dataset.py first.")

    model = YOLO(args.base_weights)
    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        verbose=True,
    )

    best = Path(model.trainer.best) if hasattr(model, "trainer") else None
    if best is None or not best.exists():
        # fallback to ultralytics' default run location
        candidates = sorted(Path("runs/detect").glob("train*/weights/best.pt"))
        best = candidates[-1] if candidates else None

    if best is None or not best.exists():
        raise RuntimeError("Could not locate best.pt after training -- check the runs/detect/ output above.")

    out_path = Path(args.out_weights)
    shutil.copy2(best, out_path)
    print(f"\nFine-tuned weights copied to {out_path.resolve()}")


if __name__ == "__main__":
    main()
