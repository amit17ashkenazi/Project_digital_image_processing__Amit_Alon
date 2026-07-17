import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

import os
import json
import shutil
import random
import zipfile
import ijson

# ============================================================
# PATHS (from config.py -- do not hardcode machine-specific paths here)
# ============================================================

# Folder containing ONLY the selected images for this task
SELECTED_IMAGES = str(config.TASK3_CLEAN_DIR)

# Path to the big zip file containing the BDD100K annotations
ZIP_PATH = str(config.BDD100K_ZIP_PATH)

# Where to extract just the needed JSON file to
EXTRACT_TO = str(config.BDD100K_EXTRACT_DIR)

# The exact name/path of the JSON file INSIDE the zip
ZIP_INTERNAL_JSON_NAME = config.BDD100K_LABELS_JSON_NAME

# Output dataset
OUTPUT = str(config.PROJECT_ROOT / "data" / "BDD150")

# Train / Val / Test split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# STEP 0 (optional, run once): List contents of the zip
# to find the exact internal name of the JSON file.
# Uncomment to inspect, then comment back out once you know the name.
# ============================================================

# with zipfile.ZipFile(ZIP_PATH, "r") as zf:
#     for name in zf.namelist():
#         print(name)
# exit()

# ============================================================
# Create output folders
# ============================================================

for folder in [
    "images/train",
    "images/val",
    "images/test",
    "labels/train",
    "labels/val",
    "labels/test"
]:
    os.makedirs(os.path.join(OUTPUT, folder), exist_ok=True)

# ============================================================
# STEP 1: Extract ONLY the needed JSON file from the zip
# (does not extract/touch anything else in the archive)
# ============================================================

os.makedirs(EXTRACT_TO, exist_ok=True)
extracted_json_path = os.path.join(EXTRACT_TO, ZIP_INTERNAL_JSON_NAME)

if not os.path.exists(extracted_json_path):
    print(f"Extracting {ZIP_INTERNAL_JSON_NAME} from {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extract(ZIP_INTERNAL_JSON_NAME, EXTRACT_TO)
    print("Extraction done.")
else:
    print("JSON already extracted, skipping extraction.")

ANNOTATION_JSON = extracted_json_path

# ============================================================
# STEP 2: Selected images list (needed BEFORE parsing JSON,
# so we know which annotations to keep while streaming)
# ============================================================

images = sorted(os.listdir(SELECTED_IMAGES))
needed_names = set(images)

random.seed(42)
random.shuffle(images)

n = len(images)

train_end = int(TRAIN_RATIO * n)
val_end = train_end + int(VAL_RATIO * n)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

splits = {
    "train": train_images,
    "val": val_images,
    "test": test_images
}

print(f"Train: {len(train_images)}")
print(f"Val:   {len(val_images)}")
print(f"Test:  {len(test_images)}")

# ============================================================
# STEP 3: Stream-parse the big JSON, keeping ONLY the
# annotations for our ~150 selected images (memory-safe)
# ============================================================

print("Streaming JSON and extracting only needed annotations...")

annotation_dict = {}

with open(ANNOTATION_JSON, "rb") as f:
    for item in ijson.items(f, "item"):
        name = item.get("name")
        if name in needed_names:
            annotation_dict[name] = item

print(f"Found annotations for {len(annotation_dict)} of {len(needed_names)} selected images.")

missing = needed_names - set(annotation_dict.keys())
if missing:
    print(f"WARNING: {len(missing)} selected images have NO matching annotation entry.")
    print("Example missing names:", list(missing)[:5])

# ============================================================
# Category mapping
# ============================================================

classes = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "motor": 3,
    "bike": 4,
    "person": 5
}

# ============================================================
# STEP 4: Convert to YOLO format (same logic as original)
# ============================================================

for split in splits:

    for image_name in splits[split]:

        src = os.path.join(SELECTED_IMAGES, image_name)

        dst = os.path.join(
            OUTPUT,
            "images",
            split,
            image_name
        )

        shutil.copy(src, dst)

        ann = annotation_dict.get(image_name)

        label_path = os.path.join(
            OUTPUT,
            "labels",
            split,
            image_name.replace(".jpg", ".txt")
        )

        with open(label_path, "w") as f:

            if ann is None:
                continue

            for obj in ann["labels"]:

                if "box2d" not in obj:
                    continue

                category = obj["category"]

                if category not in classes:
                    continue

                cls = classes[category]

                box = obj["box2d"]

                x1 = box["x1"]
                y1 = box["y1"]
                x2 = box["x2"]
                y2 = box["y2"]

                width = 1280
                height = 720

                xc = ((x1 + x2) / 2) / width
                yc = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height

                f.write(
                    f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
                )

print("Finished.")