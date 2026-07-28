"""
config.py
Central path configuration for the project. Every script should import
paths from here instead of hardcoding them.
"""

from pathlib import Path

# =========================
# Repo root (computed automatically -- works regardless of where
# scripts are actually run from, as long as this file stays at the
# repo root and nested scripts import it correctly -- see note below)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent

# =========================
# Clean image folders (per task)
# =========================
CLEAN_IMAGES_ROOT = PROJECT_ROOT / "data" / "clean_images"

TASK1_CLEAN_DIR = CLEAN_IMAGES_ROOT / "1_data_lane_detection_low_level"
TASK2_CLEAN_DIR = CLEAN_IMAGES_ROOT / "2_data_feature_matching_high_level"
TASK3_CLEAN_DIR = CLEAN_IMAGES_ROOT / "3_data_vehicle_detection_deep_learnning"

# =========================
# Augmented images
# -- LEGACY/ARCHIVED (TASKS.md Phase 3, 2026-07-16): Task 1 now uses the
# in-memory driver (src_Alon/lane_detection/run_lane_detection_degradation.py),
# same pattern as Tasks 2/3. These constants are only kept so the archived
# reference scripts in src_Alon/archive/ still run if anyone wants to consult
# them -- do not build anything new on top of this.
# =========================
AUGMENTED_ROOT = PROJECT_ROOT / "data" / "augmented_images"

# =========================
# Outputs
# =========================
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CSV_OUTPUT_DIR = OUTPUTS_ROOT / "csv_results"
GRAPH_OUTPUT_DIR = OUTPUTS_ROOT / "graph_results"
VISUALIZATIONS_ROOT = OUTPUTS_ROOT / "visualizations"

# Specific output files referenced by name across scripts
# -- LEGACY/ARCHIVED (TASKS.md Phase 3): flat CSV paths from before the
# per-task unification. Task 1 now writes to TASK1_CSV_DIR directly (see
# run_lane_detection_degradation.py); these two are only kept so the archived
# reference scripts in src_Alon/archive/ still run if anyone wants to consult
# them.
SNR_LOG_CSV = CSV_OUTPUT_DIR / "achieved_snr_log.csv"
LANE_COMPARISON_CSV = CSV_OUTPUT_DIR / "lane_comparison_results.csv"

# Make sure output directories exist so scripts don't fail on first run
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# Unified per-task result layout (target structure for Tasks 9-11):
#   outputs/csv_results/task{1,2,3}/     -- one consolidated CSV per task
#   outputs/graph_results/task{1,2,3}/   -- summary plots (bar/line charts)
#   outputs/visualizations/task{1,2,3}/  -- before/after sample images
# =========================
TASK1_CSV_DIR = CSV_OUTPUT_DIR / "task1"
TASK2_CSV_DIR = CSV_OUTPUT_DIR / "task2"
TASK3_CSV_DIR = CSV_OUTPUT_DIR / "task3"

TASK1_GRAPH_DIR = GRAPH_OUTPUT_DIR / "task1"
TASK2_GRAPH_DIR = GRAPH_OUTPUT_DIR / "task2"
TASK3_GRAPH_DIR = GRAPH_OUTPUT_DIR / "task3"

TASK1_VIS_DIR = VISUALIZATIONS_ROOT / "task1"
TASK2_VIS_DIR = VISUALIZATIONS_ROOT / "task2"
TASK3_VIS_DIR = VISUALIZATIONS_ROOT / "task3"

for _dir in (
    TASK1_CSV_DIR, TASK2_CSV_DIR, TASK3_CSV_DIR,
    TASK1_GRAPH_DIR, TASK2_GRAPH_DIR, TASK3_GRAPH_DIR,
    TASK1_VIS_DIR, TASK2_VIS_DIR, TASK3_VIS_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# =========================
# Amit's part
# =========================
AMIT_PROJECT_DIR = PROJECT_ROOT / "Amit_project"
YOLO_WEIGHTS_PATH = AMIT_PROJECT_DIR / "yolov8n.pt"

# =========================
# Ground truth (Task 3)
# -- Populated by Task 5 (src_Alon/helper_files/extract_task3_gt.py), which
# extracts real BDD100K box2d annotations for the Task 3 image subset.
# =========================
GT_LABELS_ROOT = PROJECT_ROOT / "data" / "gt_labels"
TASK3_GT_CSV = GT_LABELS_ROOT / "task3_vehicle_gt.csv"
GT_LABELS_ROOT.mkdir(parents=True, exist_ok=True)

# =========================
# Raw BDD100K annotation archive (the full-dataset labels zip, not committed --
# large, gitignored). Used by src_Alon/helper_files/import_tags.py and
# filter_bdd100k.py to pull real box2d/lane annotations for our image subset.
# -- LEGACY/unused for now: these assume a v1 "bdd100k_labels_images_train.json"
# inside a zip, which we don't have locally. See BDD100K_V2_* below instead --
# that's what extract_task3_gt.py actually uses.
# =========================
BDD100K_ZIP_PATH = PROJECT_ROOT / "data" / "bdd100k.zip"
BDD100K_EXTRACT_DIR = PROJECT_ROOT / "data" / "annotations_extracted"
BDD100K_LABELS_JSON_NAME = "bdd100k_labels_images_train.json"

# =========================
# BDD100K v2 detection labels (100k images) -- what we actually have locally.
# External/machine-specific path (large files, not copied into the repo):
# update this if running on a different machine.
# =========================
BDD100K_V2_LABELS_DIR = Path(r"C:\Users\amit\Downloads\archive\labels")
BDD100K_V2_TRAIN_JSON = BDD100K_V2_LABELS_DIR / "det_v2_train_release.json"
BDD100K_V2_VAL_JSON = BDD100K_V2_LABELS_DIR / "det_v2_val_release.json"

# =========================
# BDD100K label category -> COCO/YOLO vehicle class name (Task 3 GT).
# "rider"/"person"/"traffic light"/"traffic sign"/"train" are intentionally
# excluded -- not vehicle classes for this task.
# =========================
BDD100K_TO_COCO_VEHICLE_CLASS = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motor": "motorcycle",
    "bike": "bicycle",
}

# =========================
# Shared constants (used by multiple scripts -- keep them here so both
# your code and Amit's stay in sync on naming/levels)
# =========================
AUGMENTATIONS = ["motion_blur", "low_light", "rain"]
NUM_LEVELS = 9
LEVEL_NAMES = [f"level_{i}" for i in range(1, NUM_LEVELS + 1)]

MAX_ERROR_PX = 45  # pixel-distance threshold used for "survival" in threshold_survival.py


if __name__ == "__main__":
    # Quick sanity check: run `python config.py` to print resolved paths
    # and confirm everything points where you expect before wiring it
    # into the other scripts.
    print(f"PROJECT_ROOT:        {PROJECT_ROOT}")
    print(f"TASK1_CLEAN_DIR:     {TASK1_CLEAN_DIR}  (exists: {TASK1_CLEAN_DIR.exists()})")
    print(f"TASK2_CLEAN_DIR:     {TASK2_CLEAN_DIR}  (exists: {TASK2_CLEAN_DIR.exists()})")
    print(f"TASK3_CLEAN_DIR:     {TASK3_CLEAN_DIR}  (exists: {TASK3_CLEAN_DIR.exists()})")
    print(f"YOLO_WEIGHTS_PATH:   {YOLO_WEIGHTS_PATH}  (exists: {YOLO_WEIGHTS_PATH.exists()})")

    print("\n-- Unified per-task output layout --")
    for task_n, csv_dir, graph_dir, vis_dir in (
        (1, TASK1_CSV_DIR, TASK1_GRAPH_DIR, TASK1_VIS_DIR),
        (2, TASK2_CSV_DIR, TASK2_GRAPH_DIR, TASK2_VIS_DIR),
        (3, TASK3_CSV_DIR, TASK3_GRAPH_DIR, TASK3_VIS_DIR),
    ):
        print(f"  task{task_n}: csv={csv_dir}  graph={graph_dir}  vis={vis_dir}")

    print(f"\nTASK3_GT_CSV:        {TASK3_GT_CSV}  (exists: {TASK3_GT_CSV.exists()})")

    print("\n-- Legacy/archived (only used by src_Alon/archive/ reference scripts) --")
    print(f"AUGMENTED_ROOT:      {AUGMENTED_ROOT}  (exists: {AUGMENTED_ROOT.exists()})")
    print(f"SNR_LOG_CSV:         {SNR_LOG_CSV}")
    print(f"LANE_COMPARISON_CSV: {LANE_COMPARISON_CSV}")