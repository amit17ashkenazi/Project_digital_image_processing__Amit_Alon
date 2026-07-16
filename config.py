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
# =========================
AUGMENTED_ROOT = PROJECT_ROOT / "data" / "augmented_images"

# =========================
# Outputs
# =========================
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
CSV_OUTPUT_DIR = OUTPUTS_ROOT / "csv_results"
GRAPH_OUTPUT_DIR = OUTPUTS_ROOT / "graph_results"

# Specific output files referenced by name across scripts
SNR_LOG_CSV = CSV_OUTPUT_DIR / "achieved_snr_log.csv"
LANE_COMPARISON_CSV = CSV_OUTPUT_DIR / "lane_comparison_results.csv"

# Make sure output directories exist so scripts don't fail on first run
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Amit's part
# =========================
AMIT_PROJECT_DIR = PROJECT_ROOT / "Amit_project"
YOLO_WEIGHTS_PATH = AMIT_PROJECT_DIR / "yolov8n.pt"
TASK2_RESULTS_DIR = OUTPUTS_ROOT / "task2_feature_matching"
TASK3_RESULTS_DIR = OUTPUTS_ROOT / "task3_vehicle_detection"
TASK2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    print(f"AUGMENTED_ROOT:      {AUGMENTED_ROOT}  (exists: {AUGMENTED_ROOT.exists()})")
    print(f"SNR_LOG_CSV:         {SNR_LOG_CSV}")
    print(f"LANE_COMPARISON_CSV: {LANE_COMPARISON_CSV}")
    print(f"YOLO_WEIGHTS_PATH:   {YOLO_WEIGHTS_PATH}  (exists: {YOLO_WEIGHTS_PATH.exists()})")