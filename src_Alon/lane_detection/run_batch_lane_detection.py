import os
import glob

from lane_detection_pipeline import process_image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def main():
    folder_path = config.TASK1_CLEAN_DIR
    num_images = 2

    # Grab image files from the folder (add/remove extensions as needed)
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    image_paths = sorted(image_paths)[:num_images]

    if not image_paths:
        print(f"No images found in: {folder_path}")
        return

    print(f"Found {len(image_paths)} image(s) to process.\n")

    results = []
    for path in image_paths:
        result = process_image(path, show=True)
        if result is not None:
            results.append(result)

    print(f"\nProcessed {len(results)} / {len(image_paths)} images successfully.")


if __name__ == "__main__":
    main()
