# Archive

Superseded by `src_Alon/lane_detection/run_lane_detection_degradation.py`
(TASKS.md Phase 3, 2026-07-16): a single in-memory driver replacing this
disk-based two-step flow (write augmented images to disk, then compare
against them), matching the pattern Tasks 2/3 already used.

Kept here for reference rather than deleted, since it's a working
implementation of the same idea and Alon may want to consult it later.

- `create_augmented_images.py` — wrote distorted images to `data/augmented_images/` + a SNR log CSV.
- `compere_gt_to_aug.py` — ran the lane pipeline on clean vs. those on-disk augmented images, diffed the detected lines.
- `threshold_survival.py` — plotted survival-rate-vs-SNR from `compere_gt_to_aug.py`'s output CSV.
