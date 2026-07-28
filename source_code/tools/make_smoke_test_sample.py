"""
Creates a small, fixed smoke-test sample per task under smoke_test/, so every
phase's quick-check can run in seconds instead of over the full dataset.
Re-run any time to regenerate (deterministic: alphabetical filenames, fixed
count) -- smoke_test/ is gitignored like all images, nothing is lost by not
committing it.

Usage:
    python tools/make_smoke_test_sample.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

SAMPLE_SIZE = 6
SMOKE_ROOT = config.PROJECT_ROOT / "smoke_test"

FRAME_RE = re.compile(r"^(?P<prefix>.+)-(?P<frame>\d+)$")


def reset_dir(dst_dir: Path):
    """Clear any stale files from a previous run before repopulating --
    otherwise leftover images from an older source dataset silently linger
    (e.g. after the source folder was pruned/changed)."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def copy_simple_sample(src_dir: Path, dst_dir: Path, n: int = SAMPLE_SIZE):
    reset_dir(dst_dir)
    images = list_images(src_dir)[:n]
    for img in images:
        shutil.copy2(img, dst_dir / img.name)
    return images


def copy_sequence_sample(src_dir: Path, dst_dir: Path, n: int = SAMPLE_SIZE):
    """Task 2 needs at least one real <video_id>-<frame> consecutive sequence
    so collect_pairs_from_dataset(mode="sequence") finds a pair -- a random
    alphabetical slice isn't guaranteed to contain one."""
    reset_dir(dst_dir)
    images = list_images(src_dir)

    groups: dict[str, list[Path]] = {}
    for img in images:
        m = FRAME_RE.match(img.stem)
        if not m:
            continue
        groups.setdefault(m.group("prefix"), []).append(img)

    # pick the largest sequence so we're guaranteed >= 2 consecutive frames
    best_prefix = max(groups, key=lambda k: len(groups[k])) if groups else None
    chosen: list[Path] = []
    if best_prefix:
        chosen.extend(sorted(groups[best_prefix])[:n])

    # top up with other images (alphabetical) if the sequence is shorter than n
    for img in images:
        if len(chosen) >= n:
            break
        if img not in chosen:
            chosen.append(img)

    for img in chosen:
        shutil.copy2(img, dst_dir / img.name)
    return chosen


def main():
    t1 = copy_simple_sample(config.TASK1_CLEAN_DIR, SMOKE_ROOT / "task1")
    t2 = copy_sequence_sample(config.TASK2_CLEAN_DIR, SMOKE_ROOT / "task2")
    t3 = copy_simple_sample(config.TASK3_CLEAN_DIR, SMOKE_ROOT / "task3")

    print(f"task1: {len(t1)} images -> {SMOKE_ROOT / 'task1'}")
    print(f"task2: {len(t2)} images -> {SMOKE_ROOT / 'task2'} (includes a consecutive sequence)")
    print(f"task3: {len(t3)} images -> {SMOKE_ROOT / 'task3'}")


if __name__ == "__main__":
    main()
