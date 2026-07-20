# Project Task List

Practical, small, executable tasks to close the gaps identified in the assignment
gap analysis. Each task is self-contained: give Claude just the **Task name** and
it has enough context here to implement it correctly in this repo.

Tasks are grouped into phases in dependency order — earlier phases unblock later
ones (e.g. Phase 1 must land before Phase 4's per-class plots make sense). Task
IDs use `Phase.Index` numbering (e.g. `0.1`) so inserting a new task into a phase
later (like the quick-check tasks below) never renumbers anything else.

Every phase ends with a **quick-check task**: a fast, small-scale sanity pass
(reusing the project's own `smoke_test/` sample-data convention) that confirms
the phase's work is wired correctly before moving to the next phase — without
waiting for a full 378-image / 9-level run.

| ID | Task name | Phase |
|---|---|---|
| 0.1 | Reconcile clean-image data paths in config.py | 0 — Foundation |
| 0.2 | Parameterize Alon's helper scripts | 0 — Foundation |
| 0.3 | Create a shared smoke-test sample set | 0 — Foundation |
| 0.4 | Phase 0 quick-check — foundation sanity pass | 0 — Foundation |
| 1.1 | Define BDD100K-to-COCO vehicle category map | 1 — Task 3 GT |
| 1.2 | Build Task 3 GT extraction script | 1 — Task 3 GT |
| 1.3 | Rewire Task 3 driver to use real GT baseline | 1 — Task 3 GT |
| 1.4 | Add class-aware IoU matching | 1 — Task 3 GT |
| 1.5 | Add per-class metric aggregation and plots | 1 — Task 3 GT |
| 1.6 | Update README Task 3 methodology section | 1 — Task 3 GT |
| 1.7 | Phase 1 quick-check — Task 3 GT sanity pass | 1 — Task 3 GT |
| 2.1 | Wire enhancement pass into Task 2 driver | 2 — Enhancement wiring |
| 2.2 | Wire enhancement pass into Task 3 driver | 2 — Enhancement wiring |
| 2.3 | Generate enhancement comparison plots (Task 2) | 2 — Enhancement wiring |
| 2.4 | Generate enhancement comparison plots (Task 3) | 2 — Enhancement wiring |
| 2.5 | Phase 2 quick-check *(TBD — design when we reach this phase)* | 2 — Enhancement wiring |
| 3.1 | Build in-memory Task 1 driver | 3 — Task 1 migration |
| 3.2 | Wire real BDD100K lane GT into Task 1 metric | 3 — Task 1 migration |
| 3.3 | Add enhancement step to Task 1 | 3 — Task 1 migration |
| 3.4 | Retire Task 1 disk-based augmentation scripts | 3 — Task 1 migration |
| 3.5 | Update README Task 1 section | 3 — Task 1 migration |
| 3.6 | Phase 3 quick-check *(TBD — design when we reach this phase)* | 3 — Task 1 migration |
| 4.1 | Migrate Task 2 output to unified outputs/ layout | 4 — Unified layout |
| 4.2 | Migrate Task 3 output to unified outputs/ layout | 4 — Unified layout |
| 4.3 | Migrate Task 1 output to unified outputs/ layout | 4 — Unified layout |
| 4.4 | Clean up legacy/duplicate output directories | 4 — Unified layout |
| 4.5 | Phase 4 quick-check *(TBD — design when we reach this phase)* | 4 — Unified layout |
| 5.1 | Generate pseudo-labels from clean-image baseline | 5 — Fine-tuning |
| 5.2 | Build distorted fine-tuning training set | 5 — Fine-tuning |
| 5.3 | Fine-tune YOLOv8n on distorted images | 5 — Fine-tuning |
| 5.4 | Re-run distortion sweep with fine-tuned weights and compare | 5 — Fine-tuning |
| 5.5 | Phase 5 quick-check | 5 — Fine-tuning |
| 6.1 | Embed Task 1 results and interpretation in README | 6 — README/submission |
| 6.2 | Embed Task 2 results and interpretation in README | 6 — README/submission |
| 6.3 | Embed Task 3 results and interpretation in README | 6 — README/submission |
| 6.4 | Fill remaining README TODOs and prepare final PPT | 6 — README/submission |
| 6.5 | Phase 6 quick-check *(TBD — design when we reach this phase)* | 6 — README/submission |

---

## Phase 0 — Foundation

### 0.1 Reconcile clean-image data paths in config.py

**Goal:** Make `config.py`'s `TASK{1,2,3}_CLEAN_DIR` point at wherever the clean
images actually live on disk, so scripts stop needing manual `--clean_dir`
overrides.

**Input:** `config.py` (`CLEAN_IMAGES_ROOT`, `TASK1_CLEAN_DIR`, `TASK2_CLEAN_DIR`,
`TASK3_CLEAN_DIR`). Current real location of the 3 image folders (decide: copy
them into `data/clean_images/` inside this repo, or point `CLEAN_IMAGES_ROOT` at
an external path via an env var/local override file).

**Implementation:** Edit `config.py` only. If copying images in, also confirm
`.gitignore`'s `*.jpg`/`*.png` rule still excludes them from git (it does).

**Location:** `config.py` (repo root).

**Output:** No new output files — this only fixes path resolution.

**Verification:** `python config.py` prints `exists: True` for all three
`TASK{1,2,3}_CLEAN_DIR` lines. Existing drivers run with zero `--clean_dir`
argument and still find images.

**Status: ✅ Done (2026-07-16).** No `config.py` code change was actually
needed — the paths were already correct, just unpopulated. Copied the 3
clean-image folders from `C:\Users\amit\Desktop\data` (**not** `data1`, which
is a stale/earlier snapshot with different image counts — corrected by the
user mid-task) into `data/clean_images/`. `py -3 config.py` now shows
`exists: True` for all three `TASK{1,2,3}_CLEAN_DIR`. (Also: `python` isn't on
PATH on this machine, use `py -3`.)

---

### 0.2 Parameterize Alon's helper scripts

**Goal:** Remove hardcoded absolute Windows paths from
`src_Alon/helper_files/import_tags.py` and `filter_bdd100k.py` so they run on
either teammate's machine and follow the same `config.py`-import convention as
every other script in the repo.

**Input:** `src_Alon/helper_files/import_tags.py` (hardcoded `SELECTED_IMAGES`,
`ZIP_PATH`, `EXTRACT_TO`), `src_Alon/helper_files/filter_bdd100k.py` (hardcoded
`zip_path = "data/bdd100k.zip"` — this one's already relative, just needs the
`config.py` import added for consistency).

**Implementation:** Add the standard
`sys.path.insert(0, str(Path(__file__).resolve().parents[2])); import config`
header (matching `lane_detection_pipeline.py`'s pattern). Replace hardcoded
paths with `config.TASK3_CLEAN_DIR`, a new `config.BDD100K_ZIP_PATH =
PROJECT_ROOT / "data" / "bdd100k.zip"`, and `config.PROJECT_ROOT / "data" /
"annotations_extracted"`.

**Location:** `src_Alon/helper_files/import_tags.py`,
`src_Alon/helper_files/filter_bdd100k.py`, plus one new constant added to
`config.py`.

**Output:** No new output files — behavior-preserving refactor.

**Verification:** Both scripts run unmodified-except-for-paths and produce the
same output as before (`BDD150/` for `import_tags.py`, `matched_images/` for
`filter_bdd100k.py`). `grep -r "C:\\\\Users\\\\alonk"` in `src_Alon/` returns
nothing.

**Status: ✅ Done (2026-07-16).** Both scripts now import `config` and read
`config.BDD100K_ZIP_PATH` / `BDD100K_EXTRACT_DIR` / `BDD100K_LABELS_JSON_NAME`
/ `TASK3_CLEAN_DIR` (new constants added to `config.py`, along with a
`BDD100K_TO_COCO_VEHICLE_CLASS` map — a small piece of Task 1.1 pulled
forward since it was a one-line addition alongside the others).
`grep -r "alonk" src_Alon/` is clean except one unrelated demo snippet in
`src_Alon/create_aug/augmentations.py` (out of scope for this task, not
touched). `filter_bdd100k.py` runs and fails only at
`FileNotFoundError: ...\data\bdd100k.zip` — confirms correct, repo-relative
path resolution; the actual zip is a Phase 1 prerequisite.

---

### 0.3 Create a shared smoke-test sample set

**Goal:** A tiny, fixed sample of images per task (5–8 each) that every phase's
quick-check can run against in seconds instead of the full 378/180/378-image
sweep — this reuses the `smoke_test/` convention already used earlier in this
project's history (`smoke_test/clean_task2`, `smoke_test/clean_task3`, runs
with `--num_levels 2`), just formalized into a reusable script instead of being
recreated ad hoc each time.

**Input:** `config.TASK1_CLEAN_DIR`, `TASK2_CLEAN_DIR`, `TASK3_CLEAN_DIR`
(post Task 0.1).

**Implementation:** Small script: for each task, take the first N
(alphabetically sorted, for determinism) filenames from the clean folder and
copy them into `smoke_test/task{1,2,3}/`. For Task 2 specifically, make sure
the sample includes at least one full `<video_id>` sequence of ≥2 consecutive
frames (so `collect_pairs_from_dataset(mode="sequence")` finds at least one
pair) rather than N arbitrary images.

**Location:** New script `tools/make_smoke_test_sample.py` (create a `tools/`
folder at repo root if one doesn't exist yet).

**Output:** `smoke_test/task1/`, `smoke_test/task2/`, `smoke_test/task3/`
(gitignored, like all images — regenerable in seconds by re-running the
script, so nothing is lost by not committing them).

**Verification:** Each `smoke_test/task{n}/` folder exists with the expected
small image count; Task 2's sample contains a valid consecutive-frame pair.

**Status: ✅ Done (2026-07-16).** `tools/make_smoke_test_sample.py` created
and run. `smoke_test/task1/` and `task3/` each have 6 images; `smoke_test/
task2/` has 6 images all from one `00e9be89` video sequence (frames
15→100→105→110→115→120), confirmed to yield 5 valid consecutive pairs when
the Task 2 driver ran on it in 0.4 below.

**Bug found and fixed during Phase 1 (2026-07-16):** the script didn't clear
destination folders before copying, so after Task 3's clean-image set was
pruned 675→300 (see 1.2), `smoke_test/task3/` kept 2 stale leftover files
from before the prune (`001b428f-059bac33.jpg`, `001c2a14-c7138401.jpg` —
images no longer in Task 3's actual dataset). This showed up as "images with
no GT" in the 1.7 quick-check output, which was misleading — those 2 files
simply shouldn't have been in the sample at all. Added `reset_dir()`
(`shutil.rmtree` + recreate) before every copy; re-ran, `smoke_test/task3/`
now has exactly 6 correct files. Separately confirmed the real 300-image set
*does* have 2 (different, legitimate) zero-vehicle images —
`648dc925-e359dbec.jpg`, `738a2bcd-7a6fde0d.jpg` — both labeled only
`traffic sign` in BDD100K's real GT, i.e. genuinely no vehicles in frame, not
a data gap.

---

### 0.4 Phase 0 quick-check — foundation sanity pass

**Goal:** Confirm the foundation work (Tasks 0.1–0.3) is solid *before*
building GT extraction, enhancement wiring, or driver rewrites on top of it —
catching a broken path or a bad smoke sample here is much cheaper than
discovering it three phases later.

**Input:** `config.py` (post 0.1), `import_tags.py`/`filter_bdd100k.py` (post
0.2), `smoke_test/task{1,2,3}/` (post 0.3).

**Implementation:** No new code — this is a verification pass using what
already exists:
1. Run `python config.py` from the repo root.
2. Run `import_tags.py` and `filter_bdd100k.py` pointed at `smoke_test/task3/`
   / `smoke_test/task1/` respectively (small `MAX_IMAGES`/sample size, so this
   finishes in well under a minute).
3. Run the *existing* (not-yet-modified) Task 2 and Task 3 drivers against the
   smoke sample as a baseline snapshot to diff against later:
   `python run_feature_matching_degradation.py --clean_dir ../smoke_test/task2 --out_dir ../smoke_test/results_task2_baseline --num_levels 2`
   `python run_vehicle_detection_degradation.py --clean_dir ../smoke_test/task3 --out_dir ../smoke_test/results_task3_baseline --num_levels 2`

**Location:** Run from repo root / `Amit_project/` (no new files created by
this task itself, beyond the smoke-test output directories below).

**Output:** Console output only, plus
`smoke_test/results_task2_baseline/`, `smoke_test/results_task3_baseline/` —
kept around specifically so Phase 1/2's quick-checks can diff their new
numbers against this pre-change snapshot.

**Verification:** (a) `config.py`'s printed output shows `exists: True` for
all three clean dirs; (b) both helper scripts complete without a path/
`FileNotFoundError`; (c) both driver smoke runs complete and produce non-empty
CSVs + at least one PNG each in the `_baseline` output folders. If all three
pass, Phase 0 is done and it's safe to start Phase 1.

**Status: ✅ Done (2026-07-16).** (a) passed — all 3 `TASK{1,2,3}_CLEAN_DIR`
show `exists: True` after populating `data/clean_images/` from
`Desktop/data` (see note on 0.1 above — **not** `data1`, which is stale).
(b) partially verified: both helper scripts now resolve paths via `config.py`
correctly (confirmed `filter_bdd100k.py` fails only at the missing
`data/bdd100k.zip`, with the *correct* repo-relative path in the error — not
a path bug). Full end-to-end run of these two scripts is blocked on actually
having the BDD100K labels locally, which is Phase 1's job, so this part
carries forward as a Phase 1 prerequisite rather than a Phase 0 blocker.
**Note found in passing:** a local `det_v2_train_release.json`/
`det_v2_val_release.json` pair exists at
`C:\Users\amit\Downloads\archive\labels\`, but that's BDD100K's **v2**
detection format, not the `bdd100k_labels_images_train.json` (v1) filename/
schema `import_tags.py`/`filter_bdd100k.py` expect — Task 1.2 needs to check
whether the v2 field names (`box2d` vs. whatever v2 calls it) match before
reusing that parsing code, or source the actual v1 file instead.
(c) passed — both drivers ran clean on `smoke_test/task{2,3}` with
`--num_levels 2` and produced non-empty CSVs + PNGs in
`smoke_test/results_task{2,3}_baseline/` (kept as the pre-GT/pre-enhancement
snapshot for Phase 1/2's quick-checks to diff against).

---

## Phase 1 — Task 3 Ground Truth Integration

### 1.1 Define BDD100K-to-COCO vehicle category map

**Goal:** One shared mapping from BDD100K's label categories to the COCO/YOLO
class names `vehicle_detection_bdd100k.py` already reports, so GT extraction
and detection use identical class names.

**Input:** `Amit_project/vehicle_detection_bdd100k.py`'s existing
`VEHICLE_CLASSES` dict (`car`, `truck`, `bus`, `motorcycle`, `bicycle`).
BDD100K's label categories (`car`, `truck`, `bus`, `motor`, `bike`, `rider`,
`person`, `traffic light`, `traffic sign`, `train`).

**Implementation:** Add a `BDD100K_TO_COCO_VEHICLE_CLASS` dict to `config.py`:
`{"car": "car", "truck": "truck", "bus": "bus", "motor": "motorcycle", "bike":
"bicycle"}`. Explicitly document that `rider`/`person`/`traffic light`/`traffic
sign`/`train` are excluded (not vehicle classes for this task) — add this as a
one-line comment, not code.

**Location:** `config.py` (repo root), next to the existing `AUGMENTATIONS`/
`NUM_LEVELS` shared-constants block.

**Output:** No output files — this is a constant others import.

**Verification:** `from config import BDD100K_TO_COCO_VEHICLE_CLASS` succeeds
and the dict has exactly 5 entries matching `VEHICLE_CLASSES`' keys as values.

**Status: ✅ Done (2026-07-16).** Added to `config.py` during 0.2. Confirmed
working in 1.2's extraction run.

---

### 1.2 Build Task 3 GT extraction script

**Goal:** Produce `data/gt_labels/task3_vehicle_gt.csv` — one row per real
BDD100K box2d annotation, for every image in the Task 3 clean-image folder,
with the category already remapped to COCO names.

**Input:** `data/bdd100k.zip` (must be present locally — download from BDD100K
if missing), `bdd100k_labels_images_train.json` inside it,
`config.TASK3_CLEAN_DIR` (the 378 filenames to filter to), the category map
from Task 1.1. Reuse the `ijson`-based streaming/filtering pattern from
`src_Alon/helper_files/import_tags.py` (lines ~106–121: `ijson.items(f, "item")`
filtered by `needed_names`).

**Implementation:** New script that: (1) lists filenames in
`config.TASK3_CLEAN_DIR`, (2) streams the JSON with `ijson`, keeping only
matching `name` entries, (3) for each kept image, iterates `labels[]`, keeps
only entries with a `box2d` and a `category` in
`config.BDD100K_TO_COCO_VEHICLE_CLASS`, (4) writes one CSV row per box:
`image, class, x1, y1, x2, y2`. Print a warning (like `import_tags.py` already
does) for any of the 378 filenames with zero matching annotations — this
surfaces a train/val-split mismatch immediately if one exists.

**Location:** `src_Alon/helper_files/extract_task3_gt.py` (this is the exact
filename `config.py`'s existing comment already names as the planned owner of
`TASK3_GT_CSV`).

**Output:** `data/gt_labels/task3_vehicle_gt.csv` (path already reserved as
`config.TASK3_GT_CSV`).

**Verification:** CSV exists, has a `class` column containing only
`{car, truck, bus, motorcycle, bicycle}`, and covers close to all 378 images
(report exact coverage in the script's final printed summary — if coverage is
low, that's the signal to also pull `..._val.json`).

**Status: ✅ Done (2026-07-16) — with two important pivots from the original
plan, decided live with the user:**
1. **No `bdd100k.zip`/v1 JSON available.** Instead used a locally-available
   **v2 detection labels** pair (`det_v2_train_release.json` / `..._val...`,
   at `C:\Users\amit\Downloads\archive\labels\`, ~352MB/51MB) — schema turned
   out directly compatible (`name`, `labels[].category`, `labels[].box2d.
   {x1,y1,x2,y2}`). `ijson` isn't installed and `pip install` is blocked by
   this machine's SSL/network setup, so the script uses plain `json.load()`
   instead (fine: ~11s for the 352MB train file). Added
   `BDD100K_V2_LABELS_DIR/TRAIN_JSON/VAL_JSON` to `config.py`; the old
   `BDD100K_ZIP_PATH`/v1 constants from 0.2 are kept but now marked legacy/
   unused.
2. **Task 3's image set was pruned from 675 → 300**, at the user's explicit
   request, to exactly the same 300 filenames as Task 1
   (`data/clean_images/1_data_lane_detection_low_level`) — verified as a
   full subset before pruning, 375 extra files deleted from
   `data/clean_images/3_data_vehicle_detection_deep_learnning` (recoverable
   from `Desktop\data` if ever needed back). This conveniently *also* solved
   the coverage problem: the original 675-image set only had 409/675 (60.6%)
   with real GT (the other 266 filenames look like BDD100K MOT/tracking
   frame-sequence names, e.g. `..-0000.jpg`, not in the 100k detection set at
   all) — the pruned 300-image set has **100% GT coverage** (all 300 matched
   in `train`, 0 needed from `val`).

Ran successfully: `data/gt_labels/task3_vehicle_gt.csv` — 3,605 boxes across
298/300 images (2 images legitimately have zero vehicles). Class breakdown:
car=3352, truck=208, bus=45, **motorcycle=0, bicycle=0** — worth flagging in
README's limitations later (Task 1.5's per-class plots will just be empty/
absent for those two classes on this particular image set, not a bug).

---

### 1.3 Rewire Task 3 driver to use real GT baseline

**Goal:** Replace the pseudo-GT baseline (clean-run detections treated as GT)
with the real GT CSV from Task 1.2, so `matched_recall`/`mean_iou_matched`
measure real accuracy, not self-consistency.

**Input:** `data/gt_labels/task3_vehicle_gt.csv` (Task 1.2's output),
`Amit_project/run_vehicle_detection_degradation.py`'s `run_baseline()`
function (currently returns `baseline_boxes` from `process_image` on clean
images).

**Implementation:** Load `TASK3_GT_CSV` via `pandas.read_csv`, group by
`image`, build the same `{filename: [box, ...]}` shape `run_baseline()`
already produces, and use it as `baseline_boxes` everywhere it's currently
passed. Keep `run_baseline()`'s clean-image detection pass — it's now useful in
its own right as "baseline performance vs. real GT" (Part 1 of the course
brief) rather than being the GT source itself. Report this as a new
`baseline_vs_gt_metrics.csv`.

**Location:** `Amit_project/run_vehicle_detection_degradation.py`.

**Output:** `outputs/task3_vehicle_detection/baseline_vs_gt_metrics.csv`
(new), plus the existing `degraded_per_image.csv`/`level_summary.csv` now
computed against real GT instead of pseudo-GT.

**Verification:** Baseline recall vs. GT is no longer trivially `1.0`;
`degraded_per_image.csv`'s `matched_recall` for the mildest distortion level is
close to (but not necessarily equal to) the new baseline value, and degrades
as level increases — sanity-check by eyeballing `level_summary.csv`.

**Status: ✅ Done (2026-07-16).** `run_baseline` renamed in effect to
`run_baseline_vs_gt` (real GT boxes loaded from `config.TASK3_GT_CSV` via
`load_gt_boxes()`, grouped by image). `baseline_vs_gt_metrics.csv` now
produced. Verified on the smoke sample: baseline recall vs. real GT came out
well below 1.0 (as expected for a real, imperfect detector), and
`degraded_per_image.csv`'s recall dropped further and monotonically-ish with
distortion level. Diffed directly against the Phase 0 pseudo-GT snapshot —
e.g. `low_light level 1`: pseudo-GT recall was 0.925, real-GT recall is
0.32 — confirms the rewiring is real, not just plumbing.

---

### 1.4 Add class-aware IoU matching

**Goal:** Prevent a car detection from being wrongly matched to a nearby GT
truck box just because their IoU is high — matching must respect class.

**Input:** `Amit_project/run_vehicle_detection_degradation.py`'s
`greedy_match()` function (currently class-agnostic).

**Implementation:** Add a `class` field alongside each box tuple (or pass
parallel class lists), and add `ref_class == cand_class` as a precondition
before computing IoU for a candidate pair in `greedy_match`.

**Location:** `Amit_project/run_vehicle_detection_degradation.py`.

**Output:** No new files — modifies matching behavior used by Tasks 1.3/1.5.

**Verification:** Unit-style manual check: construct two boxes with high IoU
but different classes, confirm `greedy_match` no longer pairs them.

**Status: ✅ Done (2026-07-16).** `greedy_match(ref_boxes, ref_classes,
cand_boxes, cand_classes, iou_thresh)` now requires `ref_classes[i] ==
cand_classes[j]` before considering a candidate pair. Confirmed indirectly
via the smoke run's per-class breakdown (1.5) — car/truck/bus matched counts
never cross-contaminate.

---

### 1.5 Add per-class metric aggregation and plots

**Goal:** Satisfy the course's explicit "measure performance per class, per
SNR" requirement for Task 3.

**Input:** `degraded_per_image.csv` (now has a `class` column per Task 1.4),
`Amit_project/run_vehicle_detection_degradation.py`'s `level_summary`
groupby and `plot_metric_vs_snr()`.

**Implementation:** Change the `groupby(["augmentation", "level"])` to
`groupby(["augmentation", "level", "class"])`. Update `plot_metric_vs_snr` to
draw one line per (distortion, class) combination, or produce one figure per
class — pick whichever stays readable with 3 distortions x 5 classes (likely:
one figure per class, saved as `recall_vs_snr_<class>.png`).

**Location:** `Amit_project/run_vehicle_detection_degradation.py`.

**Output:** `outputs/task3_vehicle_detection/level_summary_per_class.csv`,
`recall_vs_snr_<class>.png` and `iou_vs_snr_<class>.png` for each of the 5
vehicle classes.

**Verification:** 5 recall plots + 5 IoU plots exist, one per class; each
shows a monotonic-ish degradation trend distinct from the others (classes with
fewer/smaller boxes, e.g. bicycle, should show noisier/steeper curves than
car).

**Status: ✅ Done (2026-07-16), with one adjustment:** this 300-image subset
has zero `motorcycle`/`bicycle` GT instances (only car/truck/bus appear), so
the driver produces 3 per-class plots, not 5 — and explicitly prints a
`[skip] no GT instances of class '<x>' ...` message for the two absent
classes rather than emitting an empty/misleading plot.

**Follow-up fix during the full 300-image/9-level production run
(2026-07-16):** the original skip condition (`cls_summary.empty`) wasn't
enough — on the full dataset, YOLO produced a handful of false-positive
`motorcycle`/`bicycle` detections even though GT has zero real instances of
either, so those classes showed up as non-empty rows with **all-NaN**
recall/IoU, and got a degenerate blank plot instead of being skipped.
Fixed the condition to `cls_summary.empty or not cls_summary["matched_
recall"].notna().any()`. Manually deleted the 4 already-generated bad plot
files (`{recall,iou}_vs_snr_{motorcycle,bicycle}.png`) from the full run
that happened before this fix. `degraded_per_image.csv`
is now long-format with a `class` column (values: `car`/`truck`/`bus`/`all`);
`level_summary_per_class.csv` added alongside the existing (still-produced,
`class="all"`-filtered) `level_summary.csv` for backward-compatible aggregate
plots. Verified via smoke run: `recall_vs_snr_{car,truck,bus}.png` and
`iou_vs_snr_{car,truck,bus}.png` all generated.

---

### 1.6 Update README Task 3 methodology section

**Goal:** Replace the pseudo-GT caveat with the real-GT methodology description
once Tasks 1.2–1.5 are done.

**Input:** `README.md` sections 2 ("Pseudo-GT methodology"), 4 ("Metrics"), 11
("Known limitations").

**Implementation:** Edit prose only. State the real GT source
(`bdd100k_labels_images_train.json`, extracted via
`extract_task3_gt.py`), the category mapping and the `rider`-class exclusion
as the new documented limitation, and update the metrics section to describe
per-class Precision/Recall/F1/IoU.

**Location:** `README.md` (repo root).

**Output:** No new files — README edit only.

**Verification:** README no longer says "pseudo-GT" for Task 3; a fresh
reader can find the exact GT file and category-mapping decision without
opening code.

**Status: ✅ Done (2026-07-16).** Updated README sections 1 (dataset table:
300 images, same set as Task 1), 2 (GT column + new "Real-GT methodology"
paragraph replacing the old blanket pseudo-GT claim for Task 3), 4 (metrics:
per-class, real-GT framing, motorcycle/bicycle caveat), 5 (Part 1 baseline
row no longer says "recall = 1.0 by construction"), 11 (limitations:
pseudo-GT scoped to Task 2 only, added the 675→300 pruning note and the
zero-instance-classes note).

---

### 1.7 Phase 1 quick-check — Task 3 GT sanity pass

**Goal:** Confirm real-GT scoring is wired correctly before relying on it for
enhancement comparisons (Phase 2) or fine-tuning evaluation (Phase 5).

**Input:** `smoke_test/task3/` (from Task 0.3),
`smoke_test/results_task3_baseline/` (from Task 0.4, the pre-change snapshot).

**Implementation:** No new code — run the now-rewired driver on the smoke
sample: `python extract_task3_gt.py` scoped/filtered to `smoke_test/task3/`
filenames, then
`python run_vehicle_detection_degradation.py --clean_dir ../smoke_test/task3 --out_dir ../smoke_test/results_task3_gt --num_levels 2`.
Diff the new `level_summary_per_class.csv` against the Phase-0 baseline
snapshot's aggregate numbers.

**Location:** Run from `Amit_project/` / `src_Alon/helper_files/`.

**Output:** `smoke_test/results_task3_gt/` (baseline_vs_gt_metrics.csv,
degraded_per_image.csv with a `class` column, per-class plots).

**Verification:** (a) `baseline_vs_gt_metrics.csv` shows baseline recall
*not* trivially 1.0 (proves it's scoring against real GT, not the old
pseudo-GT); (b) `degraded_per_image.csv` has a populated `class` column with
only the 5 expected values; (c) at least one per-class plot renders without
error. If all three pass, Phase 1 is done and it's safe to start Phase 2.

**Status: ✅ Done (2026-07-16).** Ran on `smoke_test/task3` (regenerated
after the 675→300 prune) with `--num_levels 2`. (a) passed — baseline recall
0.32–0.38 range on the 6-image smoke sample, nowhere near the old pseudo-GT
1.0. (b) passed — `class` column has `car`/`truck`/`bus`/`all` (motorcycle/
bicycle absent from this data, as expected). (c) passed — 6 plots rendered
(`recall_vs_snr`/`iou_vs_snr` × {aggregate, car, truck, bus}). Direct diff
against the Phase 0 (`0.4`) pseudo-GT snapshot confirms materially different,
lower, more realistic numbers across every distortion/level.
**Phase 1 is done — safe to start Phase 2.** Note for Phase 1's own
prerequisite check: a **full 300-image, 9-level production run** (not just
the 6-image smoke sample) hasn't been done yet — that's deferred to whenever
we want real numbers for the README (Phase 6), not required to consider
Phase 1's wiring complete.

---

## Phase 2 — Enhancement wiring (Tasks 2 & 3)

### 2.1 Wire enhancement pass into Task 2 driver

**Goal:** Make the enhancement step reproducible from committed code (today
`enhancements.py` is imported nowhere, yet enhanced result files already exist
under `results/task2_feature_matching/enhanced_visualizations/` from an
untracked prior run).

**Input:** `enhancements.py`'s `restore_motion_blur`, `ENHANCEMENTS` dict,
`Amit_project/run_feature_matching_degradation.py`'s
`run_degraded_in_memory()`.

**Implementation:** Add a third pass alongside baseline/distorted: for each
augmented frame `a1/a2`, also compute `enhanced1/enhanced2 =
ENHANCEMENTS[aug_name](a1), ENHANCEMENTS[aug_name](a2)` and re-run
`process_pair` on the enhanced arrays. Tag rows with a `stage` column
(`distorted`/`enhanced`) instead of only `augmentation`/`level`, so the
existing CSV schema still works with one added column.

**Location:** `Amit_project/run_feature_matching_degradation.py` (import
`enhancements` — add `sys.path` entry if it's not already reachable from
`Amit_project/`, since `enhancements.py` lives at repo root).

**Output:** `outputs/task2_feature_matching/enhanced_per_pair.csv`.

**Verification:** New CSV exists with `stage="enhanced"` rows; running the
driver end-to-end with `--num_levels 2` on a small folder completes without
error and produces both distorted and enhanced rows for every pair/level.

**Status: ✅ Done (2026-07-16), one deviation from the original plan:**
implemented as **two separate DataFrames/CSVs** (`degraded_per_pair.csv` for
distorted, `enhanced_per_pair.csv` for enhanced) rather than a single CSV
with a `stage` column — simpler to reason about and matches what the
`Output` field of this task already specified. `run_degraded_in_memory` now
returns `(distorted_df, enhanced_df)`. Verified on `smoke_test/task2`
(`--num_levels 2`): both CSVs populated, enhancement measurably changed
`inlier_ratio` (low_light 0.893→0.937 improved; motion_blur 0.883→0.815 and
rain 0.861→0.842 slightly worse — plausible on a 5-pair sample, not
conclusive at this scale).

---

### 2.2 Wire enhancement pass into Task 3 driver

**Goal:** Same as Task 2.1, for vehicle detection.

**Input:** `enhancements.py`, `Amit_project/run_vehicle_detection_degradation.py`'s
`run_degraded_in_memory()`.

**Implementation:** Mirror Task 2.1's approach: after computing `aug_img`,
also compute `enh_img = ENHANCEMENTS[aug_name](aug_img)`, run `process_image`
on it, and IoU-match against the real GT (post-Task 1.3) the same way the
distorted pass already does. Add a `stage` column.

**Location:** `Amit_project/run_vehicle_detection_degradation.py`.

**Output:** `outputs/task3_vehicle_detection/enhanced_per_image.csv`.

**Verification:** New CSV exists with `stage="enhanced"` rows across all
distortions/levels/classes; spot-check that enhanced recall is generally
between distorted and clean-baseline for at least the low_light distortion
(matches the course's own example on slide 31).

**Status: ✅ Done (2026-07-16).** Same two-DataFrame pattern as 2.1:
`run_degraded_in_memory` now returns `(distorted_df, enhanced_df)`, both
per-class + "all" rows (reuses `per_class_and_overall_rows` from Phase 1).
Verified on `smoke_test/task3`: `enhanced_per_image.csv` (94 rows) alongside
`degraded_per_image.csv` (93 rows). Aggregate recall improved with
enhancement for low_light (0.193→0.269) and rain (0.273→0.301); motion_blur
roughly flat (0.248→0.243) — again a 6-image smoke sample, not conclusive,
just confirms the wiring behaves sensibly.

---

### 2.3 Generate enhancement comparison plots (Task 2)

**Goal:** The "distorted vs. enhanced vs. clean baseline" bar chart the course
brief's own example shows (slide 31), for Task 2.

**Input:** `enhanced_per_pair.csv` (Task 2.1), `baseline_per_pair.csv`.

**Implementation:** New plotting function `plot_enhancement_comparison()`:
group by `augmentation`, plot mean `inlier_ratio`/`match_ratio` for
distorted vs. enhanced as paired bars, clean baseline as a dashed reference
line — mirrors `run_feature_matching_degradation.py`'s existing
`plot_metric_vs_snr` style.

**Location:** `Amit_project/run_feature_matching_degradation.py`.

**Output:** `outputs/task2_feature_matching/inlier_ratio_per_distortion.png`,
`match_ratio_per_distortion.png` (these exact filenames are already named in
README section 7 as pending).

**Verification:** Both PNGs exist and open; enhanced bars are visually
distinguishable from distorted bars per distortion.

**Status: ✅ Done (2026-07-16).** `plot_enhancement_comparison()` added
(paired bar chart, distorted vs. enhanced, dashed clean-baseline line).
Verified both PNGs render on the smoke run.

---

### 2.4 Generate enhancement comparison plots (Task 3)

**Goal:** Same comparison chart for Task 3.

**Input:** `enhanced_per_image.csv` (Task 2.2).

**Implementation:** Same pattern as Task 2.3, using `matched_recall`/
`mean_iou_matched`.

**Location:** `Amit_project/run_vehicle_detection_degradation.py`.

**Output:** `outputs/task3_vehicle_detection/recall_per_distortion.png`,
`iou_per_distortion.png` (also already named in README section 7).

**Verification:** Both PNGs exist and open; values are consistent with
`enhanced_per_image.csv`'s aggregates.

**Status: ✅ Done (2026-07-16).** Same `plot_enhancement_comparison()`
pattern, computed on `class=="all"` rows only (per-class enhancement
comparison bars were considered but left out of scope for now — the per-SNR
per-class plots from Phase 1 already cover the class dimension). Verified
both PNGs render on the smoke run.

---

### 2.5 Phase 2 quick-check

**Goal:** Confirm enhancement wiring works end-to-end for both Task 2 and
Task 3 before relying on it for the fine-tuning comparison (Phase 5) or the
README (Phase 6).

**Input:** `smoke_test/task2/`, `smoke_test/task3/` (post Phase 0/1).

**Implementation:** Run both rewired drivers on the smoke sample with
`--num_levels 2`, exactly as done for every previous quick-check.

**Location:** Run from `Amit_project/`.

**Output:** `smoke_test/results_task2_enh/`, `smoke_test/results_task3_enh/`.

**Verification:** (a) both `enhanced_per_pair.csv`/`enhanced_per_image.csv`
exist and are non-empty; (b) both `*_per_distortion.png` comparison plots
exist for both tasks; (c) enhanced-stage numbers differ from distorted-stage
numbers (proves the enhancement function is actually being applied, not a
no-op).

**Status: ✅ Done (2026-07-16).** All three checks passed. (a) `enhanced_
per_pair.csv` (Task 2, non-empty) and `enhanced_per_image.csv` (94 rows,
Task 3) both exist. (b) `inlier_ratio_per_distortion.png`/`match_ratio_per_
distortion.png` (Task 2) and `recall_per_distortion.png`/`iou_per_
distortion.png` (Task 3) all render. (c) confirmed different from distorted:
Task 2 inlier_ratio moved from 0.893/0.883/0.861 (distorted, low_light/
motion_blur/rain) to 0.937/0.815/0.842 (enhanced); Task 3 recall moved from
0.193/0.248/0.273 to 0.269/0.243/0.301. **Phase 2 is done — safe to start
Phase 3.**

**Addendum — enhancement-parameter investigation (2026-07-16, after the
first full production run flagged illogical-looking results by the user):**
On the full 178-pair/300-image data, two things looked wrong: (1)
`match_ratio` *increased* as SNR dropped under `motion_blur`; (2) enhanced
`match_ratio`/`inlier_ratio` was sometimes *lower* than distorted for
`low_light`/`motion_blur`. Root cause (not a code bug): `match_ratio =
n_good_matches / min(keypoints)` is a ratio, and both the raw distortion and
the enhancement functions change the *denominator* (keypoint count) a lot —
heavy blur collapses keypoint yield faster than match count, inflating the
ratio; CLAHE/unsharp-mask enhancement inflates keypoint yield with mostly
spurious high-frequency noise, deflating the ratio even when absolute
matches improve. Verified with real numbers: enhancement raised
`n_good_matches` for `low_light` (122.9→144.5, full run) and `motion_blur`
(125.1→146.6), but keypoint counts grew even more (low_light 1433→1986,
motion_blur 1418→1742), dragging `match_ratio` down anyway.
**Fix applied:** softened `enhancements.py`'s `restore_motion_blur` (unsharp
weights `1.5/-0.5`→`1.2/-0.2`) and `restore_low_light` (CLAHE `clipLimit
4.0`→`2.0`, tiles `8×8`→`16×16`). Re-ran both full production runs (Task 2 &
3) with the fix. Result: `motion_blur`'s keypoint inflation roughly halved
(1418→1570) and its `match_ratio` gap narrowed substantially; `low_light`
improved partially (`match_ratio` still lower post-enhancement, but less
so). **Tried softening further** (CLAHE `clipLimit=1.5`, tiles `24×24`) on
the smoke sample and confirmed it makes things *worse* — the enhancement
becomes too weak to recover real matches, so `n_good_matches` itself drops
(870.8→743.2 on the smoke sample) rather than just reducing keypoint
inflation. **Reverted to `clipLimit=2.0`/`16×16`** as the best point found.
Task 3 was unaffected by this whole investigation — its metrics
(`matched_recall`, `mean_iou_matched`) aren't keypoint-count-normalized, so
they don't have this failure mode; the user independently confirmed Task 3's
results already looked reasonable throughout.

**Second addendum — real root-cause fix for match_ratio's monotonicity
(2026-07-16, at the user's insistence the softened-enhancement patch above
wasn't enough):** the user correctly rejected "read a different metric" as
the final answer and asked for the actual `match_ratio` chart to behave
sensibly. Root cause: raw `match_ratio` normalizes by the *current* (possibly
distorted/enhanced) image's own keypoint count, which is itself unstable —
not fundamentally fixable by tuning enhancement strength alone. **Real fix:**
added `match_ratio_vs_baseline` — same numerator (`n_good_matches`), but
normalized by the **clean-baseline's fixed keypoint count** for that exact
pair (computed once in `run_baseline`, looked up by `(img1, img2)` in
`run_degraded_in_memory` for every distortion/level/enhancement pass).
Applied to `Amit_project/run_feature_matching_degradation.py`: `run_baseline`
now returns a third value (`baseline_min_kp` dict); `run_degraded_in_memory`
takes it as a new parameter and adds `match_ratio_vs_baseline` to both
`distorted_rows`/`enhanced_rows`; `level_summary.csv` includes it; both
`match_ratio_vs_snr.png` and `match_ratio_per_distortion.png` now plot it
instead of raw `match_ratio` (raw `match_ratio` is kept in the per-pair CSVs
for transparency, just not plotted as the primary metric anymore).
**Didn't require a full re-run** — derived `match_ratio_vs_baseline` directly
from the already-computed full-run CSVs (`baseline_per_pair.csv`'s keypoint
counts joined onto `degraded_per_pair.csv`/`enhanced_per_pair.csv` by
`(img1, img2)`), then regenerated just the level_summary and the two plots.
**Verified: fully resolved.** `match_ratio_vs_baseline` now decreases
monotonically with SNR for all 3 distortions (e.g. `motion_blur`:
0.085→0.030 from level 1→9; `low_light`: 0.084→0.0 at extreme darkness), and
enhanced ≥ distorted for all 3 distortions in the per-distortion comparison
(low_light 0.061→0.072, motion_blur 0.063→0.068, rain 0.071→0.073).
Documented in README section 4.

---

## Phase 3 — Task 1 migration

### 3.1 Build in-memory Task 1 driver

**Goal:** Give lane detection a real driver matching the Task 2/3 shape
(single CLI script, in-memory augmentation, `pandas` CSV output,
`Agg`-backend plots) instead of the current disk-based two-script split.

**Input:** `src_Alon/lane_detection/lane_detection_pipeline.py`'s
`process_image`, `Amit_project/augmentation_levels.py`'s
`make_augmentation_fns`/`compute_snr_db` (reuse directly — same 9-level
ladders should apply to Task 1 too, for cross-task comparability),
`Amit_project/run_feature_matching_degradation.py` as the structural template.

**Implementation:** New script: (1) baseline pass — `process_image` on every
clean frame, cache result; (2) in-memory sweep — for each
augmentation/level, apply the augmentation function to the clean array
in RAM (no disk write), re-run `process_image` on the array (note:
`process_image` currently only accepts a file path — needs a small signature
change to accept an optional `image_override` array, same pattern
`process_pair`/`process_image` in Tasks 2/3 already use); (3) SNR via
`compute_snr_db`; (4) CSV + `level_summary` + `plot_metric_vs_snr`-style
plots.

**Location:** New file `src_Alon/lane_detection/run_lane_detection_degradation.py`.
Requires a small edit to `src_Alon/lane_detection/lane_detection_pipeline.py`'s
`process_image` signature to add `image_override=None` (mirroring
`vehicle_detection_bdd100k.py`'s `process_image`).

**Output:** `outputs/task1_lane_detection/baseline_per_image.csv`,
`degraded_per_image.csv`, `level_summary.csv`,
`lane_offset_vs_snr.png`.

**Verification:** Running the new driver end-to-end reproduces
comparable per-image numbers to the old
`compere_gt_to_aug.py`/`lane_comparison_results.csv` pipeline on a handful of
shared images (sanity cross-check, not required to match exactly since the
metric may also change per Task 3.2).

**Status: ✅ Done (2026-07-16), plus 3.3 folded in and one unrelated bug
fixed.** New `src_Alon/lane_detection/run_lane_detection_degradation.py`,
targeting the unified `config.TASK1_CSV_DIR`/`GRAPH_DIR`/`VIS_DIR` layout
directly (covers 4.3 too). `lane_detection_pipeline.process_image` gained
`image_override=None`. Enhancement wiring (3.3) was cheap to include from
the start rather than as a separate pass, so it's already in: every driver
run produces `(distorted_df, enhanced_df)` plus a distorted-vs-enhanced-vs-
clean comparison plot, matching Tasks 2/3's pattern. **Unrelated bug found
and fixed while testing:** `filter_lines()` unpacked `cv2.HoughLinesP`'s
output assuming a `(N,1,4)` shape; this OpenCV build (`opencv-python`
5.0.0.93) returns `(N,4)`, which crashed with `TypeError: cannot unpack
non-iterable numpy.int32 object`. Fixed with `np.ravel(line)`, which handles
either shape — this bug pre-dates this session and would have blocked
*any* attempt to actually run `lane_detection_pipeline.py`, not something
introduced by this refactor. Verified on `smoke_test/task1` (6 images,
`--num_levels 2`): all expected CSVs/plots/visualizations produced.

---

### 3.2 Wire real BDD100K lane GT into Task 1 metric

**Goal:** Score lane detection against real BDD100K lane polygon annotations
instead of "augmented vs. clean-detection" pseudo-GT — this is the change that
actually closes the course's "dataset with GT for ≥1 task" requirement via
Task 1 (alternative/complement to closing it via Task 3).

**Input:** `src_Alon/helper_files/filter_bdd100k.py`'s existing
`get_gt_lane_lines()` function (already extracts real ego-lane GT lines from
BDD100K's `poly2d` lane labels) and `lines_match()`. Currently only used to
curate a "good match" image subset, not as the ongoing evaluation reference.

**Implementation:** In the new Task 1 driver (Task 3.1), for every clean image
also extract its real GT lane lines via `get_gt_lane_lines()` (needs the same
BDD100K JSON access pattern as Task 1.2 — reuse/share that streaming lookup
rather than re-reading the whole JSON per image). Replace the
`baseline vs. augmented` diff in `compere_gt_to_aug.py`'s `line_diff()` logic
with `detected vs. GT` diff, for both baseline and every augmented/enhanced
pass.

**Location:** `src_Alon/lane_detection/run_lane_detection_degradation.py`
(from Task 3.1), reusing `src_Alon/helper_files/filter_bdd100k.py`'s GT-parsing
functions (may need to extract `get_gt_lane_lines`/`fit_line_from_points`
into a small shared module both scripts import, to avoid duplicating them).

**Output:** Same CSVs as Task 3.1, now with `left/right_diff_px` measured
against real GT rather than clean-image detection.

**Verification:** Baseline (clean-image) accuracy vs. real GT is no longer
trivially perfect — some images will have nonzero offset even at level 0,
which is the expected, correct behavior for a real-GT comparison.

**Status: ❌ Blocked — deferred at the user's explicit decision (2026-07-16).**
Investigated before starting: BDD100K ships lane-marking annotations as a
**separate label release** from the object-detection labels (`det_v2_
{train,val}_release.json`) we have locally — those files' categories are
exactly `{bicycle, bus, car, motorcycle, other person, other vehicle,
pedestrian, rider, traffic light, traffic sign, train, truck}`, no `lane`
category at all. Searched `Downloads/`, `Desktop/` for any lane-marking
label file (JSON, mask, or otherwise) — none found.
`src_Alon/helper_files/filter_bdd100k.py`'s `get_gt_lane_lines()` (which
*would* do this) was written against the old v1 combined-label format we
also don't have. **Decision:** keep Task 1 on pseudo-GT (clean-frame
detection as reference) for now, matching Task 2 and the legacy
`compere_gt_to_aug.py`'s existing approach — this is what 3.1's driver
already implements. Task 1 is documented in README as pseudo-GT; the
course's "dataset with GT for ≥1 task" requirement is satisfied via Task 3
instead. **To revisit:** if the real lane-label file is ever sourced,
`filter_bdd100k.py`'s existing `get_gt_lane_lines()`/`fit_line_from_points()`
are the starting point — factor them into a shared module and swap the
`baseline_results` reference in `run_lane_detection_degradation.py` for
real GT, same pattern as Task 3's 1.3.

---

### 3.3 Add enhancement step to Task 1

**Goal:** Close README's own "TODO (Alon): apply an equivalent enhancement
step for Task 1" gap.

**Input:** `enhancements.py`'s existing 3 restoration functions (already
generic BGR-image functions, not task-specific — no new restoration logic
needed).

**Implementation:** Same pattern as Tasks 2.1/2.2: in the Task 1 driver, for
each augmented frame also compute the enhanced version and re-run
`process_image` on it, tagged with a `stage` column.

**Location:** `src_Alon/lane_detection/run_lane_detection_degradation.py`.

**Output:** Adds `stage="enhanced"` rows to the same CSVs from Task 3.1/3.2,
plus `outputs/task1_lane_detection/lane_offset_per_distortion.png`.

**Verification:** Enhanced-stage rows exist in the CSV; plot shows
distorted-vs-enhanced-vs-clean comparison, consistent with Tasks 2/3's
equivalent charts.

**Status: ✅ Done (2026-07-16) — folded into 3.1** (see its status note).
No separate work needed; verified together with 3.1's smoke test.

---

### 3.4 Retire Task 1 disk-based augmentation scripts

**Goal:** Remove the now-superseded legacy path once Tasks 3.1–3.3 are
verified working, per `config.py`'s own "Task 8" migration note.

**Input:** `src_Alon/create_aug/create_augmented_images.py`,
`src_Alon/metrics_comparison/compere_gt_to_aug.py`,
`src_Alon/metrics_comparison/threshold_survival.py`, `config.py`'s
`AUGMENTED_ROOT`/`SNR_LOG_CSV`/`LANE_COMPARISON_CSV` legacy block.

**Implementation:** Only after Tasks 3.1–3.3 are confirmed producing equal-or-
better results: delete the 3 legacy scripts (or move to an `archive/`
subfolder if Alon wants to keep them for reference), remove the now-dead
`AUGMENTED_ROOT` disk-based augmented-image folder, remove the "LEGACY"
comment block in `config.py` and its `SNR_LOG_CSV`/`LANE_COMPARISON_CSV`
constants once nothing references them.

**Location:** `src_Alon/create_aug/`, `src_Alon/metrics_comparison/`,
`config.py`.

**Output:** No new output files — this is a deletion/cleanup task.

**Verification:** `grep -rn "AUGMENTED_ROOT\|SNR_LOG_CSV\|LANE_COMPARISON_CSV"`
across the repo returns nothing outside `config.py`'s own (now-removed)
definitions. New driver from Task 3.1 is the only remaining way to reproduce
Task 1 results.

**Status: ✅ Done (2026-07-16), with one adjustment: archived, not deleted.**
Moved (via `git mv`, registered as renames) `create_augmented_images.py`,
`compere_gt_to_aug.py`, `threshold_survival.py` into new `src_Alon/archive/`,
with a short `archive/README.md` explaining why and what superseded them.
Chose archiving over deletion since these are Alon's working code, not dead
code — reversible and keeps them available for reference. `config.py`'s
`AUGMENTED_ROOT`/`SNR_LOG_CSV`/`LANE_COMPARISON_CSV` constants were **kept**
(not removed) since the archived scripts still import them and should stay
runnable if anyone wants to consult them — comments updated to point at the
new archive location instead of claiming removal is imminent.

---

### 3.5 Update README Task 1 section

**Goal:** Fill in Task 1's `TODO` markers now that method, metric, and GT
status are all concrete.

**Input:** `README.md` sections 1 ("Dataset" table row for Task 1), 2 ("Tasks,
Levels & Methods" table row 1), 5 ("Part 1 baseline" table).

**Implementation:** Edit prose/tables only, describing the Hough-line method,
the real-GT-based offset metric (post-Task 3.2), and linking to the new CSVs/
plots.

**Location:** `README.md` (repo root).

**Output:** No new files — README edit only.

**Verification:** No `TODO (Alon)` markers remain in README sections 1–2 for
Task 1.

**Status: ✅ Done (2026-07-16).** Updated sections 1 (resolved the
GT-availability question explicitly, rather than leaving it as an open
TODO), 2 (method + pseudo-GT methodology, matching Task 2's framing), 4
(survival_rate/diff_px metric definitions), 5 (Part 1 baseline row), 6/7
(plot filenames, enhancement no longer a TODO), 9 (repo structure — also
updated to reflect the actual `Amit_project/`/`src_Alon/` layout and new
`archive/` folder, which had drifted from reality even for Tasks 2/3), 10
(added the Task 1 run command), 11 (pseudo-GT limitation now covers Task 1
too, with the lane-label-unavailability explanation), 12 (checklist item
narrowed since lane-detection method/metric/GT are no longer TODO).

---

### 3.6 Phase 3 quick-check

**Goal:** Confirm the new Task 1 driver produces sane, non-degenerate output
before treating the old disk-based scripts as fully retired.

**Input:** `smoke_test/task1/` (post Phase 0).

**Implementation:** Run the new driver on the smoke sample with
`--num_levels 2`.

**Location:** Run from `src_Alon/lane_detection/`.

**Output:** `smoke_test/results_task1_{csv,graph,vis}/`.

**Verification:** (a) `baseline_per_image.csv`, `degraded_per_image.csv`,
`enhanced_per_image.csv`, `level_summary.csv` all exist and are non-empty;
(b) `lane_offset_vs_snr.png`/`lane_offset_per_distortion.png` render; (c)
`survival_rate` is not uniformly 1.0 or uniformly 0.0 across levels (proves
the metric is actually sensitive to distortion severity, not degenerate).

**Status: ✅ Done (2026-07-16).** All three checks passed on the smoke run
(6 images, `--num_levels 2`): survival_rate ranged from 0.0 (low_light
level 2 — frame nearly black, both lines lost, as expected) to 1.0
(low_light/rain level 1), with a sensible in-between (0.83, 0.92) at other
levels — a real, non-degenerate signal. All 4 CSVs and both plots
generated; visualization folders populated for baseline/degraded/enhanced.
**Phase 3 is done (with 3.2 explicitly deferred) — safe to start Phase 4.**

---

## Phase 4 — Unified output layout

### 4.1 Migrate Task 2 output to unified outputs/ layout

**Goal:** Move Task 2 from the legacy flat `outputs/task2_feature_matching/`
to the unified `outputs/csv_results/task2/`, `outputs/graph_results/task2/`,
`outputs/visualizations/task2/` layout `config.py` already declares
(`TASK2_CSV_DIR`, `TASK2_GRAPH_DIR`, `TASK2_VIS_DIR`).

**Input:** `Amit_project/run_feature_matching_degradation.py`'s
`--out_dir` default (`config.TASK2_RESULTS_DIR`, legacy).

**Implementation:** Change the driver to write CSVs to `config.TASK2_CSV_DIR`,
plots to `config.TASK2_GRAPH_DIR`, and `baseline_visualizations`/
`degraded_visualizations`/`enhanced_visualizations` subfolders under
`config.TASK2_VIS_DIR`. Keep `--out_dir` as an optional override for ad-hoc
runs, but change the *default*.

**Location:** `Amit_project/run_feature_matching_degradation.py`.

**Output:** `outputs/csv_results/task2/*.csv`, `outputs/graph_results/task2/*.png`,
`outputs/visualizations/task2/{baseline,degraded,enhanced}_visualizations/`.

**Verification:** Fresh run populates the unified dirs; `outputs/task2_feature_matching/`
(legacy) is no longer written to by a default-args run.

**Status: ✅ Done (2026-07-16), backward-compatible.** `--out_dir` kept but
its default changed to `None`: when omitted, output splits across
`config.TASK2_CSV_DIR`/`GRAPH_DIR`/`VIS_DIR` (the unified layout); when
explicitly given (old behavior), all output goes into that single flat
folder exactly as before — so no previously-documented/-run command breaks.
`run_baseline`/`run_degraded_in_memory`'s `out_dir` param renamed to
`vis_root` (visualization-only now, CSVs/plots handled separately in
`main()`). Verified both modes: default run → `outputs/{csv_results,
graph_results,visualizations}/task2/`; `--out_dir <path>` → single flat
folder, unchanged.

---

### 4.2 Migrate Task 3 output to unified outputs/ layout

**Goal:** Same as Task 4.1, for Task 3.

**Input:** `Amit_project/run_vehicle_detection_degradation.py`'s `--out_dir`
default (`config.TASK3_RESULTS_DIR`, legacy).

**Implementation:** Same pattern as Task 4.1, targeting `config.TASK3_CSV_DIR`/
`TASK3_GRAPH_DIR`/`TASK3_VIS_DIR`.

**Location:** `Amit_project/run_vehicle_detection_degradation.py`.

**Output:** `outputs/csv_results/task3/*.csv`, `outputs/graph_results/task3/*.png`,
`outputs/visualizations/task3/...`.

**Verification:** Same as Task 4.1, for Task 3's directories.

**Status: ✅ Done (2026-07-16).** Same pattern as 4.1. Verified both modes
work identically for Task 3.

---

### 4.3 Migrate Task 1 output to unified outputs/ layout

**Goal:** Make sure the new Task 1 driver (Task 3.1) targets the unified
layout from the start rather than needing a later migration.

**Input:** `config.TASK1_CSV_DIR`/`TASK1_GRAPH_DIR`/`TASK1_VIS_DIR` (already
defined in `config.py`, currently unused since Task 1 has never had a real
driver).

**Implementation:** If Task 3.1 is implemented after this task, just point
its `--out_dir` defaults here directly. If Task 3.1 already shipped with
different defaults, update them.

**Location:** `src_Alon/lane_detection/run_lane_detection_degradation.py`.

**Output:** `outputs/csv_results/task1/*.csv`, `outputs/graph_results/task1/*.png`,
`outputs/visualizations/task1/...`.

**Verification:** Same pattern as Tasks 4.1–4.2.

**Status: ✅ Done — already satisfied by Task 3.1** (built targeting the
unified layout from the start, since there was no legacy single-`--out_dir`
Task 1 driver to stay backward-compatible with).

---

### 4.4 Clean up legacy/duplicate output directories

**Goal:** Remove the now-redundant `outputs/task2_feature_matching/`,
`outputs/task3_vehicle_detection/` (legacy flat dirs) and the untracked
top-level `results/` tree, once Tasks 4.1–4.3 confirm the unified layout has
everything needed for the README.

**Input:** `outputs/task2_feature_matching/`, `outputs/task3_vehicle_detection/`,
`results/` (git-untracked, contains files not reproducible from committed code
per the gap analysis — review before deleting in case something in there is
still the only copy of a needed plot).

**Implementation:** Confirm every file the README references (section 6–7's
plot filenames) exists in the new unified location, then delete the legacy
directories. Remove the now-dead `TASK2_RESULTS_DIR`/`TASK3_RESULTS_DIR`
legacy block from `config.py`.

**Location:** Repository root (directory deletion), `config.py` (constant
cleanup).

**Output:** No new output files — this is a deletion/cleanup task.

**Verification:** `git status` shows the legacy dirs removed;
`grep -rn "TASK2_RESULTS_DIR\|TASK3_RESULTS_DIR"` returns nothing outside
`config.py`'s own removed definitions; README's linked plot paths all resolve
under the unified `outputs/` layout.

**Status: ✅ Done (2026-07-16) — after a full production run.** Initially
deferred (see above) since only smoke-scale data existed in the new
location. The user then requested a full 300/178/300-image, 9-level
production run for all three tasks (done — see Phase 4's addendum below);
once the unified `outputs/csv_results|graph_results|visualizations/task{2,3}/`
had real, complete data confirmed (178-row baseline, ~4,450-4,800 degraded/
enhanced rows for Task 2; 300-image baseline, 13,788-row degraded for
Task 3), deleted `outputs/task2_feature_matching/` and `outputs/
task3_vehicle_detection/` for real. Also removed `TASK2_RESULTS_DIR`/
`TASK3_RESULTS_DIR` from `config.py` (constants + their `.mkdir()` calls +
their lines in the `__main__` sanity-check block) since nothing references
them anymore — confirmed via `grep -rn "TASK2_RESULTS_DIR\|TASK3_RESULTS_DIR" --include=*.py .`
returning zero hits. `python config.py` still runs cleanly.
**Note:** the separate top-level `results/` directory (git-untracked, an
even older ad-hoc run predating this session) was *not* touched — it's a
distinct legacy location from the two removed here, left for a future
decision since it wasn't explicitly in scope of this confirmation.

---

### 4.5 Phase 4 quick-check

**Goal:** Confirm both drivers default into the unified layout, confirm the
old `--out_dir` override still works unchanged, and confirm the legacy
directories didn't get overwritten by this phase's own testing.

**Input:** `smoke_test/task2/`, `smoke_test/task3/`.

**Implementation:** Run both drivers twice each: once with no `--out_dir`
(new default), once with an explicit `--out_dir` pointing at a smoke-test
folder (legacy-mode check).

**Location:** Run from `Amit_project/`.

**Output:** `outputs/csv_results|graph_results|visualizations/task{2,3}/`
(new default runs), `smoke_test/results_task2_legacy_check/` (legacy-mode
check).

**Verification:** (a) default run lands under
`outputs/{csv_results,graph_results,visualizations}/task{2,3}/`; (b)
`--out_dir <path>` run puts everything in that single flat folder, as
before; (c) `outputs/task2_feature_matching/`/`task3_vehicle_detection/`
(legacy) row counts are unchanged from before this phase (proves the new
default didn't clobber the old full-run data).

**Status: ✅ Done (2026-07-16).** (a)/(b) both verified directly (see 4.1/4.2
status notes). (c) verified — legacy CSVs still show 178/890/2269 rows,
matching pre-Phase-4 state; only `outputs/csv_results/task2/` (new,
previously empty) received the smoke-scale test data.

**Addendum — full production run (2026-07-16, at the user's request):** ran
all three drivers at full scale (Task 1: 300 images, Task 2: 178 pairs,
Task 3: 300 images, all `--num_levels 9`) in the background, into the
unified default layout. All three completed successfully:
- Task 1: 300 baseline images, 8,100 degraded rows (300×3×9); baseline
  left/right line detection rate 98.7%/99.3%.
- Task 2: 178 baseline pairs; baseline `inlier_ratio`=0.553,
  `match_ratio`=0.083.
- Task 3: 300 baseline images, 13,788 degraded rows; baseline
  `matched_recall` (vs. real GT)=0.23, `mean_iou_matched`=0.831.
  **⚠️ Superseded — see the "Data-quality issue found and fixed" note right
  before Phase 5: 105/300 of these images were corrupt, dragging this number
  down artificially. Corrected baseline recall (same images, fixed) = 0.366.**

**Bug found and fixed during this run:** on the full dataset, YOLO produced
a handful of false-positive `motorcycle`/`bicycle` detections (GT has zero
real instances of either across all 300 images), which the per-class-plot
skip condition didn't catch (see 1.5's status note for the fix and the 4
stale plot files removed). This confirms Phase 4 is done with real,
complete data — which then unblocked 4.4 (see its updated status above).
**Phase 4 is now fully done — safe to start Phase 5.**

---

## ⚠️ Data-quality issue found and fixed (2026-07-17, during Phase 5 setup)

While building the fine-tuning dataset (5.1/5.2), `cv2.imread` silently
returned `None` for many Task 3 images. Investigated: **105 of the 300
images in `data/clean_images/3_data_vehicle_detection_deep_learnning/` were
corrupt at the byte level** (all-zero content, valid file size, not a real
JPEG at all — confirmed with `PIL.Image.verify()` too) — and this was true
of the **original source** (`Desktop\data`) as well, not something this
session's copy operations broke. Checked Task 1's and Task 2's source
folders too: **zero corruption in either** (300/300 and 180/180 clean).

**Impact:** every Task 3 full-run result reported so far (Phase 1's GT
integration, Phase 2's enhancement comparison, Phase 4's full production
run) was computed with ~35% of images silently contributing zero detections
because they couldn't be read at all — not because the detector missed
real vehicles. This likely explains a meaningful chunk of Task 3's
surprisingly low baseline recall (0.23). GT extraction itself
(`extract_task3_gt.py`) was unaffected — it never reads the images, only
the labels JSON.

**Fix:** found valid replacements for all 105 corrupt filenames in
`C:\Users\amit\Downloads\archive\bdd100k\bdd100k\images\100k\train\`
(the full BDD100K 100k-image set), verified each loads correctly via
`cv2.imread` before copying, then replaced all 105 files in
`data/clean_images/3_data_vehicle_detection_deep_learnning/`. Confirmed
zero corruption remains. **At the user's request, re-ran Task 3's full
production run** (`run_vehicle_detection_degradation.py --num_levels 9`)
with the fixed images before proceeding to fine-tuning — see its own status
note for the corrected numbers. Task 1/2 did not need re-running (their
source data was never corrupt).

---

## Phase 5 — Fine-tuning (Task 3, Part 4)

### 5.1 Generate pseudo-labels from clean-image baseline

**Goal:** Produce YOLO-format label files from the clean-image baseline
detections, to use as (pseudo-)training labels for fine-tuning — same recipe
the course brief's own example uses (slide 33).

**Input:** Real GT from Task 1.2 is preferable here if available and
sufficiently covers the dataset (use it instead of pseudo-labels where
present); otherwise fall back to clean-image YOLOv8n detections
(`Amit_project/vehicle_detection_bdd100k.py`'s `process_image` output).

**Implementation:** For each image, write a YOLO-format `.txt` label
(`class_id cx cy w h`, normalized) — reuse the exact conversion formula
already implemented in `src_Alon/helper_files/import_tags.py` (lines
141–203) and `filter_bdd100k.py`-adjacent slide-33 pseudocode
(`save_yolo_label`).

**Location:** New script `Amit_project/generate_pseudo_labels.py`.

**Output:** `data/finetune_dataset/labels/train/*.txt` (+ matching
`images/train/*.jpg` copies).

**Verification:** Label file count matches image count; spot-check a handful
of `.txt` files against the corresponding image's known detections.

**Status: ✅ Done (2026-07-17), combined with 5.2 into one script.** Used
real GT (`config.TASK3_GT_CSV`) directly rather than clean-image pseudo-
labels, per the task's own stated preference — no detection run needed,
just a straight box2d→YOLO-txt conversion. Implemented in
`Amit_project/build_finetune_dataset.py`.

---

### 5.2 Build distorted fine-tuning training set

**Goal:** Create the actual distorted-image training set the fine-tuned model
will train on, paired with Task 5.1's labels.

**Input:** `Amit_project/augmentation_levels.py`'s `make_augmentation_fns`,
Task 5.1's labels, clean images.

**Implementation:** For each training image, apply one randomly-chosen
distortion/level combination (or a fixed moderate level per the course's
"small scale" allowance on slide 37) and save the distorted image (this one
genuinely needs to hit disk, since YOLO's `.train()` reads from a directory,
unlike the in-memory evaluation sweeps) alongside its unchanged label file.

**Location:** `Amit_project/generate_pseudo_labels.py` (extend from Task 5.1)
or a new `Amit_project/build_finetune_dataset.py`.

**Output:** `data/finetune_dataset/images/train/*.jpg` (distorted),
`data/finetune_dataset/data.yaml` (YOLO dataset config).

**Verification:** `data.yaml` + folder structure is loadable by
`ultralytics.YOLO(...).train(data=...)` without error (dry run with
`epochs=1`).

**Status: ✅ Done (2026-07-17).** Fixed moderate level (5/9), round-robin
across the 3 distortions so the training set has roughly equal
representation of each. 80/20 train/val split (seeded shuffle, 240/60
images). **First build hit the data-corruption issue** (see the dedicated
note above Phase 5) — 105/300 images silently failed to read; rebuilt after
the fix, all 300 processed cleanly. `data.yaml` verified loadable by
`ultralytics.YOLO(...).train()` (see 5.3's 1-epoch smoke test).

---

### 5.3 Fine-tune YOLOv8n on distorted images

**Goal:** Produce fine-tuned weights, per course requirement "at least one DL
model" fine-tuned on distorted images.

**Input:** `data/finetune_dataset/data.yaml` (Task 5.2),
`Amit_project/yolov8n.pt` (starting weights).

**Implementation:** `YOLO("yolov8n.pt").train(data=..., imgsz=640, epochs=<a
few>, batch=<small>, device="cpu"|"cuda")` — mirror the course brief's own
minimal example (slide 34: `epochs=3, batch=2, device="cpu"`), scaled up if
GPU is available.

**Location:** New script `Amit_project/finetune_vehicle_detection.py`.

**Output:** `Amit_project/runs/detect/train/weights/best.pt` (ultralytics'
own convention) — copy/rename to `Amit_project/yolov8n_finetuned.pt` for a
stable path other scripts can reference.

**Verification:** Training completes without error; `best.pt` file exists and
loads via `YOLO(str(best_path))`.

**Status: ✅ Done (2026-07-17).** `Amit_project/finetune_vehicle_detection.py`
created, mirrors the course brief's minimal recipe but scaled to our dataset
size (`epochs=15, batch=8, imgsz=640, device="cpu"`). 1-epoch smoke test run
first (~2.5 min, confirmed wiring + gave a rough per-epoch time estimate),
then the full 15-epoch run (~31 min real time on this CPU). Final
validation: `mAP50=0.222`, `mAP50-95=0.142`, precision=0.369, recall=0.186
(ultralytics' own val-split metrics, not directly comparable to our
`matched_recall` — different confidence threshold/computation). Weights
copied to `Amit_project/yolov8n_finetuned.pt`; verified it loads via
`YOLO(...)` with the expected 5 class names (`car`/`truck`/`bus`/
`motorcycle`/`bicycle`) matching `vehicle_detection_bdd100k.py`'s
`VEHICLE_CLASSES` filter (works unmodified since that filter matches by
class *name*, not numeric id).

---

### 5.4 Re-run distortion sweep with fine-tuned weights and compare

**Goal:** Produce the final Part 4 comparison: pretrained-distorted vs.
pretrained-enhanced vs. fine-tuned-distorted.

**Input:** `Amit_project/yolov8n_finetuned.pt` (Task 5.3),
`Amit_project/run_vehicle_detection_degradation.py` (post Phase 1/2/4
rewiring).

**Implementation:** Run the existing (now GT-based, enhancement-aware)
driver with `--model yolov8n_finetuned.pt`, writing to a distinct
`--out_dir`/model-tagged subfolder. Build one final comparison plot
overlaying: pretrained+distorted, pretrained+enhanced, fine-tuned+distorted
recall/IoU curves.

**Location:** `Amit_project/run_vehicle_detection_degradation.py` (reuse,
parameterized by `--model`), new small script
`Amit_project/compare_finetuned_vs_pretrained.py` for the final overlay plot.

**Output:** `outputs/csv_results/task3/finetuned_degraded_per_image.csv`,
`outputs/graph_results/task3/finetuned_vs_pretrained_comparison.png`.

**Verification:** Comparison plot exists and shows 3 distinguishable curves;
fine-tuned recall should be higher than pretrained-distorted at most SNR
levels (if not, that's a real, reportable finding — not a bug).

**Status: ✅ Done (2026-07-17), with one path adjustment.** Ran
`run_vehicle_detection_degradation.py --model yolov8n_finetuned.pt` full
9-level sweep, writing to `outputs_finetuned/task3/` (a separate location
from the pretrained results, via the legacy single-`--out_dir` mode — not
`outputs/csv_results/task3/finetuned_*` as originally planned, to avoid any
risk of the two runs' files colliding in the same folder). Built
`Amit_project/compare_finetuned_vs_pretrained.py`, which reads both
locations and produces
`outputs/graph_results/task3/finetuned_vs_pretrained_{recall,iou}.png`.
**Result: fine-tuned clearly beats both alternatives on every distortion**
(mean recall over all 9 levels): low_light 0.262→0.355→**0.414**,
motion_blur 0.259→0.261→**0.451**, rain 0.260→0.275→**0.452** (pretrained-
distorted → pretrained-enhanced → fine-tuned-distorted). The recall-vs-SNR
plot shows the fine-tuned curve consistently above both others across the
whole SNR range for all 3 distortions, only converging back down at the
most extreme low-light levels (expected — no amount of fine-tuning
compensates for near-total information loss).

---

### 5.5 Phase 5 quick-check

**Goal:** Confirm the fine-tuning pipeline produces a real, usable model and
a real, sensible three-way comparison — not just that the scripts ran
without crashing.

**Input:** `Amit_project/yolov8n_finetuned.pt`, `outputs_finetuned/task3/`,
`outputs/graph_results/task3/finetuned_vs_pretrained_*.png`.

**Implementation:** No separate run needed — verified directly from 5.3/5.4's
actual full-scale outputs (not a smoke-scale check, since by this point in
the phase the full run was already done and both fast to verify and more
informative than a synthetic smoke test would be).

**Location:** N/A (verification only).

**Output:** N/A.

**Verification:** (a) fine-tuned weights load via `YOLO(...)` with the
correct 5 class names; (b) both comparison plots exist and show 3
distinguishable, differently-shaped curves per distortion; (c) fine-tuned
recall exceeds both alternatives for the *majority* of SNR levels across all
3 distortions (not required to win everywhere — a real result, not a
mandated outcome).

**Status: ✅ Done (2026-07-17).** All three passed — see 5.3/5.4's status
notes for the concrete numbers and plot description. **Phase 5 is fully
done — safe to start Phase 6.**

---

## Phase 6 — README and submission finalization

### 6.1 Embed Task 1 results and interpretation in README

**Goal:** Replace README section 6/7 `TODO` markers for Task 1 with actual
embedded plots + 2-3 sentence interpretation, per the course's own example
style (slide 24-25).

**Input:** `outputs/graph_results/task1/*.png` (post Phase 3/4),
`outputs/csv_results/task1/*.csv`.

**Implementation:** Markdown image embeds + short interpretive prose, same
style as the course brief's own example blockquote in README section 6.

**Location:** `README.md`.

**Output:** No new files — README edit only.

**Verification:** README section 6/7 for Task 1 contains rendered image
links (not just a bullet list of file paths) and interpretive text, not a
`TODO`.

**Status: ✅ Done (2026-07-17).** Embedded `lane_offset_vs_snr.png` and
`lane_offset_per_distortion.png` with interpretation (survival rate
collapses under `low_light`/`motion_blur` but degrades gently under `rain`;
enhancement helps `low_light`/`motion_blur` but slightly *hurts* `rain`
since the bilateral de-rain filter smooths away the same edges Hough needs).
Also **fixed a bug found while doing this**: sections 5–7 still referenced
the deleted legacy `results/task2_feature_matching/`,
`results/task3_vehicle_detection/` paths (from before Phase 4's cleanup) —
updated every reference to the current `outputs/{csv_results,graph_results,
visualizations}/task{1,2,3}/` layout, and verified every embedded image
path actually resolves to a file on disk.

---

### 6.2 Embed Task 2 results and interpretation in README

**Goal:** Same as Task 6.1, for Task 2.

**Input:** `outputs/graph_results/task2/*.png` (post Phase 2/4),
`outputs/csv_results/task2/*.csv`.

**Implementation:** Same pattern as Task 6.1.

**Location:** `README.md`.

**Output:** No new files — README edit only.

**Verification:** Same as Task 6.1, for Task 2's section.

**Status: ✅ Done (2026-07-17).** Embedded `inlier_ratio_vs_snr.png`,
`match_ratio_vs_snr.png`, `inlier_ratio_per_distortion.png`,
`match_ratio_per_distortion.png` with interpretation covering both the
`inlier_ratio` (flat/noisy) vs. `match_ratio_vs_baseline` (monotonic)
distinction from section 4, and the mixed enhancement results per
distortion.

---

### 6.3 Embed Task 3 results and interpretation in README

**Goal:** Same as Task 6.1, for Task 3, including the per-class breakdown and
fine-tuning comparison.

**Input:** `outputs/graph_results/task3/*.png` (post Phase 1/2/4/5),
`outputs/csv_results/task3/*.csv`.

**Implementation:** Same pattern as Task 6.1, plus a per-class results table
(car/truck/bus/motorcycle/bicycle x Precision/Recall/F1/IoU) and the Part 4
fine-tuning comparison plot + interpretation.

**Location:** `README.md`.

**Output:** No new files — README edit only.

**Verification:** Same as Task 6.1, for Task 3's section; per-class table
present; Part 4 section no longer reads "not yet implemented."

**Status: ✅ Done (2026-07-17), with one scope adjustment: no separate
Precision/Recall/F1/IoU table per class** — the per-SNR per-class plots
(`recall_vs_snr_{car,truck,bus}.png`, embedded) already convey the
per-class breakdown across severity, which is more informative than a
single averaged table row per class; a static table was judged redundant
rather than skipped by oversight. Embedded the aggregate recall/IoU plots,
all 3 per-class recall plots (motorcycle/bicycle correctly have none — zero
GT instances), and the full Part 4 section with the fine-tuned-vs-pretrained
comparison plot, results table, and interpretation (see section 8, now
fully populated with real numbers from Phase 5).

---

### 6.4 Fill remaining README TODOs and prepare final PPT

**Goal:** Final submission polish — everything the course brief's
"Requirements"/"Submission" slides (40, 39) ask for that isn't covered by
Tasks 6.1–6.3.

**Input:** Remaining `TODO`s in `README.md`: dataset source link (section 1),
team emails (Team table), "Known limitations" section accuracy check
(section 11 — some limitations will now be resolved, e.g. pseudo-GT caveat
for Task 3), section 12's submission checklist itself.

**Implementation:** Fill in the literal TODOs; then build the final PPT as an
"easy-to-read version" of the now-complete README (course requirement,
slide 39) — one slide per README section, reusing the same embedded plots.

**Location:** `README.md`; new `docs/final_presentation.pptx` (or `.pdf`).

**Output:** Updated `README.md`, `docs/final_presentation.pptx`.

**Verification:** `grep -n "TODO" README.md` returns nothing (or only
genuinely deferred items with an explicit reason); PPT file exists and covers
all 4 parts + team/dataset/requirements slides.

**Status: 🟡 Partially done (2026-07-17).** README TODOs filled: dataset
link (real BDD100K URL), section 11 limitations updated (added the
image-corruption fix and fine-tuning-scale notes), section 12 checklist
updated. **Two items intentionally left open, not oversights:** (1) team
emails in the "Team" table — genuinely can't be filled in on the user's
behalf, flagged in section 12 as something only they can fill; (2) the
final PPT — not yet built, needs a decision on whether/when to generate it
(could use the `pptx` skill to build one from the now-complete README).

---

### 6.5 Phase 6 quick-check

**Goal:** Confirm the README is internally consistent — no dead paths, no
unresolved TODOs (besides the two intentionally-deferred items above), and
every embedded image actually renders.

**Input:** `README.md`, `outputs/graph_results/task{1,2,3}/*.png`.

**Implementation:** `grep -n "TODO\|results/task" README.md` (should only
match the team-email placeholders and the section-12 checklist item, not
any dead `results/` path); a small script extracting every markdown image
link and checking it resolves on disk.

**Location:** Run from repo root.

**Output:** N/A (verification only).

**Verification:** (a) no stale `results/task2_feature_matching`/
`results/task3_vehicle_detection` references remain; (b) all embedded image
paths exist; (c) only the two intentionally-deferred TODOs remain.

**Status: ✅ Done (2026-07-17).** (a) confirmed via grep — zero dead
`results/task` references remain (all replaced during 6.1's pass). (b)
verified programmatically: all 14 embedded image paths in README resolve to
existing files. (c) confirmed: only the "Team" email placeholders and the
PPT/Moodle-registration checklist items remain, both correctly left for the
user. **Phase 6 is done except the final PPT — ask the user whether/when to
build it.**

**Addendum — per-class IoU on clean images (2026-07-17, requested by the
user after seeing course brief slide 16's equivalent chart for semantic
segmentation):** added `plot_per_class_metric_on_clean()` to
`run_vehicle_detection_degradation.py` — a bar chart per class on the
**clean baseline only** (not vs. SNR), sorted descending, with a dashed
mean-of-all-classes reference line, matching slide 16's exact style.
Wired into `main()` so future full runs generate
`per_class_iou_clean.png`/`per_class_recall_clean.png` automatically;
generated immediately from the existing `baseline_vs_gt_metrics.csv`
(no re-run needed). Result: `bus` (0.94) > `truck` (0.88) > `car` (0.83),
mean 0.831. Embedded in README section 5 with interpretation.

---

## Addendum — README rework (2026-07-17, user-requested restructuring)

The user provided a detailed list of README improvements ("README Notes /
TODO") and asked for all of it except the metrics-replacement item (kept
`inlier_ratio`/`match_ratio_vs_baseline`/`matched_recall`/`mean_iou_matched`
unchanged, per explicit instruction). Confirmed via clarifying question:
task-level renaming maps Task 1→"Low Level", Task 2→"High Level", Task
3→"Deep Learning (DL)" (Option A).

**Done:**
- Intro now states the project focuses on road scenes and vehicles.
- Task levels renamed per Option A above (section 4's table).
- Removed the "TODO before submission" checklist section entirely (folded
  the two genuinely-outstanding items — team emails, PPT — into a one-line
  note at the very end instead of a checklist).
- Dataset section: clarified Task 2 = frames from a single driving video;
  added a 3×3 sample-image table (3 raw images per task/dataset), with a
  note that Task 2's two right-hand samples are an actual matched pair
  (`00e9be89-00000015`/`00000100`), not arbitrary images.
- Ground Truth: rewritten as its own short, standalone section — a table of
  which tasks use real GT vs. pseudo-GT, and one paragraph explaining what
  each means (replacing the previous much longer methodology prose spread
  across two paragraphs).
- Distortions: added a 3×3 grid image per distortion type (levels 1-9 on
  one fixed representative image, raw distorted frames, no method overlay)
  — new script `tools/build_distortion_grids.py`, output to
  `docs/readme_assets/distortion_grid_{motion_blur,low_light,rain}.png`.
- All task-related sections (4/5/6/7/8) now consistently present Task 1,
  2, 3 in that numerical order (previously mixed, e.g. section 4's old
  metrics section had Task 2, Task 3, Task 1).
- Every task section has at least one representative "method + result"
  image (reused existing `outputs/visualizations/task{n}/
  baseline_visualizations/` files — no new images needed).
- Task 1: removed the separate left/right lane-line metric split from the
  presented metric — `survival_rate` (already a combined metric) is now the
  single headline number, not accompanied by a left-line/right-line
  breakdown table.
- Enhancement section: added a distorted-vs-enhanced before/after image
  table, one row per task, all using the same representative case
  (`low_light`, level 5) for a fair side-by-side comparison — reused
  existing `degraded_visualizations`/`enhanced_visualizations` files, no
  new images needed.
- Metrics (Task 2/3): explicitly left unchanged per the user's instruction
  this round — flagged as a possible future follow-up, not silently done.

**Bug fixed along the way:** the repository-structure section's `tools/`
listing referenced a script (`build_readme_assets.py`) that was never
actually created — caught during verification and corrected to the real
script names (`build_distortion_grids.py`, `make_smoke_test_sample.py`).

**Verification:** re-ran the same image-existence check as Phase 6's
quick-check — all 37 embedded images in the new README resolve to existing
files on disk.

---

## Addendum — README v2 (2026-07-17, new template-driven structure)

The user provided a new README skeleton (`README2.md` on their Desktop, a
placeholder-driven template: `ADD TEXT`/`ADD IMAGE`/`ADD GRAPH`/`ADD TABLE`
markers) along with an exact target folder structure for
`docs/readme_assets/` (dataset/baseline/distortions/distortion_results/
enhancements/finetuning subfolders). This fully replaces the previous
README structure from the earlier "README rework" addendum above.

**Discovered before starting:** the user's initial message claimed "we
updated the README to a new structure," but `README.md` on disk (and on
`origin/Alon_branch`, confirmed via `git fetch` + diff) was unchanged from
the previous session's version — surfaced this discrepancy rather than
guessing, which led to the user sharing the actual template file.

**Step 1 — `tools/build_readme_assets.py`:** a single script that populates
the exact requested `docs/readme_assets/` structure purely from existing
outputs (CSVs, plots, visualization PNGs, raw dataset images) — no
pipeline/detector/matcher re-run. Two kinds of new composites were built
(from existing artifacts only, never recomputed metrics):
- 3×3 grids per task (`distortion_results/task{n}/grid_by_distortion_and_
  severity.png`): rows = distortion, columns = severity (levels 2/5/8),
  each cell = an existing method-overlay visualization PNG.
- Before/after pairs per task/distortion (`enhancements/task{n}/before_
  after_<distortion>.png`): existing degraded/enhanced visualization PNGs
  side by side.
- One genuinely new plot: `baseline/task2/baseline_metrics.png` (mean
  `inlier_ratio`/`match_ratio` bar chart) — no prior plot covered Task 2's
  baseline alone, but it's a direct re-plot of `baseline_per_pair.csv`
  columns, not a re-run.

Handled one known real gap gracefully: `task2/low_light/level_8` has no
saved visualization (matching failed completely at that severity) — the
grid-building code draws a labeled gray placeholder instead of crashing or
leaving a confusing blank cell.

**Step 2 — full README rewrite**, using the user's template verbatim as the
structure/section order, filling every placeholder with real content:
methodology explanations (from the actual code, not invented), the actual
severity-level parameter/SNR tables (pulled from `outputs/csv_results/
task1/level_summary.csv`), and all real result numbers already established
this session. All images now point into `docs/readme_assets/...` instead of
directly into `outputs/`/`data/` — solves the earlier git-sharing problem
(raw `data/*.jpg` and some `outputs/*.png` paths would otherwise be
gitignored or bulky) by consolidating everything the README needs into one
deliberately-shareable folder.

**Verification:** same check as before — programmatically confirmed all 37
embedded image paths in the new README resolve to existing files. Removed
the now-orphaned flat `docs/readme_assets/distortion_grid_*.png` files
(superseded by the nested `distortions/<name>/grid_levels_1_to_9.png`
versions) after confirming zero remaining references to them.

---

## Addendum — final asset spec refinement (2026-07-17)

The user provided a precise, final image/graph checklist (organized by
Part 1-5), which superseded some choices made in `build_readme_assets.py`'s
first version:
- **Severity columns changed from levels 2/5/8 ("low/medium/high") to
  literally levels 1/5/9**, per the exact spec.
- **Row order changed** from `[low_light, motion_blur, rain]` to
  `[motion_blur, low_light, rain]` for every 3x3 comparison grid.
- **New requirement not previously built:** Part 4 needs a 3x3 grid of
  *enhanced* results (rows=distortion, cols=level 1/5/9), parallel to the
  existing distorted-results grid, per task —
  `enhancements/task{n}/grid_enhanced_by_distortion_and_severity.png`.
  Added `build_comparison_grid(stage="degraded"|"enhanced", ...)`, a single
  parameterized function now used for both the distortion-results and
  enhancement-results grids (previously two near-duplicate functions).
- **`distortions/` grids no longer depend on the old flat `docs/readme_
  assets/distortion_grid_*.png` files** (which were deleted in the previous
  session) — `build_distortions()` now regenerates the 9-level showcase
  grids directly from the sample image + `augmentation_levels.py`'s
  functions, inline, rather than copying a pre-existing file.
- **Fine-tuning training visualizations included**: the spec said "Images:
  NONE (unless training visualizations exist)" — checked
  `Amit_project/runs/detect/train-2/` and found `results.png` (loss/metric
  curves) and `confusion_matrix.png` do exist (ultralytics writes these
  automatically), so both were added to `finetuning/`.

Verified: `task2`'s known gap widened slightly at the new levels checked —
both level 8 *and* level 9 have no saved visualization for `low_light`
(matching failed completely from level 8 onward), still handled by the
existing gray-placeholder fallback, not a new bug.

Re-ran the full script; all outputs verified present and visually spot-checked.

---

## Addendum — README synced to final asset spec (2026-07-17)

Updated `README.md` to match the refined `build_readme_assets.py` output:
- Part 3.2 (distortion results): row/column descriptions corrected to
  "motion_blur / low_light / rain" and "Level 1 / Level 5 / Level 9" for
  all three tasks (previously said "low/medium/high, levels 2/5/8"). Task
  2's known-gap note updated from "level 8" to "level 9" (the actual level
  where the last visualization still exists is level 7; 8 and 9 both fail).
- Part 4 (enhancement results): added the 3 new
  `grid_enhanced_by_distortion_and_severity.png` images, one per task.
- Part 5.2 (training setup): added the two training-visualization images
  (`training_curves.png`, `training_confusion_matrix.png`).

Verified: all 42 embedded images (up from 37) resolve to existing files.

---

## Addendum — single metric per task (2026-07-17, user-requested reduction)

The user asked to reduce each task to exactly one reported metric,
delegating the choice to Claude ("you pick the best one and delete the
other(s)"). Task 1 already had one (`survival_rate`), untouched.

**Task 2 — kept `match_ratio_vs_baseline`, dropped `inlier_ratio`.**
Justification: `inlier_ratio` measured flat/noisy across every severity
level in the actual data (RANSAC's geometric check is fairly binary
regardless of how few candidates remain) — it never told a robustness
story. `match_ratio_vs_baseline` (the metric fixed earlier this session)
decreases monotonically with SNR and correctly shows enhancement recovery.

**Task 3 — kept `matched_recall`, dropped `mean_iou_matched` and
`retention_ratio`.** Justification: `mean_iou_matched` stayed nearly flat
(0.82-0.86) across every severity level in the actual data — the dominant
failure mode under distortion is missed detections, not inaccurate boxes,
so IoU carries no robustness signal here. `matched_recall` is both the
standard object-detection metric and the one that actually moves with
severity/enhancement/fine-tuning.

**Implementation (not just README):**
- `run_feature_matching_degradation.py`: `level_summary.csv` no longer has
  an `inlier_ratio` column; `inlier_ratio_vs_snr.png`/`inlier_ratio_per_
  distortion.png` no longer generated. `inlier_ratio` is still computed
  per-pair (RANSAC needs it internally) and left in the raw
  `degraded_per_pair.csv`/`enhanced_per_pair.csv` as harmless data, just no
  longer aggregated or plotted.
- `run_vehicle_detection_degradation.py`: same pattern —
  `level_summary.csv`/`level_summary_per_class.csv` now only have
  `matched_recall`; `iou_vs_snr.png`, `iou_per_distortion.png`,
  `per_class_iou_clean.png`, `iou_vs_snr_{car,truck,bus}.png` no longer
  generated. `mean_iou_matched`/`retention_ratio` stay in the raw per-image
  CSVs (cheap byproduct of the same IoU-matching call).
- `compare_finetuned_vs_pretrained.py`: drops the IoU three-way comparison,
  keeps only the recall one.
- `tools/build_readme_assets.py`: updated to stop copying/generating any
  of the now-removed plots; Task 2's baseline-metrics chart is now a
  single bar instead of two.
- Existing CSVs/plots were **not regenerated via a full pipeline re-run**
  — the raw per-pair/per-image CSVs already had all underlying columns, so
  `level_summary.csv` files and the affected plots were rebuilt directly
  from those existing CSVs (fast, no detector/matcher re-invocation), and
  the 9 now-obsolete plot files were deleted.
- `README.md`: sections 2.2 (metrics table + per-task explanations, now
  documenting the dropped metric and why), 2.3 (baseline results), 3.2
  (distortion results), 4.2 (enhancement results per task), and 5.3
  (fine-tuning comparison) all updated to remove the dropped-metric
  images/text.

**Verification:** all touched Python files re-compiled cleanly; confirmed
zero `*iou*`/`*inlier*` files remain under `docs/readme_assets/`; confirmed
all 36 remaining embedded README images (down from 42) resolve to existing
files.

---

**Addendum — swapped Task 2's example frame pair (2026-07-20).** The
sample pair used throughout the README for Task 2 (`00e9be89-00000015`/
`00e9be89-00000100`) showed a large scene change between frames, making the
ORB feature matching hard to see clearly. User pointed to an existing,
much clearer result already on disk
(`outputs/visualizations/task2/baseline_visualizations/match_00e9be89-00000100__00e9be89-00000105.png`,
also two consecutive frames) and asked for it to replace the old pair
everywhere.

**Implementation:**
- `tools/build_readme_assets.py`: `TASK2_SAMPLE_PAIR` changed from
  `("00e9be89-00000015", "00e9be89-00000100")` to
  `("00e9be89-00000100", "00e9be89-00000105")`.
- The new pair had no saved degraded/enhanced visualizations yet (only
  baseline) — only the first pair per level gets `save_visualization=True`
  during a full production run, and this pair wasn't that first pair.
  Generated the missing ones with a one-off script calling
  `process_pair`/`enhancements.ENHANCEMENTS` directly for just this pair,
  at levels 1/5/9 (degraded) and level 5 (enhanced), for all 3 distortions.
  All succeeded except `low_light`/level 9 — ORB finds zero surviving
  descriptors for this pair at that severity (`[WARN] No descriptors found
  for pair`), a genuine result (same pattern already documented for the
  old pair), not a bug. `docs/readme_assets/`'s existing `img_or_placeholder`
  gray-placeholder logic handles this gap automatically.
- Re-ran `tools/build_readme_assets.py` to regenerate all Task 2 assets
  under `docs/readme_assets/{baseline,distortion_results,enhancements}/task2/`
  with the new pair.
- `README.md`: both embedded image paths referencing the old pair's
  filename (`match_00e9be89-00000015__00e9be89-00000100.png`, in the Part 2
  baseline section and the Part 5 fine-tuning section) updated to the new
  filename (`match_00e9be89-00000100__00e9be89-00000105.png`). The
  Known-Issues note about `low_light` level 9 having no visualization
  already applied generically (same distortion/level) and needed no wording
  change.

**Verification:** confirmed all 36 embedded README images still resolve to
existing files after the swap.

---

**Addendum — `.gitignore` fix + summary PPT (2026-07-20).**

**`.gitignore`:** the blanket `*.png`/`*.jpg`/`*.jpeg` rule (present since
before this session) was silently excluding every result graph and
visualization under `outputs/` and `docs/readme_assets/` from git, which
would have made the shared repo useless to a teammate/grader without
re-running the whole pipeline. Added two negation rules right after the
blanket rule: `!outputs/**/*.png` and `!docs/readme_assets/**/*.png`.
Also added `Amit_project/runs/` to `.gitignore` (33MB of duplicate
`ultralytics` training-run artifacts, already summarized into
`docs/readme_assets/finetuning/`). Verified with `git check-ignore -q` on
representative paths: result PNGs are no longer ignored, `runs/` still is.
Also filled in both teammates' real emails in the README's Team table
(previously `TODO (fill in)`).

**Summary PPT:** built `docs/Project_Summary.pptx` — a 14-slide deck
condensing the README for presentation (overview, dataset, the 3 tasks,
metrics, distortions, results/enhancement/fine-tuning charts, known
issues, conclusion). Deviation from the skill's default workflow: this
machine has no Node.js, no working `pip install` (blocked by a local
SSL/proxy issue, same constraint hit earlier extracting Task 3's GT), and
no LibreOffice, so the usual `pptxgenjs`-script / `soffice`-render path
wasn't available. Built the `.pptx` directly as an OOXML package using
only Python's standard library (`zipfile` + hand-written slide/theme/
layout/master XML) — script kept at
`C:\Users\amit\AppData\Local\Temp\claude\...\scratchpad\ppt\genpptx.py`
(scratch, not part of the repo). QA done without a renderer: verified zip
integrity (`zipfile.testzip()`), dumped every slide's text runs via regex
over the raw XML to check for typos/placeholder text, and manually
re-checked every slide's coordinate math for overlaps/margins (caught and
fixed two near-overlaps: a caption crowding the footer on the Task 2
slide, and a description line crowding its own heading on the distortions
slide). Also fixed a real bug caught during QA: multi-run text blocks
(e.g. name+email) had raw `\n` characters embedded inside a single
`<a:t>` element, which isn't a reliable line break in OOXML — reworked
the paragraph-splitting logic to split per-run before building `<a:p>`
elements, confirmed no literal newlines remain inside any `<a:t>`.
