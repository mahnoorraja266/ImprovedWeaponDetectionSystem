## Weapon Detection Model Plan (Based on current dataset snapshot)

Dataset snapshot used for planning:
- Total images found: `7794`
- Classes from `data.yaml`: `criminal (0)`, `person (1)`, `weapon (2)`
- Current split on disk:
  - Train: `7225` images (~92.7%)
  - Valid: `264` images (~3.4%)
  - Test: `305` images (~3.9%)
- Label quality quick check:
  - Empty label files: `3` (all in train)
  - Malformed rows: `0`
- Total labeled boxes:
  - class `0` (criminal): `3882`
  - class `1` (person): `6663`
  - class `2` (weapon): `6063`

---

## Phase 1: Dataset Audit & Validation

Goal: Ensure data is usable before training.

1) Class distribution audit
- Compute per-class image-level and box-level counts for `criminal/person/weapon`.
- Flag imbalance thresholds:
  - mild imbalance: >1.5x
  - severe imbalance: >2.5x
- Current box-level ratio is moderate (`person` > `weapon` > `criminal`), but image-level presence still needs audit.
- If minority class underperforms in pilot run, handle with:
  - targeted data collection for minority scenes
  - class-aware sampling (oversample minority images during training)

2) Annotation quality audit
- Verify boxes are tight and aligned to visible objects.
- Spot-check at least 300 random images across all splits.
- Detect and fix:
  - missing weapon labels
  - oversized/shifted boxes
  - wrong class IDs
  - duplicate overlapping labels

3) Remove low-value/noisy samples
- Remove or relabel samples that hurt CCTV realism:
  - cartoons/illustrations
  - unrealistic studio-only closeups
  - extreme edits not representative of deployment camera feed

4) Split validation and leakage control
- Current split is not ideal for robust evaluation (too train-heavy).
- Re-split to target:
  - Train 70%
  - Val 20%
  - Test 10%
- Enforce leakage rules:
  - no near-duplicate frames across splits
  - no same-video sequence split between train and val/test
  - no filename/hash duplicates across splits

Deliverable for Phase 1:
- Cleaned dataset + leakage report + final per-class statistics.

---

## Phase 2: Data Preprocessing & Enhancement

Goal: Match training data closer to real CCTV operating conditions.

1) Input normalization
- Train at `640x640` first (already aligned with dataset export).
- For edge-speed profile, run secondary experiment at `416x416`.

2) Annotation format verification
- Confirm every label row follows YOLO format:
  - `class x_center y_center width height`
- Validate normalized range [0,1] and valid class IDs 0..2.

3) Realistic augmentations (controlled)
- Keep augmentations that simulate production cameras:
  - low light / underexposure
  - sensor noise/compression artifacts
  - motion blur
  - partial occlusion
  - mild perspective/angle shifts
- Avoid aggressive transforms that change weapon semantics or create fake geometry.

4) Hard-example set creation
- Build a small curated set of difficult scenes:
  - far distance weapons
  - partial visibility
  - crowd scenes with multiple persons
  - reflections and cluttered backgrounds
- Use this set for repeated checkpoint comparisons.

Deliverable for Phase 2:
- Final preprocessing/augmentation profile and validated training-ready dataset.

---

## Phase 3: Model Selection & Baseline Training

Goal: Establish a strong, reproducible baseline.

1) Baseline model choice
- Start with `YOLOv8n` (fastest edge baseline).
- Then train `YOLOv8s` for accuracy comparison.
- Use pretrained COCO initialization (transfer learning).

2) Baseline training setup
- Use two-stage training per model on the **full dataset**:
  - Stage A: `25` epochs
  - Stage B: resume from `last.pt` for another `25` epochs
  - Effective total: `50` epochs (same model, continuous learning)
- GPU-aware config (P2000 4GB):
  - `YOLOv8n`: start with `batch=4` (drop to 2 if OOM)
  - `YOLOv8s`: start with `batch=2` (drop to 1 if OOM)
  - `imgsz=640` first; fallback to `512/416` only if memory instability appears
  - keep `amp=True`, early stopping enabled, and avoid major hyperparameter changes for fair comparison.

3) Metrics to track every run
- `mAP@0.5`
- `mAP@0.5:0.95`
- Precision / Recall
- Confusion matrix per class
- Per-class AP (especially class `weapon`)

4) Acceptance gate for “usable baseline”
- No severe class collapse in confusion matrix.
- `weapon` class recall stable under realistic confidence threshold.
- Comparable behavior on hard-example set (not only easy validation frames).

Deliverable for Phase 3:
- Baseline checkpoints + experiment log + metrics summary (`v8n` vs `v8s`).
- Artifact note on `.pt` files:
  - During training, each run typically produces both `last.pt` and `best.pt`.
  - For deployment/use, a **single `.pt` file is possible**: select the final chosen `best.pt` (usually from the better baseline run) as the canonical model artifact.
