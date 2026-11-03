# Design: Data Pipeline

Goal: turn raw videos into a **single cached feature tensor** that every model consumes, with
**saved, shared folds**. This is the most important fairness control in the study. If the inputs
differ, the comparison is meaningless.

```
data/raw/labels.json + video-*.mp4
        │  src/data/video_io.py        sample_frames(path, T) -> uint8 (T, 224, 224, 3) RGB
        ▼
        │  src/data/extract_features.py  MobileNetV2(pooling="avg") on batches of frames
        ▼
data/features/oral_T40.npz   {features: float32 (N, 40, 1280), labels: int8 (N,), ids: str (N,), ages: int (N,)}
        │  T=20 → features[:, ::2]   T=10 → features[:, ::4]   (subsample, no re-extraction)
        ▼
        │  src/data/splits.py          StratifiedKFold(5) × seeds, inner 15% stratified val
        ▼
data/splits/oral_seed{0,1,2}.json   (committed: video ids only, no medical content)
```

---

## 1. Source data (oral set; measured on all 535 videos, 3 Nov)

| Set | Path in DL_test1 | Videos | Normal / Abnormal | Resolution (w×h) | Length (median, min–max) | FPS (median, range) | Ages |
|---|---|---:|---|---|---|---|---|
| **Oral** | `Task - 2 ORAL/oral/` | 535 | 452 / 83 (84.5% / 15.5%) | 480×640 (portrait), all videos | 26.4 s (0.9–64 s); 24–1364 frames | 23.6 (13–30) | 3–18, median 10 |

- Size on disk: 3.2 GB.
- Measured from container metadata on all 535 videos. Header frame counts can be wrong, so the
  extractor logs the frames it actually decodes.
- Only 3 videos report fewer than 40 frames, and none fewer than 20.
- Every `labels.json` entry has a matching file, and vice versa.
- The ear set (874 videos) and the external ear test set were dropped on 3 Nov 2026 and are not
  used.

`labels.json` schema: `[{"id": "video-1.mp4", "interpretation": "Normal" | "Abnormal" | <free text>, "age": int}]`.
- In the curated oral set every non-"Normal" interpretation is already mapped to "Abnormal"
  (verified: only these two values occur). `DL_test1/rename.py` did that mapping and renamed files
  to `video-N.mp4`.
- Defensive rule in code: map any interpretation other than "Normal" to Abnormal.
- **There is no patient ID.** See the leakage risk in `docs/03_scope.md`.

**Label convention (changed from legacy):** `Abnormal = 1` (positive class), `Normal = 0`.
- The legacy code used Normal = 1. That made "sensitivity" mean recall of *Normal*, which is
  clinically backwards.
- All metrics in this repo treat Abnormal as positive.

## 2. Frame sampling: `src/data/video_io.py`

- **Uniform temporal sampling:** target indices `np.linspace(0, n_frames - 1, T).round().astype(int)`.
  This is the same idea as legacy, so the frames are comparable.
- **Decode sequentially** (`cap.read()` / `cap.grab()` in a loop) rather than `cap.set(POS_FRAMES)`
  for each index.
  - Seeking in mp4/H.264 is slow and can land on the wrong frame.
  - `CAP_PROP_FRAME_COUNT` is sometimes wrong. If decoding ends early, resample from the frames we
    actually decoded.
- **Short videos:** if fewer than T frames decode (the shortest oral clip has 24), repeat-pad by
  sampling indices with replacement. The legacy code **silently dropped** any video with
  `len(frames) != NUM_FRAMES`. That biases the dataset, and the number dropped was never logged.
  We log every pad or failure.
- **Colour:** OpenCV decodes **BGR**. Convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`. The
  legacy code fed BGR to an RGB ImageNet model.
- **Resize** to 224×224 with `cv2.INTER_AREA`.
  - The frames are portrait 480×640, so a direct resize squashes the aspect ratio.
  - We keep the direct resize for continuity with the legacy models. Letterboxing is a possible
    ablation but is out of scope.
- Return `uint8`. **Do not** divide by 255 or cast to float64 here.
  - Legacy `frame / 255.0` produced float64 arrays.
  - 535 × 20 × 224 × 224 × 3 × 8 bytes ≈ **12.9 GB** in RAM (≈ 32 GB at the legacy 50 frames),
    which is most or all of our 16 GB machine.

## 3. Feature extraction: `src/data/extract_features.py`

```python
backbone = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg")
backbone.trainable = False
x = keras.applications.mobilenet_v2.preprocess_input(frames.astype("float32"))  # → [-1, 1]
feats = backbone.predict(x, batch_size=64)        # (T, 1280)
```

- **`preprocess_input` is mandatory.** MobileNetV2's ImageNet weights expect inputs in `[-1, 1]`.
  The legacy code fed `[0, 1]`, which shifts every activation distribution. This is the largest
  correctness fix over DL_test1.
- `pooling="avg"` is equivalent to the legacy `TimeDistributed(GlobalAveragePooling2D())`.
- Extract **once at T = 40**. Derive T = 20 as `[:, ::2]` and T = 10 as `[:, ::4]`.
  - `linspace` over 40 points subsampled by 2 is not *exactly* `linspace` over 20, but it is
    uniform, and it gives every T the same underlying frames.
  - This is cheaper and more consistent than three separate extractions.
- Output: `data/features/{dataset}_T40.npz` with `features (N, 40, 1280) float32`, `labels`, `ids`,
  `ages`, plus a sidecar `{dataset}_T40.meta.json` holding:
  - backbone name and Keras version
  - preprocessing flags
  - extraction date
  - per-video frame counts, pad flags and decode failures
- Size: 535 × 40 × 1280 × 4 bytes ≈ **110 MB**. It fits in RAM easily.
- Estimated cost: about 21k frames through MobileNetV2 on CPU, roughly 5–15 min, plus a
  sequential decode of about 350k frames (oral clips are long, so decoding may dominate).
  This one-off cost replaces re-running the CNN on every epoch of every fold, which is what the
  legacy `TimeDistributed(base_model)` did. That change is what makes the full 90-run grid
  feasible on a laptop.
- Also record the **per-frame backbone latency** during extraction. It feeds into the end-to-end
  compute numbers (benchmarking protocol §4).

### Why frozen features are fair (and what they cost us)

- The backbone is frozen in every model, and nothing is augmented. Under those conditions,
  computing it inside the model or outside gives *identical* outputs. Caching is a pure speed-up.
- The cost: we cannot use per-epoch frame augmentation. That is acceptable, because augmentation is
  out of scope and would add a confounder.

## 4. Splits: `src/data/splits.py`

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` on labels, for seeds `{0, 1, 2}`.
  Seed 0 is mandatory. Seeds 1 and 2 are added if time allows (repeated CV makes the statistics
  stronger).
- Inside each outer training fold, hold out a **stratified 15% inner validation** split. It is
  used only for early stopping and threshold selection. The outer fold is touched once, for the
  final evaluation.
- Save as JSON: `{"seed": 0, "folds": [{"train": [ids], "val": [ids], "test": [ids]}, …]}`.
  - The files hold IDs only, so they are safe to commit.
  - **Every model reads the same file.**
- The legacy code used plain `KFold` (not stratified) and evaluated on the same split it
  early-stopped on, or had no early stopping at all.

## 5. Optional: grouped split sensitivity check

`labels.json` has no patient ID. If consecutive IDs share the same age, they *might* be the same
child (for example retakes). As a sensitivity analysis, form pseudo-groups from runs of
consecutive IDs with the same age and rerun one seed with `StratifiedGroupKFold`. Report it if the
AUROC drops noticeably. This applies equally to both families.
