# Design: Data Pipeline

Goal: turn raw videos into a **single cached feature tensor** that every model consumes, with
**saved, shared folds**. This is the most important fairness control in the study. If the inputs
differ, the comparison is meaningless.

All code is in `src/common.py` (section 2, "Data"); settings are the constants at the top of that
file (`DATASET = "ear"`, `CACHE_FRAMES = 40`, `FRAME_SIZE`, `SEEDS`, `N_FOLDS`, `VAL_FRACTION`).

```
data/raw/labels.json + video-*.mp4
        │  sample_frames(path, T) -> uint8 (T, 224, 224, 3) RGB
        ▼
        │  extract_features()  MobileNetV2(pooling="avg") on the frames of each video
        ▼
data/features/ear_T40.npz   {features: float32 (N, 40, 1280), labels: int8 (N,), ids: str (N,), ages: int (N,)}
        │  T=20 → features[:, ::2]   T=10 → features[:, ::4]   (load_features(T); no re-extraction)
        ▼
        │  make_split() / write_splits()   StratifiedKFold(3) × 2 seeds, inner 15% stratified val
        ▼
data/splits/ear_seed{0,1}.json   (committed: video ids only, no medical content)
```

Commands (from the repo root):

```bash
python -m src.common extract     # options: --workers (decode threads, default 4), --threads, --overwrite
python -m src.common splits      # writes data/splits/ear_seed{0,1}.json
```

---

## 1. Source data (ear set; measured on all 874 videos, 3 Nov 2026)

| Property | Value |
|---|---|
| Videos | 874 mp4 + `labels.json`; every id has a file and vice versa |
| Labels | 643 Normal / 231 Abnormal (73.6% / 26.4%) |
| Resolution (w×h) | 480×640 (portrait), all videos |
| Header frames | 18–2078 (median 374) |
| FPS | 12.6–29.8 (median 29.0) |
| Duration | 0.9–71.1 s (median 13.1 s) |
| Ages | 3–18 (median 11) |
| Size on disk | 2.6 GB |

- Measured from container metadata. Header frame counts can be wrong, so the extractor logs the
  frames it actually decodes.
- 4 videos report fewer than 40 header frames (repeat-padded at T = 40), and 1 reports fewer
  than 20.
- The small external ear test set (50 videos) is not used.

`labels.json` schema: `[{"id": "video-N.mp4", "interpretation": "Normal" | "Abnormal", "age": int}]`.
- Only these two interpretation values occur. Defensive rule in code (`load_labels`): any
  interpretation other than "Normal" counts as Abnormal.
- **There is no patient ID.** See the leakage risk in `docs/03_scope.md`.

**Label convention (changed from legacy):** `Abnormal = 1` (positive class), `Normal = 0`.
- The legacy code used Normal = 1. That made "sensitivity" mean recall of *Normal*, which is
  clinically backwards.
- All metrics in this repo treat Abnormal as positive.

### Dataset history

The study briefly used the legacy project's oral-cavity set (535 videos, 83 Abnormal) before
returning to the ear set on 3 Nov 2026. A sanity run on the oral set (`lstm_base` and
`transformer_base`, T = 20, seed 0, fold 0) ran cleanly but learned nothing (test AUROC 0.52 and
0.50). A logistic regression on mean-pooled frozen MobileNetV2 features reached only AUROC
0.57–0.60 over 5 folds, and patient age alone reached about 0.66, so frozen whole-frame ImageNet
features carried weak signal for that task. The ear set is larger and has more Abnormal cases, so
the team switched to it. The oral data, features, splits and results were removed from the repo.

## 2. Frame sampling: `sample_frames()`

- **Uniform temporal sampling:** target indices `np.linspace(0, n_frames - 1, T).round().astype(int)`.
  This is the same idea as legacy, so the frames are comparable.
- **Decode sequentially** (`cap.grab()` in a loop, `retrieve()` only for wanted frames) rather
  than `cap.set(POS_FRAMES)` for each index.
  - Seeking in mp4/H.264 is slow and can land on the wrong frame.
  - `CAP_PROP_FRAME_COUNT` is sometimes wrong. The file is always read to the end; if the true
    count differs from the header, the indices are resampled over the true count (second pass).
  - Decoding stops only after 25 consecutive `grab()` failures (`MAX_CONSECUTIVE_FAILURES`), so a
    single corrupt packet mid-file does not truncate the clip.
- **Short videos:** if fewer than T frames decode, indices repeat (repeat-padding). The legacy code
  **silently dropped** any video with `len(frames) != NUM_FRAMES`. That biases the dataset, and
  the number dropped was never logged. We log every pad in the metadata file.
- **Colour:** OpenCV decodes **BGR**. Convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`. The
  legacy code fed BGR to an RGB ImageNet model.
- **Resize** to 224×224 with `cv2.INTER_AREA`.
  - The frames are portrait 480×640, so a direct resize squashes the aspect ratio.
  - We keep the direct resize for continuity with the legacy models. Letterboxing is a possible
    ablation but is out of scope.
- Return `uint8`. **Do not** divide by 255 or cast to float64 here.
  - Legacy `frame / 255.0` produced float64 arrays.
  - 874 × 20 × 224 × 224 × 3 × 8 bytes ≈ **21 GB** in RAM, more than our 16 GB machine.

## 3. Feature extraction: `extract_features()`

```python
backbone = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg")
backbone.trainable = False
x = keras.applications.mobilenet_v2.preprocess_input(frames.astype("float32"))  # → [-1, 1]
feats = backbone.predict_on_batch(x)               # (T, 1280)
```

- **`preprocess_input` is mandatory.** MobileNetV2's ImageNet weights expect inputs in `[-1, 1]`.
  The legacy code fed `[0, 1]`, which shifts every activation distribution. This is the largest
  correctness fix over the legacy code.
- `pooling="avg"` is equivalent to the legacy `TimeDistributed(GlobalAveragePooling2D())`.
- Videos decode in a thread pool (`--workers`, OpenCV releases the GIL) while the main thread runs
  the backbone. Any failure aborts before writing, so the cache always covers all of `labels.json`.
- Extract **once at T = 40**. Derive T = 20 as `[:, ::2]` and T = 10 as `[:, ::4]`.
  - `linspace` over 40 points subsampled by 2 is not *exactly* `linspace` over 20, but it is
    uniform, and it gives every T the same underlying frames.
  - This is cheaper and more consistent than three separate extractions.
- Output: `data/features/ear_T40.npz` with `features (N, 40, 1280) float32`, `labels`, `ids`,
  `ages`, plus a sidecar `ear_T40.meta.json` holding:
  - backbone name, Keras version and preprocessing flags
  - extraction date
  - per-video header and decoded frame counts, fps, pad flags and decode passes
- Size: 874 × 40 × 1280 × 4 bytes ≈ **179 MB**. It fits in RAM easily.
- Cost: about 35k frames through MobileNetV2 on CPU, plus a sequential decode of every frame in
  every file; roughly 6 minutes in total (5.5 min measured). This one-off cost replaces re-running the CNN on every
  epoch of every fold, which is what the legacy `TimeDistributed(base_model)` did. That change is
  what makes the full training grid feasible on a laptop.
- The backbone time per frame during extraction is logged as an indication only. The real
  end-to-end numbers come from `python -m src.common backbone` (benchmarking protocol §4).

### Why frozen features are fair (and what they cost us)

- The backbone is frozen in every model, and nothing is augmented. Under those conditions,
  computing it inside the model or outside gives *identical* outputs. Caching is a pure speed-up.
- The cost: we cannot use per-epoch frame augmentation. That is acceptable, because augmentation is
  out of scope and would add a confounder.

## 4. Splits: `make_split()` and `write_splits()`

- `StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)` on labels, for seeds `{0, 1}`
  (`N_FOLDS`, `SEEDS` in `src/common.py`). 3 folds give larger test folds (77 Abnormal each);
  the second seed makes the statistics stronger (repeated CV).
- Inside each outer training fold, hold out a **stratified 15% inner validation** split. It is
  used only for early stopping and threshold selection. The outer fold is touched once, for the
  final evaluation.
- Resulting sizes, every fold and seed: train 494–495 (131 Abnormal), val 88 (23 Abnormal),
  test 291–292 (77 Abnormal).
- Balanced class weights are about 0.68 Normal / 1.89 Abnormal.
- Saved as JSON: `{"seed": 0, "n_splits": 3, "val_fraction": 0.15, "dataset": "ear", "folds": [{"train": [ids], "val": [ids], "test": [ids]}, …]}`.
  - The files hold IDs only, so they are safe to commit.
  - **Every model reads the same file** (`load_split(seed)`, `fold_indices()`).
  - `src/tests.py` checks that the committed files equal `make_split(labels.json)`.
- The legacy code used plain `KFold` (not stratified) and evaluated on the same split it
  early-stopped on, or had no early stopping at all.

## 5. Optional: grouped split sensitivity check

`labels.json` has no patient ID. If consecutive IDs share the same age, they *might* be the same
child (for example both ears, or retakes). As a sensitivity analysis, form pseudo-groups from runs
of consecutive IDs with the same age and rerun one seed with `StratifiedGroupKFold`. Report it if
the AUROC drops noticeably. This applies equally to both families. Not implemented yet.
