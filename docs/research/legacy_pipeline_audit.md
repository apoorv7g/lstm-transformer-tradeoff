# Legacy Pipeline Audit: `DL_test1`

**Location:** `C:\Users\apoor\PycharmProjects\DL_test1` (local only; not in this repo).
**Audited:** 23 Oct 2026.
**Purpose:** record what the previous LSTM project did, what we reuse, and which bugs this repo
fixes. Read this before touching the data or comparing to the old numbers.

---

## 1. What DL_test1 is

DL_test1 is an earlier project, from Oct 2024 to Jan 2025. It classified pediatric screening videos
as **Normal / Abnormal** for two tasks, **ear (otoscopy)** and **oral cavity**, using a frozen
MobileNetV2 with an LSTM. It also shipped a desktop GUI (pywebview) that loads one ear model and one
oral model.

### Inventory

| Path | What it is |
|---|---|
| `dl_model_1.py` | First prototype: 64×64 frames, first 30 frames, tiny `TimeDistributed(Conv2D(32))` → LSTM(100). Training call commented out |
| `dl_model_2.py` | **Main training script.** Oral, 20 frames, 224², MobileNetV2 (frozen) → LSTM(256) → LSTM(128) → Dense(64) → Dropout(0.5) → sigmoid. `KFold(5, shuffle, rs=42)`, 30 epochs, batch 16. Saves `oral_..._fold-new-data-{k}.h5` |
| `dl_model_2_2.0.py` | Same model, oral, **NUM_FRAMES = 50**, single 80/20 split, saves `oral_video_classifier_lstm_mobilenet_final_v2.h5` |
| `dl_model_2_new_data_ear.py` | Same model, ear, single 80/20 split |
| `miniLSTM_miniGRU.py` | Ear, 5-fold, custom `MiniLSTM` cell (see §3.8). Despite the name, no GRU |
| `minLSTM_new_data.py` | Oral, custom `MiniLSTM`, **unshuffled** 80/20 split by index |
| `minLSTM/minLSTM.py` | Serializable `MiniLSTM` cell (with `add_weight`) for loading saved models |
| `*/ConfusionMatrix.py` (3 copies) | Load one model, predict **every** video in `labels.json`, plot the confusion matrix |
| `Task - 1 EAR/cli_ear_model.py` | CLI predictor (loads an *oral* model by mistake) |
| `tests/normal test.py`, `tests/abnormal test.py` | Per-class "accuracy" on `../ear` using an oral fold model |
| `TensorBoard/tfboard1.py` | Logs per-video prediction scalars to TensorBoard |
| `class_imbalance_check.py` | Counts Normal / Abnormal in the oral labels |
| `rename.py` | Renames videos to `video-N.mp4` (starting at 664) and maps free-text labels to Normal/Abnormal (merges a new data batch) |
| `gui/app.py`, `gui/index.html` | pywebview desktop app: upload a video, choose ear/oral, get Normal/Abnormal |
| `Task - 1 EAR/ear/` | **874 mp4 + labels.json** (643 N / 231 A) |
| `Task - 2 ORAL/oral/` | **535 mp4 + labels.json** (452 N / 83 A) |
| `test/` | 50 ear videos, ages 3–6, labels Normal / "Wax is present in ear canal…" |
| `*_models/`, `models/`, `minLSTM/`, `gui/` | About 30 `.h5` files, each ~30 MB (the full model including the MobileNetV2 weights) |

Environment (`venv`): Python 3.11.5, TensorFlow 2.17.0, Keras 3.6.0, OpenCV 4.10.0, scikit-learn
1.5.2, NumPy 1.26.4. `gui/requirements.txt` is UTF-16 encoded, so `pip -r` may choke on it.

## 2. What we reuse

- **The data** (ear as primary), unchanged. We copy or symlink it into `data/raw/ear/`.
- **The architecture idea**, kept exactly as our `lstm_base` preset:
  frozen MobileNetV2 → GAP → LSTM(256) → LSTM(128) → Dense(64) → Dropout(0.5) → sigmoid.
- **Uniform `linspace` frame sampling with T = 20** as the default sequence length.
- The 5-fold CV habit (made stratified).

We do **not** reuse any trained `.h5` weights. They were trained with the preprocessing bugs below,
and their evaluation numbers are unreliable (§3.4).

## 3. Issues found (and the fix in this repo)

Severity is how much each issue would invalidate a comparison.

### 3.1 [High] Wrong input scaling for MobileNetV2
`frame / 255.0` gives `[0, 1]`. Keras' ImageNet MobileNetV2 expects
`mobilenet_v2.preprocess_input`, which gives `[-1, 1]`. Every activation in the frozen backbone is
off-distribution.
**Fix:** `preprocess_input` in `src/data/extract_features.py`.

### 3.2 [High] BGR fed as RGB
`cv2.VideoCapture.read()` returns BGR, and it is never converted. The ImageNet weights expect RGB.
For otoscopy (red/orange-dominated tympanic membrane) the channel swap is not harmless.
**Fix:** `cv2.cvtColor(..., COLOR_BGR2RGB)`.

### 3.3 [High] About 21 GB float64 in-memory dataset
All videos are loaded into one `np.array` of `frame / 255.0`, which is float64.
Ear at T = 20: 874 × 20 × 224 × 224 × 3 × 8 B ≈ **21 GB**. The 16 GB laptop either swaps heavily
or crashes, and this probably explains slow or failed runs.
**Fix:** uint8 frames, streamed per video into the backbone. Only the 1280-d features are kept
(≈ 179 MB at T = 40).

### 3.4 [High] Evaluation leakage in `ConfusionMatrix.py` / `tfboard1.py` / `tests/*`
These scripts run one fold's model over **every** video in `labels.json`, and about 80% of those
videos were in that model's training set. The "overall accuracy" they print is mostly training
accuracy. Missing videos were also counted as predicted Abnormal.
**Fix:** metrics only on held-out outer folds, with out-of-fold predictions saved per video.

### 3.5 [Medium] Non-stratified KFold, no separate validation, accuracy-only metric
- `KFold` (not stratified) on 26%-Abnormal ear and 15.5%-Abnormal oral data gives uneven folds.
- The "validation" fold is also the reported test fold. There is no early stopping, just a fixed
  30 epochs, so there is no selection leakage, but there is no tuning hold-out either.
- Only `accuracy` is tracked. The majority-class baseline is 73.6% for ear and 84.5% for oral.
- No class weights.

**Fix:** `StratifiedKFold`, an inner 15% val for early stopping, AUROC as primary metric, balanced
class weights.

### 3.6 [Medium] Silent sample dropping
`if len(frames) == NUM_FRAMES` drops any video where a seek or read failed, and the count is never
logged. With `NUM_FRAMES = 50` (`dl_model_2_2.0.py`), short oral clips (min 24 frames) would
always be dropped.
**Fix:** sequential decode, repeat-pad, and logged counts.

### 3.7 [Medium] Label polarity
`Normal = 1`. Every "sensitivity" or "recall" computed by the old code measures Normal detection,
which is clinically inverted.
**Fix:** `Abnormal = 1` throughout.

### 3.8 [Low] "MiniLSTM" is not minLSTM
The custom cell in `miniLSTM_miniGRU.py` / `minLSTM*.py` computes the full LSTM equations: input,
forget and output gates, all conditioned on h_{t−1}, plus tanh. It is mathematically a standard
LSTM (≈ the same parameter count) run through a slower Python-level `RNN` wrapper. It is **not**
the minLSTM of Feng et al. (2024), which drops h_{t−1} from the gates and the output gate. Its
results cannot be read as "minLSTM vs LSTM". Also, the two copies differ: one uses `Dense`
sub-layers and the other `add_weight`. Saved models load only with the matching class.
**Fix:** not carried over. A true minLSTM is an optional baseline (`lstm_design.md` §3).

### 3.9 [Low] Wrong model or data pairings in utility scripts
- `cli_ear_model.py` loads `oral_video_classifier_..._fold-5.h5`.
- `tests/*.py` evaluate an *oral* model on `../ear`.
- In `tests/normal test.py`, the `models = [...]` list has no commas, so it is one concatenated
  string.
- `tfboard1.py` points at a non-existent `DL_test1/ear` folder.

### 3.10 [Low] Engineering hygiene
- About 10 copies of `load_video` / `preprocess_video`.
- Hard-coded absolute Windows paths.
- No seeds.
- Legacy `.h5` save format.
- Model files embed the full frozen backbone, so each is about 30 MB for 1.78 M trainable params.
- `cap.set(POS_FRAMES)` seeking is slow and inexact on H.264.
- 480×640 portrait frames are squashed to 224×224. This one is kept for continuity and documented.

**Fix:** a single `src/data/video_io.py`, configs, seeds, `.keras` format, and head-only weights.

## 4. Implications for the study

- Legacy accuracy numbers, wherever they were printed, are **not a valid baseline**: they come from
  leaky evaluation with wrong preprocessing. Our `lstm_base` re-run on correct features is the
  baseline.
- **Fixing preprocessing may raise *both* families' accuracy.** That is fine. The comparison is
  relative.
- The single most useful change for feasibility is **caching features**. The legacy loop re-ran
  MobileNetV2 on 20 × 224² frames per clip, per epoch, per fold. We run it once.
