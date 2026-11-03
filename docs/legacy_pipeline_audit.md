# Legacy Pipeline Audit

**Subject:** the earlier MobileNetV2 + LSTM screening project this study builds on (local only;
not in this repo).
**Audited:** 23 Oct 2026; reframed for the ear dataset on 3 Nov 2026.
**Purpose:** record what the previous LSTM project did, what we reuse, and which bugs this repo
fixes. Read this before touching the data or comparing to the old numbers.

---

## 1. What the legacy project is

An earlier project, from Oct 2024 to Jan 2025. It classified pediatric screening videos as
**Normal / Abnormal** with a frozen MobileNetV2 and an LSTM, for two tasks: **ear (otoscopy)**,
which is the dataset of this study, and a second, smaller task on another body site. It also
shipped a desktop GUI (pywebview) that loads one model per task.

### Inventory

| Path | What it is |
|---|---|
| `Task - 1 EAR/ear/` | **874 mp4 + labels.json** (643 N / 231 A). The source of `data/raw/` in this repo |
| `dl_model_2_new_data_ear.py` | **Ear training script.** MobileNetV2 (frozen) → LSTM(256) → LSTM(128) → Dense(64) → Dropout(0.5) → sigmoid, single 80/20 split |
| `miniLSTM_miniGRU.py` | Ear, 5-fold, custom `MiniLSTM` cell (see §3.8). Despite the name, no GRU |
| `dl_model_2.py`, `dl_model_2_2.0.py` | The same model for the second task: 20 frames with `KFold(5, shuffle, rs=42)`, 30 epochs, batch 16; and a 50-frame, single-split variant |
| `dl_model_1.py` | First prototype: 64×64 frames, first 30 frames, tiny `TimeDistributed(Conv2D(32))` → LSTM(100). Training call commented out |
| `minLSTM_new_data.py`, `minLSTM/minLSTM.py` | `MiniLSTM` training on the second task with an **unshuffled** split, and a serializable copy of the cell for loading saved models |
| `*/ConfusionMatrix.py` (3 copies) | Load one model, predict **every** video in `labels.json`, plot the confusion matrix |
| `Task - 1 EAR/cli_ear_model.py` | Ear CLI predictor (loads the wrong task's model, §3.9) |
| `tests/normal test.py`, `tests/abnormal test.py` | Per-class "accuracy" on the ear folder, using the wrong task's fold model |
| `TensorBoard/tfboard1.py` | Logs per-video prediction scalars to TensorBoard |
| `class_imbalance_check.py` | Counts Normal / Abnormal in a labels file |
| `rename.py` | Renames videos to `video-N.mp4` and maps free-text labels to Normal/Abnormal (merges a new data batch) |
| `gui/app.py`, `gui/index.html` | pywebview desktop app: upload a video, choose a task, get Normal/Abnormal |
| `test/` | 50 external ear videos, ages 3–6, labels Normal / free-text findings. **Not used** in this study |
| `*_models/`, `models/`, `minLSTM/`, `gui/` | About 30 `.h5` files, each ~30 MB (the full model including the MobileNetV2 weights) |

Environment (`venv`): Python 3.11.5, TensorFlow 2.17.0, Keras 3.6.0, OpenCV 4.10.0, scikit-learn
1.5.2, NumPy 1.26.4. `gui/requirements.txt` is UTF-16 encoded, so `pip -r` may choke on it.

## 2. What we reuse

- **The data**: the ear set only, unchanged, copied into `data/raw/`. The external ear test set
  is not used.
- **The architecture idea**, kept exactly as our `lstm_base` preset:
  frozen MobileNetV2 → GAP → LSTM(256) → LSTM(128) → Dense(64) → Dropout(0.5) → sigmoid.
- **Uniform `linspace` frame sampling with T = 20** as the default sequence length.
- The 5-fold CV habit (made stratified).

We do **not** reuse any trained `.h5` weights. They were trained with the preprocessing bugs below,
and their evaluation numbers are unreliable (§3.4).

## 3. Issues found (and the fix in this repo)

Severity is how much each issue would invalidate a comparison. The issues apply to the ear scripts
and to the shared helper code.

### 3.1 [High] Wrong input scaling for MobileNetV2
`frame / 255.0` gives `[0, 1]`. Keras' ImageNet MobileNetV2 expects
`mobilenet_v2.preprocess_input`, which gives `[-1, 1]`. Every activation in the frozen backbone is
off-distribution.
**Fix:** `preprocess()` in `src/common.py`.

### 3.2 [High] BGR fed as RGB
`cv2.VideoCapture.read()` returns BGR, and it is never converted. The ImageNet weights expect RGB.
For otoscopy video, where redness of the canal and eardrum is a clinical sign, the channel swap is
not harmless.
**Fix:** `cv2.cvtColor(..., COLOR_BGR2RGB)` in `sample_frames()`.

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

### 3.5 [Medium] Non-stratified folds, no separate validation, accuracy-only metric
- `KFold` (not stratified) or a single random split on 26%-Abnormal ear data gives uneven folds.
- The "validation" split is also the reported test split. There is no early stopping, just a fixed
  epoch count, so there is no selection leakage, but there is no tuning hold-out either.
- Only `accuracy` is tracked. The majority-class baseline for ear is 73.6%.
- No class weights.

**Fix:** `StratifiedKFold`, an inner 15% val for early stopping, AUROC as primary metric, balanced
class weights.

### 3.6 [Medium] Silent sample dropping
`if len(frames) == NUM_FRAMES` drops any video where a seek or read failed, and the count is never
logged. One ear video reports fewer than 20 header frames, and four fewer than 40, so they would
be dropped silently at those lengths, besides any clip where a seek failed.
**Fix:** sequential decode, repeat-pad, and logged counts.

### 3.7 [Medium] Label polarity
`Normal = 1`. Every "sensitivity" or "recall" computed by the old code measures Normal detection,
which is clinically inverted.
**Fix:** `Abnormal = 1` throughout.

### 3.8 [Low] "MiniLSTM" is not minLSTM
The custom cell in `miniLSTM_miniGRU.py` / `minLSTM*.py` computes the full LSTM equations: input,
forget and output gates, all conditioned on h_{t−1}, plus tanh. It is mathematically a standard
LSTM (≈ the same parameter count) run through a slower Python-level `RNN` wrapper. It is **not**
the minLSTM of Feng et al. (2024), which drops h_{t−1} from the gates and the output gate. Its ear
results cannot be read as "minLSTM vs LSTM". Also, the two copies differ: one uses `Dense`
sub-layers and the other `add_weight`. Saved models load only with the matching class.
**Fix:** not carried over. A true minLSTM is an optional baseline (`lstm_design.md` §3).

### 3.9 [Low] Wrong model or data pairings in utility scripts
- `cli_ear_model.py` loads a model trained for the other task, not an ear model.
- `tests/*.py` evaluate that other task's model on the ear folder.
- In `tests/normal test.py`, the `models = [...]` list has no commas, so it is one concatenated
  string.
- `tfboard1.py` points at a non-existent ear folder.

### 3.10 [Low] Engineering hygiene
- About 10 copies of `load_video` / `preprocess_video`.
- Hard-coded absolute Windows paths.
- No seeds.
- Legacy `.h5` save format.
- Model files embed the full frozen backbone, so each is about 30 MB for 1.78 M trainable params.
- `cap.set(POS_FRAMES)` seeking is slow and inexact on H.264.
- 480×640 portrait frames are squashed to 224×224. This one is kept for continuity and documented.

**Fix:** one shared `src/common.py`, settings as constants, seeds, `.keras` format, and head-only
weights.

## 4. Implications for the study

- Legacy accuracy numbers, wherever they were printed, are **not a valid baseline**: they come from
  leaky evaluation with wrong preprocessing. Our `lstm_base` re-run on correct features is the
  baseline.
- **Fixing preprocessing may raise *both* families' accuracy.** That is fine. The comparison is
  relative.
- The single most useful change for feasibility is **caching features**. The legacy loop re-ran
  MobileNetV2 on 20 × 224² frames per clip, per epoch, per fold. We run it once.
