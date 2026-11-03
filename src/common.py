"""Everything shared by lstm.py, transformer.py and hybrid.py.

Sections, in the order the study runs:
    1. Settings        paths and fixed numbers (no config files; edit here)
    2. Data            labels, frame sampling, MobileNetV2 feature cache, splits, loaders
    3. Model parts     the input and the classifier head every model shares
    4. Metrics         AUROC (primary) and a wide set of secondary scores
    5. Logging         console log files, event journal, live run status, crash-safe writes
    6. Training        one run per (preset, T, seed, fold), logged extensively
    7. Compute         latency, throughput, memory, size, MACs on the CPU
    8. Summary         mean/sd tables from the results CSVs
    9. Command line    the CLI each model file uses, plus this file's own commands

Crash safety: every command writes its console output to a new file in results/logs/; every event
(run start, each epoch, run end or failure, each benchmark job) is appended to results/events.jsonl
and flushed to disk immediately; each training run keeps a live status.json and writes history.csv
one epoch at a time. If a run dies, its files stay on disk; the next attempt moves them to
<run>/attempts/attempt<N>_<status>/ and starts new files.

This file's own commands (run from the repo root):
    python -m src.common extract                 # build data/features/ear_T40.npz (about 6 min)
    python -m src.common splits                  # write data/splits/ear_seed{0,1,2}.json
    python -m src.common backbone --T 20         # MobileNetV2 + decode cost (end-to-end part)
    python -m src.common benchmark-all           # compute grid, one fresh process per model; resumable
    python -m src.common summary                 # mean/sd tables + Markdown report from results/*.csv
    python -m src.common graphs                  # compute-vs-accuracy figures -> graphs/
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import psutil

# =============================================================================================
# 1. Settings
# =============================================================================================

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"                # labels.json + video-*.mp4 (never committed)
FEATURES_DIR = ROOT / "data" / "features"      # ear_T40.npz (never committed)
SPLITS_DIR = ROOT / "data" / "splits"          # ear_seed{0,1,2}.json (IDs only, committed)
RESULTS_DIR = ROOT / "results"
BENCH_DIR = RESULTS_DIR / "benchmark"
LOG_DIR = RESULTS_DIR / "logs"                 # one console log per command (not committed)

DATASET = "ear"
NORMAL_VALUE = "Normal"                        # anything else is Abnormal = 1

CACHE_FRAMES = 40                              # extract once; T = 20 is [:, ::2], T = 10 is [:, ::4]
SEQUENCE_LENGTHS = (10, 20, 40)
FRAME_SIZE = (224, 224)
FEATURE_DIM = 1280                             # MobileNetV2 global-average-pooled output
MAX_CONSECUTIVE_FAILURES = 25                  # tolerate a corrupt packet mid-file; keep this

SEEDS = (0, 1)
N_FOLDS = 3
VAL_FRACTION = 0.15

# Shared by both families. Each model file adds its own optimiser settings (RECIPE).
TRAINING = dict(
    loss="binary_crossentropy",
    class_weight="balanced",
    batch_size=32,
    max_epochs=100,
    clipnorm=1.0,
    early_stopping_patience=15,                # on exact val AUROC, best weights restored
    plateau_factor=0.5,
    plateau_patience=5,
    warmup_epochs=0,
    weight_decay=0.0,
)

BENCHMARK = dict(
    warmup_calls=50,
    timed_calls=300,
    batch_sizes=(1, 32),
    threads=(1, 8),
    sessions=3,
)

FAMILIES = ("baseline", "lstm", "transformer", "hybrid")   # module names in src/

# Screening priority: missing an Abnormal clip costs more than a false alarm. Every run reports
# sensitivity at a fixed 90% specificity (read off the ROC curve) and an operating point whose
# cut-off is chosen on the inner val set to reach 95% sensitivity, then applied unchanged to test.
FIXED_SPECIFICITY = 0.90
HIGH_SENSITIVITY_TARGET = 0.95


# =============================================================================================
# 2. Data
# =============================================================================================

def load_labels(path=None):
    """ids (str), labels (int8, Abnormal = 1) and ages (int16), sorted by video number.

    The legacy code used Normal = 1; do not copy that. Free-text findings count as Abnormal.
    """
    path = path or RAW_DIR / "labels.json"
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    number = lambda v: int(m.group(1)) if (m := re.search(r"(\d+)", v)) else -1
    records = sorted(records, key=lambda r: (number(r["id"]), r["id"]))
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate ids in {path}")
    return {
        "ids": np.array(ids),
        "labels": np.array([0 if str(r["interpretation"]).strip() == NORMAL_VALUE else 1
                            for r in records], dtype=np.int8),
        "ages": np.array([int(r.get("age", -1)) for r in records], dtype=np.int16),
    }


def target_indices(n_frames, num_frames):
    """Uniformly spaced frame indices; repeats indices (repeat-pad) when the clip is short."""
    return np.linspace(0, n_frames - 1, num_frames).round().astype(int)


def _decode_pass(path, wanted, size):
    """Decode the whole file in order, keeping the wanted frames as RGB uint8.

    Sequential grab()/retrieve() instead of seeking: seeking in H.264 is slow and inexact.
    The file is always read to the end so the true frame count is known even if the header lies.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    kept, idx, misses = {}, 0, 0
    try:
        while misses < MAX_CONSECUTIVE_FAILURES:
            if not cap.grab():
                misses += 1
                continue
            misses = 0
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    kept[idx] = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            idx += 1
    finally:
        cap.release()
    return kept, idx


def sample_frames(path, num_frames=CACHE_FRAMES, size=FRAME_SIZE):
    """Return (uint8 frames (T, 224, 224, 3) RGB, meta dict). Short clips are padded, never dropped."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    passes, indices = 1, target_indices(max(header, 1), num_frames)
    kept, decoded = _decode_pass(path, set(indices.tolist()), size)
    if decoded == 0:
        raise IOError(f"no frames decoded: {path}")
    if decoded != header or len(kept) < len(set(indices.tolist())):
        # Header frame count was wrong: resample over the true count.
        passes, indices = 2, target_indices(decoded, num_frames)
        kept, _ = _decode_pass(path, set(indices.tolist()), size)
        missing = set(indices.tolist()) - kept.keys()
        if missing:
            raise IOError(f"{len(missing)} frames failed to decode on pass 2: {path}")

    meta = dict(header_frames=header, decoded_frames=decoded, fps=round(fps, 3), width=width,
                height=height, padded=bool(decoded < num_frames), passes=passes,
                indices=indices.tolist())
    return np.stack([kept[i] for i in indices]), meta


def build_backbone():
    """Frozen ImageNet MobileNetV2 with global average pooling: (224, 224, 3) -> 1280."""
    import keras

    backbone = keras.applications.MobileNetV2(input_shape=(*FRAME_SIZE, 3), include_top=False,
                                              weights="imagenet", pooling="avg")
    backbone.trainable = False
    return backbone


def preprocess(frames_uint8):
    """MobileNetV2 preprocessing to [-1, 1]. NOT /255 (a legacy bug)."""
    import keras

    return keras.applications.mobilenet_v2.preprocess_input(frames_uint8.astype(np.float32))


def extract_features(frames=CACHE_FRAMES, limit=None, workers=4, overwrite=False):
    """Cache frozen MobileNetV2 embeddings of every video to data/features/ear_T{frames}.npz.

    Extract once: the backbone is frozen and nothing is augmented, so running it inside every model
    on every epoch gives identical outputs at about 100x the cost. Videos decode in a thread pool
    (OpenCV releases the GIL) while the main thread runs the backbone. Any failure aborts before
    writing, so the cache always covers all of labels.json.
    """
    import keras

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{DATASET}_T{frames}" + (f"_smoke{limit}" if limit else "")
    npz_path, meta_path = FEATURES_DIR / f"{stem}.npz", FEATURES_DIR / f"{stem}.meta.json"
    if npz_path.exists() and not overwrite:
        raise FileExistsError(f"{npz_path} exists; pass --overwrite to rebuild it")

    table = load_labels()
    ids = table["ids"][:limit] if limit else table["ids"]
    missing = [v for v in ids if not (RAW_DIR / v).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos in labels.json are missing, e.g. {missing[:3]}")

    backbone = build_backbone()
    backbone.predict_on_batch(np.zeros((frames, *FRAME_SIZE, 3), np.float32))  # trace once
    features = np.zeros((len(ids), frames, FEATURE_DIM), np.float32)
    per_video, failures, backbone_ms = {}, {}, []

    def load(video_id):
        try:
            return sample_frames(RAW_DIR / video_id, frames), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    print(f"extracting {len(ids)} videos at T={frames} with {workers} decode workers")
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (video_id, (out, err)) in enumerate(zip(ids, pool.map(load, ids))):
            if err:
                failures[video_id] = err
                print(f"  [fail] {video_id}: {err}", file=sys.stderr)
                continue
            clip, meta = out
            x = preprocess(clip)
            t0 = time.perf_counter()
            features[i] = backbone.predict_on_batch(x)
            backbone_ms.append((time.perf_counter() - t0) * 1000 / frames)
            per_video[video_id] = meta
            if meta["padded"] or meta["passes"] > 1:
                print(f"  [note] {video_id}: header {meta['header_frames']}, decoded {meta['decoded_frames']}")
            if (i + 1) % 25 == 0 or i + 1 == len(ids):
                elapsed = time.perf_counter() - start
                print(f"  {i + 1}/{len(ids)}  {elapsed / 60:.1f} min", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} videos failed: {sorted(failures)}; nothing written")

    tmp = npz_path.with_suffix(".tmp.npz")
    np.savez(tmp, features=features, labels=table["labels"][: len(ids)], ids=ids,
             ages=table["ages"][: len(ids)])
    os.replace(tmp, npz_path)
    lat = np.array(backbone_ms)
    write_json(meta_path, {
        "dataset": DATASET, "num_videos": len(ids), "num_abnormal": int(table["labels"][: len(ids)].sum()),
        "frames": frames, "feature_dim": FEATURE_DIM, "label_polarity": "Abnormal = 1, Normal = 0",
        "backbone": {"name": "MobileNetV2", "weights": "imagenet", "pooling": "avg", "trainable": False,
                     "params": int(backbone.count_params()), "keras_version": keras.__version__},
        "preprocessing": {"sampling": "uniform linspace, sequential decode", "color": "BGR -> RGB",
                          "resize": "224x224 INTER_AREA stretch", "normalisation": "preprocess_input [-1, 1]",
                          "short_video": "repeat_pad"},
        "timing": {"total_minutes": round((time.perf_counter() - start) / 60, 2), "decode_workers": workers,
                   "backbone_ms_per_frame_median": round(float(np.median(lat)), 3),
                   "backbone_ms_per_frame_p95": round(float(np.percentile(lat, 95)), 3),
                   "backbone_batch": frames,
                   "note": "measured while decode threads run; indicative only"},
        "num_padded": sum(m["padded"] for m in per_video.values()),
        "num_header_mismatch": sum(m["passes"] > 1 for m in per_video.values()),
        "created_utc": now_utc(), "videos": per_video,
    }, indent=1)
    print(f"wrote {npz_path} {features.shape}")
    return npz_path


def make_split(ids, labels, seed, n_splits=N_FOLDS, val_fraction=VAL_FRACTION):
    """Outer StratifiedKFold(5, shuffle) for testing; inner stratified 15% val for early stopping only."""
    from sklearn.model_selection import StratifiedKFold, train_test_split

    ids, labels = np.asarray(ids), np.asarray(labels)
    position = {v: i for i, v in enumerate(ids.tolist())}
    folds = []
    for train_val, test in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(ids, labels):
        train, val = train_test_split(train_val, test_size=val_fraction, stratify=labels[train_val],
                                      random_state=seed)
        folds.append({part: sorted(ids[idx].tolist(), key=position.__getitem__)
                      for part, idx in (("train", train), ("val", val), ("test", test))})
    return {"seed": seed, "n_splits": n_splits, "val_fraction": val_fraction, "folds": folds}


def write_splits(seeds=SEEDS):
    """Write data/splits/ear_seed{s}.json. Every model reads the same files (paired comparison)."""
    table = load_labels()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    label_of = dict(zip(table["ids"].tolist(), table["labels"].tolist()))
    for seed in seeds:
        split = make_split(table["ids"], table["labels"], seed)
        split["dataset"] = DATASET
        write_json(SPLITS_DIR / f"{DATASET}_seed{seed}.json", split, indent=1)
        print(f"seed {seed}:")
        for k, fold in enumerate(split["folds"]):
            print(f"  fold {k}: " + ", ".join(
                f"{p} {len(fold[p])} ({sum(label_of[v] for v in fold[p])} abn)" for p in ("train", "val", "test")))


def load_features(num_frames=20):
    """features float32 (N, T, 1280), labels, ids, ages, plus a fingerprint of the cache file."""
    if CACHE_FRAMES % num_frames:
        raise ValueError(f"T={num_frames} must divide the cached T={CACHE_FRAMES}")
    path = FEATURES_DIR / f"{DATASET}_T{CACHE_FRAMES}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: python -m src.common extract")
    with np.load(path) as data:
        stride = CACHE_FRAMES // num_frames
        out = {"features": np.ascontiguousarray(data["features"][:, ::stride]),
               "labels": data["labels"], "ids": data["ids"], "ages": data["ages"]}
    stat = path.stat()
    out["source"] = {"features_file": path.name, "features_bytes": stat.st_size,
                     "features_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                     .isoformat(timespec="seconds"), "stride": stride}
    return out


def load_split(seed=0):
    path = SPLITS_DIR / f"{DATASET}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: python -m src.common splits")
    with open(path, encoding="utf-8") as fh:
        split = json.load(fh)
    split["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    return split


def fold_indices(ids, fold):
    """Map one fold's id lists to row indices into the feature arrays."""
    position = {v: i for i, v in enumerate(np.asarray(ids).tolist())}
    return {part: np.array([position[v] for v in fold[part]]) for part in ("train", "val", "test")}


# =============================================================================================
# 3. Model parts shared by every architecture
# =============================================================================================

def feature_input(num_frames, feature_dim=FEATURE_DIM):
    import keras

    return keras.Input(shape=(num_frames, feature_dim), name="frame_features")


def classifier_head(x, hidden_units=64, dropout=0.5, name="head"):
    """Dense(64, relu) -> Dropout(0.5) -> Dense(1, sigmoid). Identical in every model, so accuracy
    differences can be attributed to the temporal block alone."""
    from keras import layers

    x = layers.Dense(hidden_units, activation="relu", name=f"{name}_dense")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return layers.Dense(1, activation="sigmoid", name=f"{name}_out")(x)


def head_macs(in_dim, hidden_units=64):
    return in_dim * hidden_units + hidden_units


# =============================================================================================
# 4. Metrics
# =============================================================================================

def threshold_metrics(y_true, y_prob, threshold):
    """Every cut-off-dependent score. Abnormal = 1, so sensitivity is Abnormal recall."""
    from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef

    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else float("nan")
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    return {
        "threshold": float(threshold),
        "accuracy": div(tp + tn, len(y_true)),
        "balanced_accuracy": (sens + spec) / 2,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_abnormal": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0,
        "predicted_abnormal": int(y_pred.sum()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def youden_threshold(y_true, y_prob):
    """Cut-off maximising sensitivity + specificity - 1. Chosen on val, never on test."""
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).astype(int)
    if y_true.min() == y_true.max():
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(np.clip(thr[int(np.argmax(tpr - fpr))], 0.0, 1.0))


def sensitivity_threshold(y_true, y_prob, target=HIGH_SENSITIVITY_TARGET):
    """Highest cut-off whose sensitivity reaches target. Chosen on val, never on test, so the
    test sensitivity it gives is an honest estimate (it can land below target on new clips)."""
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).astype(int)
    if y_true.min() == y_true.max():
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(np.clip(thr[int(np.argmax(tpr >= target))], 0.0, 1.0))


def roc_operating_points(y_true, y_prob):
    """Sensitivity at fixed specificity and specificity at fixed sensitivity, read off this set's
    own ROC curve (threshold-free summaries, like AUROC). Sensitivity at 90% specificity is the
    headline screening number: of the Abnormal clips, how many are caught while only 1 in 10
    Normal clips is flagged."""
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).astype(int)
    if y_true.min() == y_true.max():
        return {"sens_at_spec90": float("nan"), "spec_at_sens90": float("nan"),
                "spec_at_sens95": float("nan")}
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    best = lambda vals, ok: float(vals[ok].max()) if ok.any() else 0.0
    return {"sens_at_spec90": best(tpr, spec >= FIXED_SPECIFICITY - 1e-12),
            "spec_at_sens90": best(spec, tpr >= 0.90 - 1e-12),
            "spec_at_sens95": best(spec, tpr >= 0.95 - 1e-12)}


def evaluate(y_true, y_prob, youden=None, high_sens=None):
    """AUROC (primary), AUPRC, Brier, log loss, sensitivity at 90% specificity, and threshold scores
    at 0.5 (plus the val-chosen Youden and high-sensitivity cut-offs if given).

    Ranking metrics use the raw probabilities; only log loss sees clipped values, because clipping
    would merge saturated predictions into ties and change the AUROC.
    """
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    both = y_true.min() != y_true.max()
    out = {
        "n": int(len(y_true)),
        "n_abnormal": int(y_true.sum()),
        "auroc": float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if both else float("nan"),
        "prevalence": float(y_true.mean()),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "mean_prob_abnormal": float(y_prob[y_true == 1].mean()) if y_true.any() else float("nan"),
        "mean_prob_normal": float(y_prob[y_true == 0].mean()) if (y_true == 0).any() else float("nan"),
        **roc_operating_points(y_true, y_prob),
        "at_0.5": threshold_metrics(y_true, y_prob, 0.5),
    }
    if youden is not None:
        out["at_youden"] = threshold_metrics(y_true, y_prob, youden)
    if high_sens is not None:
        out["at_high_sens"] = threshold_metrics(y_true, y_prob, high_sens)
    return out


# =============================================================================================
# 5. Logging
# =============================================================================================

def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _plain(obj):
    """Make an object JSON-safe: numpy scalars to Python, NaN/inf to null."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _fsync(fh):
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except OSError:
        pass


def _replace(tmp, path, attempts=20, wait=0.1):
    """os.replace with retries. On Windows a virus scanner, the search indexer or an editor can hold
    the target open for a moment, and the rename then fails with PermissionError (seen on 3 Nov:
    one run lost its final status write this way). Waits up to about 2 s before giving up."""
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(wait)


def write_json(path, obj, indent=2):
    """Atomic write (temp file + rename), so a crash never leaves a half-written JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_plain(obj), fh, indent=indent)
        _fsync(fh)
    _replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def write_csv(path, rows):
    """Atomic CSV write; columns are the union of every row's keys, in first-seen order."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([_plain(r) for r in rows])
        _fsync(fh)
    _replace(tmp, path)


def append_csv_row(path, row):
    """Append one row and flush it to disk now (history.csv grows epoch by epoch)."""
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row), extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(_plain(row))
        _fsync(fh)


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


_CSV_LOCK = threading.Lock()


def upsert_csv(path, row, key):
    """Add a row to a results CSV, replacing any earlier row with the same key (a re-run)."""
    with _CSV_LOCK:
        rows = [r for r in read_csv(path) if r.get(key) != str(row[key])]
        write_csv(path, rows + [_plain(row)])


def flatten(record, prefix="", skip=()):
    """Nested dict -> one flat dict with dotted keys (for a single CSV row)."""
    flat = {}
    for k, v in record.items():
        if k in skip:
            continue
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(flatten(v, f"{name}.", ()))
        else:
            flat[name] = v
    return flat


def log_event(root, kind, **fields):
    """Append one JSON line to <root>/events.jsonl and flush it to disk immediately.

    The journal is append-only, so it survives any crash and holds the full timeline of the study.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_plain({"time_utc": now_utc(), "kind": kind, "pid": os.getpid(), **fields}))
    with _CSV_LOCK, open(root / "events.jsonl", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        _fsync(fh)


class _Tee:
    """Write to the real stream and to the log file; the file is flushed on every write."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        try:
            self.fh.write(s)
            self.fh.flush()
        except ValueError:
            pass
        return len(s)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_console_log(tag):
    """Copy everything printed (stdout and stderr, tracebacks included) into a NEW file
    results/logs/<utc time>_<tag>_<pid>.log. Every command gets its own file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = LOG_DIR / f"{stamp}_{re.sub(r'[^A-Za-z0-9_.-]', '_', tag)}_{os.getpid()}.log"
    fh = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.stdout, fh), _Tee(sys.stderr, fh)
    print(f"# log {path.name}\n# started {now_utc()}\n# command: {' '.join(sys.argv)}\n# cwd: {os.getcwd()}")
    return path


def environment():
    """Hardware, software and git state, stored with every result so numbers stay traceable."""
    import keras
    import sklearn
    import tensorflow as tf

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        except Exception:
            return ""

    vm = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    return {
        "platform": platform.platform(), "processor": platform.processor(),
        "cpu_physical": psutil.cpu_count(logical=False), "cpu_logical": psutil.cpu_count(),
        "cpu_freq_mhz": round(freq.current) if freq else None,
        "ram_gb": round(vm.total / 2**30, 2), "ram_available_gb": round(vm.available / 2**30, 2),
        "python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
        "numpy": np.__version__, "sklearn": sklearn.__version__, "opencv": cv2.__version__,
        "intra_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "command": " ".join(sys.argv),
    }


class RunStatus:
    """Live status.json for one run: running / completed / failed / interrupted, updated each epoch."""

    def __init__(self, out_dir, **info):
        self.path = Path(out_dir) / "status.json"
        self.data = {**info, "status": "running", "started_utc": now_utc(), "updated_utc": now_utc(),
                     "pid": os.getpid(), "epochs_done": 0}
        self.save()

    def update(self, **fields):
        self.data.update(fields, updated_utc=now_utc())
        self.save()

    def save(self):
        write_json(self.path, self.data)


def archive_previous_attempt(out_dir):
    """Move whatever an earlier attempt left in out_dir into attempts/attempt<N>_<status>/.

    Nothing is ever deleted: a crashed run's history, status and logs stay available for analysis.
    A status still saying "running" means the process died without cleaning up, so it is filed as
    "crashed". Returns the number of the new attempt.
    """
    out_dir = Path(out_dir)
    attempts = out_dir / "attempts"
    previous = len(list(attempts.glob("attempt*"))) if attempts.exists() else 0
    leftovers = [p for p in out_dir.iterdir() if p.name != "attempts"] if out_dir.exists() else []
    if not leftovers:
        return previous + 1
    status = (read_json(out_dir / "status.json") or {}).get("status", "unknown")
    status = "crashed" if status == "running" else status
    dest = attempts / f"attempt{previous + 1}_{status}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in leftovers:
        shutil.move(str(p), str(dest / p.name))
    return previous + 2


# =============================================================================================
# 6. Training
# =============================================================================================

def balanced_class_weight(labels):
    """n / (2 * count): about 0.59 Normal / 3.22 Abnormal on the full set."""
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2)
    return {c: float(len(labels) / (2 * counts[c])) for c in (0, 1) if counts[c]}


def full_recipe(recipe):
    """Shared TRAINING settings overlaid with a model file's RECIPE."""
    return {**TRAINING, **recipe}


def make_optimizer(recipe, learning_rate):
    """Adam for the LSTM; AdamW (no decay on biases, LayerNorm or embeddings) for attention models."""
    import keras

    if recipe["optimizer"] == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=recipe["clipnorm"])
    if recipe["optimizer"] == "adamw":
        opt = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=recipe["weight_decay"],
                                     clipnorm=recipe["clipnorm"])
        opt.exclude_from_weight_decay(var_names=["bias", "gamma", "beta", "embeddings"])
        return opt
    raise ValueError(f"unsupported optimizer {recipe['optimizer']!r}")


def compile_model(build, preset, num_frames, recipe, learning_rate):
    """Build by preset name and compile with several training-time metrics (all logged per epoch)."""
    import keras

    model = build(preset, num_frames)
    model.compile(
        optimizer=make_optimizer(recipe, learning_rate),
        loss=recipe["loss"],
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(curve="PR", name="auprc"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


class PeakMemory:
    """Context manager: samples this process's RSS in a background thread and keeps the peak (MB)."""

    def __init__(self, interval=0.005):
        self.interval, self.peak_mb, self.start_mb = interval, 0.0, 0.0
        self._stop, self._proc = threading.Event(), psutil.Process()

    def _run(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 2**20)
            time.sleep(self.interval)

    def __enter__(self):
        self.start_mb = self.peak_mb = self._proc.memory_info().rss / 2**20
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, self._proc.memory_info().rss / 2**20)


def _current_lr(model):
    import keras

    return float(keras.ops.convert_to_numpy(model.optimizer.learning_rate))


def _callbacks(recipe, learning_rate, out_dir, status, x_val, y_val, events_root):
    """Warm-up (attention models), early stopping, plateau LR cuts, and the per-epoch logger."""
    import keras

    class LinearWarmup(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if epoch < recipe["warmup_epochs"]:
                self.model.optimizer.learning_rate = learning_rate * (epoch + 1) / recipe["warmup_epochs"]

    class EpochLog(keras.callbacks.Callback):
        """After every epoch, immediately writes to disk:
        - one history.csv row: every Keras metric (train and val), the LR used in this epoch,
          exact sklearn val scores, timing split into train / val / val-predict, per-step time
          percentiles, RSS, global weight norm
        - val_probs_per_epoch.npy: the validation probabilities after each epoch
        - status.json (epochs done, best so far) and one events.jsonl line
        """

        def __init__(self):
            super().__init__()
            self.rows, self.val_probs, self.best = [], [], -np.inf
            self._proc = psutil.Process()

        def on_epoch_begin(self, epoch, logs=None):
            self._lr = _current_lr(self.model)   # after the warm-up callback, before any plateau cut
            self._t0, self._steps, self._val_s = time.perf_counter(), [], 0.0

        def on_train_batch_begin(self, batch, logs=None):
            self._tb = time.perf_counter()

        def on_train_batch_end(self, batch, logs=None):
            self._steps.append((time.perf_counter() - self._tb) * 1000)

        def on_test_begin(self, logs=None):
            self._tv = time.perf_counter()

        def on_test_end(self, logs=None):
            self._val_s += time.perf_counter() - self._tv

        def on_epoch_end(self, epoch, logs=None):
            epoch_s = time.perf_counter() - self._t0
            t0 = time.perf_counter()
            probs = self.model.predict(x_val, batch_size=256, verbose=0).ravel()
            predict_s = time.perf_counter() - t0
            exact = evaluate(y_val, probs)
            if logs is not None:
                # Early stopping and the LR plateau read this key. Keras's own val_auc is a
                # 200-threshold approximation that drifts badly when predictions bunch together.
                logs["val_auroc_exact"] = exact["auroc"]
            steps = np.array(self._steps) if self._steps else np.array([np.nan])
            weights = self.model.trainable_weights
            row = {"epoch": epoch + 1, "lr": self._lr}
            row.update({k: float(v) for k, v in (logs or {}).items() if k not in ("learning_rate", "lr")})
            row.update({
                "val_auroc_exact": exact["auroc"], "val_auprc_exact": exact["auprc"],
                "val_sens_at_spec90": exact["sens_at_spec90"],
                "val_brier": exact["brier"], "val_log_loss_exact": exact["log_loss"],
                "val_sensitivity_0.5": exact["at_0.5"]["sensitivity"],
                "val_specificity_0.5": exact["at_0.5"]["specificity"],
                "val_balanced_accuracy_0.5": exact["at_0.5"]["balanced_accuracy"],
                "val_mean_prob_abnormal": exact["mean_prob_abnormal"],
                "val_mean_prob_normal": exact["mean_prob_normal"],
                "epoch_seconds": round(epoch_s, 4),
                "train_seconds": round(float(np.nansum(steps)) / 1000, 4),
                "val_seconds": round(self._val_s, 4),
                "val_predict_seconds": round(predict_s, 4),
                "steps": len(self._steps),
                "step_ms_median": round(float(np.nanmedian(steps)), 3),
                "step_ms_p95": round(float(np.nanpercentile(steps, 95)), 3),
                "step_ms_max": round(float(np.nanmax(steps)), 3),
                "rss_mb": round(self._proc.memory_info().rss / 2**20, 1),
                "weight_norm": round(float(np.sqrt(sum(float(np.sum(np.square(
                    keras.ops.convert_to_numpy(w)))) for w in weights))), 4),
                "time_utc": now_utc(),
            })
            self.rows.append(row)
            append_csv_row(out_dir / "history.csv", row)
            self.val_probs.append(probs.astype(np.float32))
            np.save(out_dir / "val_probs_per_epoch.npy", np.stack(self.val_probs))
            self.best = max(self.best, row.get("val_auroc_exact", -np.inf))
            status.update(epochs_done=epoch + 1, best_val_auroc_exact_so_far=self.best,
                          last_epoch={k: row.get(k) for k in ("loss", "val_loss", "val_auc",
                                                              "val_auroc_exact", "lr")})
            log_event(events_root, "epoch_end", run=out_dir.name, epoch=epoch + 1, loss=row.get("loss"),
                      val_loss=row.get("val_loss"), val_auc=row.get("val_auc"),
                      val_auroc_exact=row.get("val_auroc_exact"), lr=self._lr,
                      epoch_seconds=row["epoch_seconds"])

    # EpochLog must come before EarlyStopping and ReduceLROnPlateau: it puts val_auroc_exact
    # into the shared logs dict that they monitor.
    log = EpochLog()
    early = keras.callbacks.EarlyStopping(monitor="val_auroc_exact", mode="max", restore_best_weights=True,
                                          patience=recipe["early_stopping_patience"])
    callbacks = [
        log,
        early,
        keras.callbacks.ReduceLROnPlateau(monitor="val_auroc_exact", mode="max",
                                          factor=recipe["plateau_factor"],
                                          patience=recipe["plateau_patience"]),
    ]
    if recipe["warmup_epochs"]:
        callbacks.insert(0, LinearWarmup())
    return callbacks, log, early


def train_fold(build, family, preset, num_frames, seed, fold, data, idx, recipe, out_dir,
               learning_rate=None, max_epochs=None, verbose=2, variant="default"):
    """Train one fold, score val and test once, and write the run folder. Returns the run record.

    Written to out_dir while the run is going (so a crash loses nothing already done):
        status.json               live: running / completed / failed / interrupted, epochs done, error
        model_summary.txt         layer table;  model_architecture.json  full Keras config
        history.csv               one row per epoch, flushed as it happens
        val_probs_per_epoch.npy   validation probabilities after every epoch
    Written at the end:
        run.json                  everything (settings, data, environment, timing, memory, all metrics)
        predictions.csv           probability for every clip in train, val and test
        model.keras               best weights (gitignored)
    plus one flat row in results/runs.csv and events in results/events.jsonl.
    An earlier attempt's files are first moved to out_dir/attempts/ (see archive_previous_attempt).
    """
    import keras

    out_dir = Path(out_dir)
    root = out_dir.parent
    attempt = archive_previous_attempt(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe = full_recipe(recipe)
    lr = float(learning_rate if learning_rate is not None else recipe["learning_rate"])
    epochs = int(max_epochs if max_epochs is not None else recipe["max_epochs"])
    x, y, ids = data["features"], data["labels"].astype("float32"), data["ids"]
    tr, va, te = idx["train"], idx["val"], idx["test"]
    info = dict(run=out_dir.name, family=family, preset=preset, T=num_frames, seed=seed, fold=fold,
                variant=variant, attempt=attempt)
    status = RunStatus(out_dir, **info, learning_rate=lr, max_epochs=epochs)
    log_event(root, "run_start", **info, learning_rate=lr, max_epochs=epochs)
    started = now_utc()

    try:
        keras.utils.set_random_seed(seed)
        with PeakMemory(interval=0.05) as mem:
            t_build = time.perf_counter()
            model = compile_model(build, preset, num_frames, recipe, lr)
            build_seconds = time.perf_counter() - t_build
            lines = []
            model.summary(line_length=110, print_fn=lambda s, *a, **k: lines.append(s))
            (out_dir / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
            (out_dir / "model_architecture.json").write_text(model.to_json(indent=1), encoding="utf-8")

            callbacks, log, early = _callbacks(recipe, lr, out_dir, status, x[va], y[va], root)
            class_weight = balanced_class_weight(y[tr]) if recipe["class_weight"] == "balanced" else None
            status.update(phase="training")
            t_fit = time.perf_counter()
            model.fit(x[tr], y[tr], validation_data=(x[va], y[va]), epochs=epochs,
                      batch_size=recipe["batch_size"], class_weight=class_weight, callbacks=callbacks,
                      shuffle=True, verbose=verbose)
            fit_seconds = time.perf_counter() - t_fit
            final_lr = _current_lr(model)

            status.update(phase="scoring")
            probs, predict_ms = {}, {}
            for part, rows in (("train", tr), ("val", va), ("test", te)):
                t0 = time.perf_counter()
                probs[part] = model.predict(x[rows], batch_size=256, verbose=0).ravel()
                predict_ms[part] = (time.perf_counter() - t0) * 1000 / max(len(rows), 1)

        status.update(phase="saving")
        youden = youden_threshold(y[va], probs["val"])
        high_sens = sensitivity_threshold(y[va], probs["val"])
        val_aucs = [r.get("val_auc", float("nan")) for r in log.rows]
        exact_aucs = [r.get("val_auroc_exact", float("nan")) for r in log.rows]
        best_epoch = int(np.nanargmax(exact_aucs)) + 1 if log.rows else 0
        col = lambda k: [r[k] for r in log.rows]
        step_ms = [r["step_ms_median"] for r in log.rows]
        model.save(out_dir / "model.keras")
        best_row = log.rows[best_epoch - 1] if log.rows else {}
        record = {
            **info, "status": "completed",
            "started_utc": started, "finished_utc": now_utc(),
            "recipe": {**recipe, "learning_rate": lr, "max_epochs": epochs},
            "class_weight": class_weight,
            "data": {"n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                     "abnormal_train": int(y[tr].sum()), "abnormal_val": int(y[va].sum()),
                     "abnormal_test": int(y[te].sum()), "feature_dim": int(x.shape[-1]),
                     **data.get("source", {}), "split_md5": data.get("split_md5")},
            "model": model_facts(model, out_dir / "model.keras"),
            "training": {
                "epochs_run": len(log.rows), "best_epoch": best_epoch,
                "stopped_early": bool(early.stopped_epoch),
                "best_val_auc": best_row.get("val_auc"),
                "best_val_auc_keras_max": float(np.nanmax(val_aucs)) if log.rows else None,
                "best_val_auroc_exact": best_row.get("val_auroc_exact"),
                "best_val_loss": best_row.get("val_loss"),
                "train_loss_at_best": best_row.get("loss"),
                "lr_at_best": best_row.get("lr"),
                "final_lr": final_lr,
                "lr_cuts": int(sum(1 for a, b in zip(col("lr"), col("lr")[1:]) if b < a - 1e-12)),
                "build_seconds": round(build_seconds, 3), "fit_seconds": round(fit_seconds, 3),
                "train_only_seconds": round(float(np.sum(col("train_seconds"))), 3),
                "val_seconds": round(float(np.sum(col("val_seconds"))), 3),
                "logging_seconds": round(float(np.sum(col("val_predict_seconds"))), 3),
                "epoch_seconds_mean": round(float(np.mean(col("epoch_seconds"))), 4) if log.rows else None,
                "epoch_seconds_median": round(float(np.median(col("epoch_seconds"))), 4) if log.rows else None,
                "ms_per_train_step_median": round(float(np.median(step_ms)), 3) if step_ms else None,
                "seconds_to_best_epoch": round(float(np.sum(col("epoch_seconds")[:best_epoch])), 3),
                "rss_mb_max_epoch": max(col("rss_mb")) if log.rows else None,
            },
            "memory": {"rss_start_mb": round(mem.start_mb, 1), "rss_peak_mb": round(mem.peak_mb, 1),
                       "rss_increase_mb": round(mem.peak_mb - mem.start_mb, 1)},
            "inference_ms_per_clip_batch256": {k: round(v, 4) for k, v in predict_ms.items()},
            "youden_threshold_from_val": youden,
            "high_sensitivity_threshold_from_val": high_sens,
            "high_sensitivity_target": HIGH_SENSITIVITY_TARGET,
            "metrics": {part: evaluate(y[rows], probs[part], youden, high_sens)
                        for part, rows in (("test", te), ("val", va), ("train", tr))},
            "environment": environment(),
        }
        write_csv(out_dir / "predictions.csv", [
            {"id": str(ids[i]), "part": part, "label": int(y[i]), "prob": float(p),
             "pred_0.5": int(p >= 0.5), "pred_youden": int(p >= youden),
             "pred_high_sens": int(p >= high_sens),
             "preset": preset, "T": num_frames, "seed": seed, "fold": fold, "variant": variant}
            for part, rows in (("test", te), ("val", va), ("train", tr))
            for i, p in zip(rows, probs[part])
        ])
        write_json(out_dir / "run.json", record)
        upsert_csv(root / "runs.csv", flatten(record, skip=("environment", "recipe", "class_weight")),
                   key="run")
        status.update(status="completed", phase="done", finished_utc=record["finished_utc"],
                      test_auroc=record["metrics"]["test"]["auroc"])
        log_event(root, "run_end", **info, test_auroc=record["metrics"]["test"]["auroc"],
                  best_epoch=best_epoch, fit_seconds=round(fit_seconds, 2))
        return record
    except BaseException as exc:
        kind = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        status.update(status=kind, error_type=type(exc).__name__, error=str(exc)[:2000],
                      traceback=traceback.format_exc(), failed_utc=now_utc())
        log_event(root, f"run_{kind}", **info, error_type=type(exc).__name__, error=str(exc)[:500],
                  epochs_done=status.data.get("epochs_done"))
        raise


def model_facts(model, saved_path=None):
    """Parameter counts, per-layer parameters, and on-disk / in-memory size."""
    import keras

    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    facts = {"params": int(model.count_params()), "trainable_params": trainable,
             "non_trainable_params": int(model.count_params()) - trainable,
             "weights_mb_fp32": round(model.count_params() * 4 / 2**20, 3),
             "layers": len(model.layers),
             "params_by_layer": {layer.name: int(layer.count_params()) for layer in model.layers}}
    if saved_path is None:
        with tempfile.TemporaryDirectory() as tmp:
            saved_path = Path(tmp) / "m.keras"
            model.save(saved_path)
            facts["keras_file_mb"] = round(saved_path.stat().st_size / 2**20, 3)
    else:
        facts["keras_file_mb"] = round(Path(saved_path).stat().st_size / 2**20, 3)
    facts["keras_version"] = keras.__version__
    return facts


def run_name(preset, num_frames, seed, fold, variant="default"):
    """Folder name; non-default variants (e.g. an LR check) get their own folders."""
    base = f"{preset}_T{num_frames}_s{seed}_f{fold}"
    return base if variant == "default" else f"{base}_{variant}"


def variant_name(recipe, lr=None, epochs=None):
    """'default', or a tag such as 'lr0.0003' / 'ep20' when the recipe is overridden."""
    parts = []
    if lr is not None and float(lr) != float(recipe["learning_rate"]):
        parts.append(f"lr{lr:g}")
    if epochs is not None and int(epochs) != int(recipe["max_epochs"]):
        parts.append(f"ep{epochs}")
    return "_".join(parts) or "default"


# =============================================================================================
# 7. Compute
# =============================================================================================

def set_threads(intra, inter=1):
    """Must run before TensorFlow executes anything, which is why each benchmark is its own process."""
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(intra)
    tf.config.threading.set_inter_op_parallelism_threads(inter)


def time_calls(fn, x, warmup, timed):
    """Run fn(x) warmup times untimed, then timed times; return per-call milliseconds."""
    for _ in range(warmup):
        fn(x)
    out = np.empty(timed)
    for i in range(timed):
        t0 = time.perf_counter()
        fn(x)
        out[i] = (time.perf_counter() - t0) * 1000
    return out


def latency_stats(ms, batch):
    q5, q25, q50, q75, q95, q99 = np.percentile(ms, [5, 25, 50, 75, 95, 99])
    return {"median_ms": round(q50, 4), "mean_ms": round(float(ms.mean()), 4), "std_ms": round(float(ms.std()), 4),
            "min_ms": round(float(ms.min()), 4), "max_ms": round(float(ms.max()), 4),
            "p5_ms": round(q5, 4), "p25_ms": round(q25, 4), "p75_ms": round(q75, 4),
            "p95_ms": round(q95, 4), "p99_ms": round(q99, 4), "iqr_ms": round(q75 - q25, 4),
            "ms_per_clip": round(q50 / batch, 4), "clips_per_second": round(batch * 1000 / q50, 2)}


def bench_name(preset, num_frames, batch, threads, session):
    return f"{preset}_T{num_frames}_b{batch}_th{threads}_r{session}"


def benchmark_model(build, family, preset, num_frames, macs_fn, session=0,
                    batch_sizes=BENCHMARK["batch_sizes"], warmup=BENCHMARK["warmup_calls"],
                    timed=BENCHMARK["timed_calls"], out_dir=None):
    """Head-only CPU benchmark of one preset: compiled tf.function, training=False, random input.

    Each batch size gets its own tf.function (a fixed-shape graph). Logs per batch size: the full
    latency distribution with every raw timing, throughput, peak RSS while timing, RSS before and
    after building, first-call (trace) time, params, file size, analytic MACs/FLOPs, CPU load.
    Threads must already be set (set_threads) by the caller.
    """
    import tensorflow as tf

    out_dir = Path(out_dir or BENCH_DIR)
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / 2**20
    t0 = time.perf_counter()
    model = build(preset, num_frames)
    build_ms = (time.perf_counter() - t0) * 1000
    rss_built = proc.memory_info().rss / 2**20
    facts = model_facts(model)
    macs = int(macs_fn(preset, num_frames))
    threads = tf.config.threading.get_intra_op_parallelism_threads()
    rows = []
    for batch in batch_sizes:
        infer = tf.function(lambda x: model(x, training=False))
        x = tf.constant(np.random.default_rng(0).normal(size=(batch, num_frames, FEATURE_DIM)).astype("float32"))
        t0 = time.perf_counter()
        infer(x)
        first_call_ms = (time.perf_counter() - t0) * 1000
        psutil.cpu_percent(None)
        with PeakMemory() as mem:
            ms = time_calls(infer, x, warmup, timed)
        row = {
            "kind": "head", "family": family, "preset": preset, "T": num_frames, "batch": batch,
            "intra_threads": threads,
            "inter_threads": tf.config.threading.get_inter_op_parallelism_threads(),
            "session": session, "warmup_calls": warmup, "timed_calls": timed,
            **latency_stats(ms, batch),
            "first_call_ms": round(first_call_ms, 3), "build_ms": round(build_ms, 3),
            "params": facts["params"], "weights_mb_fp32": facts["weights_mb_fp32"],
            "keras_file_mb": facts["keras_file_mb"],
            "macs_per_clip": macs, "gflops_per_clip": round(2 * macs / 1e9, 6),
            "rss_before_build_mb": round(rss_before, 1), "rss_after_build_mb": round(rss_built, 1),
            "rss_peak_timing_mb": round(mem.peak_mb, 1),
            "system_cpu_percent_during": psutil.cpu_percent(None),
            "measured_utc": now_utc(),
        }
        rows.append(row)
        name = bench_name(preset, num_frames, batch, threads, session)
        write_json(out_dir / f"{name}.json", {**row, "raw_ms": np.round(ms, 4).tolist(),
                                              "environment": environment()})
        upsert_csv(out_dir.parent / "benchmark.csv", {"run": name, **row}, key="run")
        log_event(out_dir.parent, "benchmark", run=name, median_ms=row["median_ms"], p95_ms=row["p95_ms"])
        print(f"{name}: median {row['median_ms']:.3f} ms, p95 {row['p95_ms']:.3f} ms, "
              f"{row['clips_per_second']:.1f} clips/s, peak RSS {row['rss_peak_timing_mb']:.0f} MB")
    return rows


def benchmark_backbone(num_frames=20, n_videos=10, session=0, warmup=10, timed=50, out_dir=None):
    """The part every model shares end-to-end: decode + sample T frames, then MobileNetV2 on T frames.

    Add this to a head's batch-1 latency to get the end-to-end cost of one clip.
    Threads must already be set (set_threads) by the caller.
    """
    import tensorflow as tf

    out_dir = Path(out_dir or BENCH_DIR)
    backbone = build_backbone()
    infer = tf.function(lambda x: backbone(x, training=False))
    x = tf.constant(preprocess(np.zeros((num_frames, *FRAME_SIZE, 3), np.uint8)))
    infer(x)
    with PeakMemory() as mem:
        ms = time_calls(infer, x, warmup, timed)
    decode_ms, preprocess_ms = [], []
    for video_id in load_labels()["ids"][:n_videos]:
        t0 = time.perf_counter()
        clip, _ = sample_frames(RAW_DIR / video_id, num_frames)
        t1 = time.perf_counter()
        preprocess(clip)
        decode_ms.append((t1 - t0) * 1000)
        preprocess_ms.append((time.perf_counter() - t1) * 1000)
    threads = tf.config.threading.get_intra_op_parallelism_threads()
    row = {"kind": "backbone", "family": "mobilenetv2", "preset": "mobilenetv2_backbone", "T": num_frames,
           "batch": 1, "intra_threads": threads,
           "inter_threads": tf.config.threading.get_inter_op_parallelism_threads(), "session": session,
           "warmup_calls": warmup, "timed_calls": timed, **latency_stats(ms, 1),
           "ms_per_frame": round(float(np.median(ms)) / num_frames, 4),
           "decode_ms_median": round(float(np.median(decode_ms)), 2),
           "decode_ms_p95": round(float(np.percentile(decode_ms, 95)), 2),
           "preprocess_ms_median": round(float(np.median(preprocess_ms)), 3),
           "decode_videos": len(decode_ms),
           "params": int(backbone.count_params()), "rss_peak_timing_mb": round(mem.peak_mb, 1),
           "measured_utc": now_utc()}
    name = bench_name("mobilenetv2_backbone", num_frames, 1, threads, session)
    write_json(out_dir / f"{name}.json", {**row, "raw_ms": np.round(ms, 4).tolist(),
                                          "decode_raw_ms": decode_ms, "environment": environment()})
    upsert_csv(out_dir.parent / "benchmark.csv", {"run": name, **row}, key="run")
    log_event(out_dir.parent, "benchmark", run=name, median_ms=row["median_ms"])
    print(f"{name}: backbone median {row['median_ms']:.1f} ms per clip, decode median {row['decode_ms_median']:.0f} ms")
    return row


def benchmark_jobs(families=FAMILIES, sequence_lengths=SEQUENCE_LENGTHS, threads=BENCHMARK["threads"],
                   sessions=BENCHMARK["sessions"], backbone=True, presets=None):
    """Every (command, expected output files) pair of the grid, shuffled within each session."""
    jobs = []
    for session in range(sessions):
        batch = []
        for family in families:
            module = __import__(f"src.{family}", fromlist=["PRESETS"])
            for preset in module.PRESETS:
                if presets and preset not in presets:
                    continue
                for t in sequence_lengths:
                    for n in threads:
                        batch.append({
                            "args": ["-m", f"src.{family}", "--preset", preset, "--T", str(t), "--benchmark",
                                     "--threads", str(n), "--session", str(session)],
                            "outputs": [bench_name(preset, t, b, n, session) for b in BENCHMARK["batch_sizes"]]})
        if backbone:
            batch += [{"args": ["-m", "src.common", "backbone", "--T", str(t), "--threads", str(n),
                                "--session", str(session)],
                       "outputs": [bench_name("mobilenetv2_backbone", t, 1, n, session)]}
                      for t in sequence_lengths for n in threads]
        random.Random(session).shuffle(batch)
        jobs += batch
    return jobs


def benchmark_all(sessions=BENCHMARK["sessions"], backbone=True, force=False, presets=None,
                  sequence_lengths=SEQUENCE_LENGTHS):
    """Compute grid: one fresh Python process per job. Resumable: jobs whose result files already
    exist are skipped (unless force). A failing job is logged and the grid carries on.
    presets / sequence_lengths narrow the grid (for example to the models actually trained)."""
    jobs = benchmark_jobs(sequence_lengths=sequence_lengths, sessions=sessions, backbone=backbone,
                          presets=presets)
    done = lambda job: all((BENCH_DIR / f"{n}.json").exists() for n in job["outputs"])
    todo = [j for j in jobs if force or not done(j)]
    print(f"{len(jobs)} benchmark jobs; {len(jobs) - len(todo)} already done; {len(todo)} to run")
    log_event(RESULTS_DIR, "grid_start", jobs=len(jobs), todo=len(todo))
    failed = []
    for i, job in enumerate(todo, 1):
        cmd = " ".join(job["args"])
        print(f"[{i}/{len(todo)}] python {cmd}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.run([sys.executable, *job["args"]], cwd=ROOT).returncode
        log_event(RESULTS_DIR, "grid_job", command=cmd, returncode=rc,
                  seconds=round(time.perf_counter() - t0, 2))
        if rc:
            failed.append(cmd)
            print(f"  [fail] exit code {rc}; continuing", file=sys.stderr)
    write_json(RESULTS_DIR / "benchmark_grid_status.json", {
        "finished_utc": now_utc(), "jobs": len(jobs), "ran": len(todo), "failed": failed,
        "complete": not failed and all(done(j) for j in jobs)})
    log_event(RESULTS_DIR, "grid_end", ran=len(todo), failed=len(failed))
    print(f"grid finished: {len(todo) - len(failed)} ok, {len(failed)} failed")
    if failed:
        sys.exit(1)


# =============================================================================================
# 8. Summary
# =============================================================================================

def summarise(results_dir=RESULTS_DIR):
    """Mean and sd per (preset, T, variant) over folds and seeds; median per benchmark cell."""
    results_dir = Path(results_dir)
    groups = {}
    for row in read_csv(results_dir / "runs.csv"):
        groups.setdefault((row["preset"], int(row["T"]), row.get("variant") or "default"), []).append(row)
    cols = ["metrics.test.auroc", "metrics.test.auprc", "metrics.test.brier",
            "metrics.test.sens_at_spec90", "metrics.test.spec_at_sens95",
            "metrics.test.at_high_sens.sensitivity", "metrics.test.at_high_sens.specificity",
            "metrics.test.at_high_sens.fn",
            "metrics.test.at_0.5.macro_f1", "metrics.test.at_0.5.balanced_accuracy",
            "metrics.test.at_0.5.sensitivity", "metrics.test.at_0.5.specificity",
            "metrics.test.at_youden.balanced_accuracy", "metrics.test.at_youden.sensitivity",
            "metrics.test.at_youden.specificity", "training.fit_seconds", "training.epochs_run",
            "training.best_epoch", "memory.rss_peak_mb", "model.params"]
    acc = []
    for (preset, t, variant), rows in sorted(groups.items()):
        out = {"preset": preset, "T": t, "variant": variant, "runs": len(rows)}
        for c in cols:
            vals = np.array([float(r[c]) for r in rows if r.get(c) not in (None, "")])
            out[f"{c}.mean"] = round(float(vals.mean()), 4) if len(vals) else None
            out[f"{c}.sd"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else None
        acc.append(out)
    write_csv(results_dir / "summary_accuracy.csv", acc)

    cells = {}
    for row in read_csv(results_dir / "benchmark.csv"):
        cells.setdefault((row["preset"], row["T"], row["batch"], row["intra_threads"]), []).append(row)
    comp = []
    for (preset, t, b, th), rows in sorted(cells.items()):
        out = {"preset": preset, "T": t, "batch": b, "intra_threads": th, "sessions": len(rows)}
        for c in ("median_ms", "p95_ms", "iqr_ms", "clips_per_second", "rss_peak_timing_mb"):
            vals = [float(r[c]) for r in rows if r.get(c) not in (None, "")]
            out[c] = round(float(np.median(vals)), 4) if vals else None
        out["params"] = rows[0].get("params")
        out["gflops_per_clip"] = rows[0].get("gflops_per_clip")
        comp.append(out)
    write_csv(results_dir / "summary_compute.csv", comp)
    print(f"{len(acc)} accuracy rows -> summary_accuracy.csv; {len(comp)} compute rows -> summary_compute.csv")
    for r in acc:
        print(f"  {r['preset']:<20} T={r['T']:<3} {r['variant']:<10} runs={r['runs']:<3} AUROC "
              f"{r['metrics.test.auroc.mean']} +/- {r['metrics.test.auroc.sd']}")
    write_report(results_dir)


# Readable Markdown report --------------------------------------------------------------------

REFERENCE_PRESET = "lstm_base"     # every model is compared with this one (paired, fold by fold)
EQUIVALENCE_MARGIN = 0.03          # pre-registered TOST margin on AUROC
FINDINGS_START, FINDINGS_END = "<!-- findings:start -->", "<!-- findings:end -->"
_THRESHOLD_METRICS = [("Accuracy", "accuracy"), ("Balanced acc.", "balanced_accuracy"),
                      ("Precision (PPV)", "ppv"), ("Recall (sensitivity)", "sensitivity"),
                      ("Specificity", "specificity"), ("F1 (Abnormal)", "f1_abnormal"),
                      ("Macro-F1", "macro_f1"), ("NPV", "npv"), ("MCC", "mcc")]
_CUTS = [
    ("at_0.5", "Cut-off 0.5 (default)",
     "The model's own decision point: probability >= 0.5 means Abnormal (training used balanced class weights)."),
    ("at_youden", "Youden cut-off (chosen on validation)",
     "The cut-off that maximises sensitivity + specificity on the inner validation set, applied unchanged to test."),
    ("at_high_sens", "High-sensitivity cut-off (chosen on validation for 95% sensitivity)",
     "The highest cut-off that catches 95% of Abnormal clips on the inner validation set, applied unchanged to test."),
]


def report_path(results_dir=RESULTS_DIR, sequence_lengths=None):
    """results/results_<folds>fold_T<T>.md, e.g. results_3fold_T20.md (T values joined by '-')."""
    t = "-".join(str(x) for x in sorted(sequence_lengths)) if sequence_lengths else "all"
    return Path(results_dir) / f"results_{N_FOLDS}fold_T{t}.md"


def _preset_order():
    """Presets in family order (baseline, lstm, transformer, hybrid), then file order."""
    order = []
    for family in FAMILIES:
        order += list(__import__(f"src.{family}", fromlist=["PRESETS"]).PRESETS)
    return order


def _macs_of(family, preset, num_frames):
    module = __import__(f"src.{family}", fromlist=["macs"])
    return module.macs(preset, num_frames)


def paired_vs_reference(rows, presets, metric="metrics.test.auroc", reference=REFERENCE_PRESET):
    """Corrected resampled t-test (Nadeau and Bengio) and TOST of each preset against the reference,
    paired by (seed, fold). rows: completed runs of one T. Shared by the report and the graphs."""
    from scipy import stats

    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
    by = {(r["preset"], int(r["seed"]), int(r["fold"])): r for r in rows}
    ratio = float(np.mean([f(r, "data.n_test") / (f(r, "data.n_train") + f(r, "data.n_val")) for r in rows]))
    out = []
    for p in presets:
        if p == reference:
            continue
        d = np.array([f(by[k], metric) - f(by[(reference, k[1], k[2])], metric)
                      for k in by if k[0] == p and (reference, k[1], k[2]) in by])
        if len(d) < 2:
            continue
        n, se = len(d), d.std(ddof=1) * np.sqrt(1 / len(d) + ratio)
        t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
        out.append({"preset": p, "n": n, "mean": float(d.mean()), "ratio": ratio,
                    "ci95": (d.mean() - t95 * se, d.mean() + t95 * se),
                    "ci90": (d.mean() - t90 * se, d.mean() + t90 * se),
                    "p": float(2 * stats.t.sf(abs(d.mean() / se), n - 1)) if se > 0 else float("nan"),
                    "equivalent": bool(d.mean() - t90 * se > -EQUIVALENCE_MARGIN
                                       and d.mean() + t90 * se < EQUIVALENCE_MARGIN)})
    return out


def write_report(results_dir=RESULTS_DIR):
    """Write the readable Markdown report from runs.csv, summary_compute.csv and the benchmark files.

    Every table is regenerated from the result files, so the report never drifts from the data.
    The hand-written findings live between FINDINGS_START and FINDINGS_END and are copied over
    unchanged on every run; edit them in the report itself. Only completed default-variant runs
    are used.
    """
    results_dir = Path(results_dir)
    rows = [r for r in read_csv(results_dir / "runs.csv")
            if r.get("status") == "completed" and (r.get("variant") or "default") == "default"]
    if not rows:
        print("no completed runs; report not written")
        return None
    order = _preset_order()
    rows.sort(key=lambda r: (int(r["T"]), order.index(r["preset"]) if r["preset"] in order else 99,
                             int(r["seed"]), int(r["fold"])))
    t_values = sorted({int(r["T"]) for r in rows})
    path = report_path(results_dir, t_values)
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if FINDINGS_START in old and FINDINGS_END in old:
        findings = old[old.index(FINDINGS_START):old.index(FINDINGS_END) + len(FINDINGS_END)]
    else:
        findings = (f"{FINDINGS_START}\n## Findings\n\nWrite the findings here. This block is kept "
                    f"unchanged when the report is regenerated.\n\n{FINDINGS_END}")

    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
    ms = lambda v, d=3: (f"{np.nanmean(v):.{d}f} +/- {np.nanstd(v, ddof=1):.{d}f}" if len(v) > 1
                         else f"{np.nanmean(v):.{d}f}")
    seeds = sorted({int(r["seed"]) for r in rows})
    span = lambda k: (lambda v: f"{int(min(v))}" if min(v) == max(v) else f"{int(min(v))}–{int(max(v))}")(
        [f(r, k) for r in rows])
    comp = {(r["preset"], int(r["T"]), r["batch"], r["intra_threads"]): r
            for r in read_csv(results_dir / "summary_compute.csv")}

    out = []
    w = out.append
    w(f"# Results: {N_FOLDS}-fold cross-validation, {len(seeds)} seed(s), T = {', '.join(map(str, t_values))}")
    w("")
    w(f"{len(rows)} completed training runs on the {DATASET} dataset. Generated by `python -m src.common summary` "
      f"from `results/runs.csv`, `results/summary_compute.csv` and `results/benchmark/`; each run's full "
      f"record is in `results/<run>/run.json`. Last generated {now_utc()}.")
    w("")
    w("## Setup")
    w("")
    w("| Item | Value |")
    w("|---|---|")
    w(f"| Data | {DATASET} videos, frozen MobileNetV2 features, T = {', '.join(map(str, t_values))} frames per clip |")
    w(f"| Folds | Stratified {N_FOLDS}-fold CV, seeds {', '.join(map(str, seeds))} "
      f"(`data/splits/{DATASET}_seed*.json`) |")
    w(f"| Fold sizes | train {span('data.n_train')} ({span('data.abnormal_train')} Abnormal), inner val "
      f"{span('data.n_val')} ({span('data.abnormal_val')} Abnormal), test {span('data.n_test')} "
      f"({span('data.abnormal_test')} Abnormal) |")
    w("| Training | Shared head, balanced class weights, batch 32, at most 100 epochs, early stopping on "
      "exact val AUROC (patience 15, best weights restored) |")
    w(f"| Runs | {len(rows)} completed |")
    w("")
    w("Metric notes (all on the held-out test fold):")
    w("")
    w("- **AUROC**: primary metric. 0.5 is chance, 1.0 is perfect.")
    w("- **Sens@Spec90**: share of Abnormal clips caught while only 10% of Normal clips are flagged "
      "(read off the ROC curve).")
    w("- **High-sensitivity cut-off**: chosen on the inner validation set to catch 95% of Abnormal clips, "
      "then applied unchanged to test. **Missed** = Abnormal clips it still misses per test fold.")
    w("- **MACs**: analytic multiply-accumulates per clip for the temporal model and head; the shared "
      "MobileNetV2 backbone is not included.")
    w("")
    w(findings)
    w("")

    for t in t_values:
        tr = [r for r in rows if int(r["T"]) == t]
        presets = [p for p in order if any(r["preset"] == p for r in tr)]
        family = {r["preset"]: r["family"] for r in tr}
        col = lambda p, k: [f(r, k) for r in tr if r["preset"] == p]
        sfx = f" (T = {t})"

        w(f"## Summary per model{sfx}, mean +/- sd over folds and seeds")
        w("")
        w("| Model | Runs | AUROC | Sens@Spec90 | AUPRC | Sens at high-sens cut-off | Spec at high-sens cut-off | "
          "Missed | Epochs | Fit s | Params | MACs |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for p in presets:
            w(f"| `{p}` | {len(col(p, 'metrics.test.auroc'))} | {ms(col(p, 'metrics.test.auroc'))} | "
              f"{ms(col(p, 'metrics.test.sens_at_spec90'))} | {ms(col(p, 'metrics.test.auprc'))} | "
              f"{ms(col(p, 'metrics.test.at_high_sens.sensitivity'))} | "
              f"{ms(col(p, 'metrics.test.at_high_sens.specificity'))} | "
              f"{ms(col(p, 'metrics.test.at_high_sens.fn'), 1)} | {ms(col(p, 'training.epochs_run'), 1)} | "
              f"{np.nanmean(col(p, 'training.fit_seconds')):.0f} | {int(col(p, 'model.params')[0]):,} | "
              f"{_macs_of(family[p], p, t) / 1e6:.2f} M |")
        w("")

        bench = {p: comp.get((p, t, "1", "1")) for p in presets}
        if any(bench.values()):
            cm = lambda p, b, th: float(comp[(p, t, b, th)]["median_ms"]) if (p, t, b, th) in comp else float("nan")
            w(f"## Compute: CPU latency, throughput and memory{sfx}")
            w("")
            w("One fresh process per measurement, compiled `tf.function`, random input (latency does not depend "
              f"on the trained weights), {BENCHMARK['warmup_calls']} warm-up and {BENCHMARK['timed_calls']} timed "
              "calls; median over sessions of each session's median. Temporal model and head only; the shared "
              "front end is listed separately.")
            w("")
            w("| Model | Sessions | Batch 1, 1 thread (ms) | Batch 1, 8 threads (ms) | Batch 32, 1 thread (ms per batch) | "
              "Batch 32, 8 threads (ms per batch) | Best throughput (clips/s) | Peak RAM (MB) | "
              f"vs `{REFERENCE_PRESET}` at batch 1, 1 thread |")
            w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            ref = cm(REFERENCE_PRESET, "1", "1")
            for p in presets:
                if not bench[p]:
                    continue
                cells = [comp[k] for k in comp if k[0] == p and k[1] == t]
                best = max(float(c["clips_per_second"]) for c in cells)
                ram = max(float(c["rss_peak_timing_mb"]) for c in cells)
                speed = ("reference" if p == REFERENCE_PRESET else
                         f"{ref / cm(p, '1', '1'):.1f}x faster" if ref >= cm(p, "1", "1") else
                         f"{cm(p, '1', '1') / ref:.1f}x slower")
                w(f"| `{p}` | {bench[p]['sessions']} | {cm(p, '1', '1'):.2f} | {cm(p, '1', '8'):.2f} | "
                  f"{cm(p, '32', '1'):.2f} | {cm(p, '32', '8'):.2f} | {best:,.0f} | {ram:.0f} | {speed} |")
            w("")
            bb = [read_json(q) for q in BENCH_DIR.glob(f"mobilenetv2_backbone_T{t}_*.json")]
            bb = [b for b in bb if b]
            if bb and ("mobilenetv2_backbone", t, "1", "1") in comp:
                decode = float(np.median([b["decode_ms_median"] for b in bb]))
                prep = float(np.median([b["preprocess_ms_median"] for b in bb]))
                bb1, bb8 = cm("mobilenetv2_backbone", "1", "1"), cm("mobilenetv2_backbone", "1", "8")
                w(f"Shared front end per clip (identical for every model, T = {t}):")
                w("")
                w("| Step | 1 thread (ms) | 8 threads (ms) |")
                w("|---|---:|---:|")
                w(f"| Decode the video and sample {t} frames (OpenCV) | {decode:.0f} | {decode:.0f} |")
                w(f"| Preprocess {t} frames | {prep:.0f} | {prep:.0f} |")
                w(f"| MobileNetV2 on {t} frames | {bb1:.0f} | {bb8:.0f} |")
                w(f"| **Front end total** | **{decode + prep + bb1:.0f}** | **{decode + prep + bb8:.0f}** |")
                w("")
                w("End-to-end latency per clip at batch 1 (front end + temporal model and head):")
                w("")
                w("| Model | 1 thread (ms) | Temporal part, 1 thread | 8 threads (ms) | Temporal part, 8 threads |")
                w("|---|---:|---:|---:|---:|")
                for p in presets:
                    if not bench[p]:
                        continue
                    h1, h8 = cm(p, "1", "1"), cm(p, "1", "8")
                    e1, e8 = decode + prep + bb1 + h1, decode + prep + bb8 + h8
                    w(f"| `{p}` | {e1:.0f} | {100 * h1 / e1:.1f}% | {e8:.0f} | {100 * h8 / e8:.1f}% |")
                w("")

        if REFERENCE_PRESET in presets:
            ratio = float(np.mean([f(r, "data.n_test") / (f(r, "data.n_train") + f(r, "data.n_val")) for r in tr]))
            w(f"## Paired comparison with `{REFERENCE_PRESET}` (AUROC){sfx}")
            w("")
            w(f"Corrected resampled t-test (Nadeau and Bengio) over paired folds, n_test/n_train = {ratio:.2f}. "
              f"Equivalence (TOST) holds when the 90% CI lies inside the pre-registered margin of "
              f"+/- {EQUIVALENCE_MARGIN} AUROC.")
            w("")
            w("| Model | Pairs | Mean difference | 95% CI | p | 90% CI | Equivalent |")
            w("|---|---:|---:|---:|---:|---:|---|")
            for c in paired_vs_reference(tr, presets):
                eq = "Yes" if c["equivalent"] else "No (CI too wide)"
                w(f"| `{c['preset']}` | {c['n']} | {c['mean']:+.4f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                  f"{c['p']:.2f} | [{c['ci90'][0]:+.3f}, {c['ci90'][1]:+.3f}] | {eq} |")
            w("")

        w(f"## Threshold metrics: accuracy, precision, recall, F1 and more{sfx}")
        w("")
        w("These depend on the cut-off that turns a probability into a Normal / Abnormal decision, so they are "
          "given at three cut-offs. Abnormal is the positive class: **precision** = of the clips flagged "
          "Abnormal, how many are; **recall** = sensitivity = of the Abnormal clips, how many are flagged; "
          "**specificity** = of the Normal clips, how many are cleared; **NPV** = of the clips cleared, how many "
          "are truly Normal; **MCC** = Matthews correlation (-1 to 1, robust to imbalance). Plain accuracy is "
          "reported but never used to decide. TP / FP / TN / FN are mean counts per test fold.")
        w("")
        w("Calibration (threshold-free):")
        w("")
        w("| Model | Brier score (lower is better) | Log loss (lower is better) | Mean prob. on Abnormal | "
          "Mean prob. on Normal |")
        w("|---|---:|---:|---:|---:|")
        for p in presets:
            w(f"| `{p}` | {ms(col(p, 'metrics.test.brier'))} | {ms(col(p, 'metrics.test.log_loss'))} | "
              f"{ms(col(p, 'metrics.test.mean_prob_abnormal'))} | {ms(col(p, 'metrics.test.mean_prob_normal'))} |")
        w("")
        for key, title, note in _CUTS:
            w(f"### {title}{sfx}")
            w("")
            w(note)
            w("")
            w("| Model | " + " | ".join(n for n, _ in _THRESHOLD_METRICS) + " | TP | FP | TN | FN |")
            w("|---|" + "---:|" * (len(_THRESHOLD_METRICS) + 4))
            for p in presets:
                c = lambda k: col(p, f"metrics.test.{key}.{k}")
                w(f"| `{p}` | " + " | ".join(ms(c(k)) for _, k in _THRESHOLD_METRICS) + " | "
                  + " | ".join(f"{np.nanmean(c(k)):.1f}" for k in ("tp", "fp", "tn", "fn")) + " |")
            w("")

        w(f"## All runs{sfx}")
        w("")
        w("| Run | Model | Seed | Fold | AUROC | Sens@Spec90 | AUPRC | High-sens cut-off | Sens | Spec | Missed | "
          "Youden sens | Youden spec | Epochs | Best epoch | Best val AUROC | Fit s |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in tr:
            v = lambda k, d=3: f"{f(r, k):.{d}f}"
            w(f"| `{r['run']}` | `{r['preset']}` | {r['seed']} | {r['fold']} | {v('metrics.test.auroc', 4)} | "
              f"{v('metrics.test.sens_at_spec90')} | {v('metrics.test.auprc')} | "
              f"{v('high_sensitivity_threshold_from_val')} | {v('metrics.test.at_high_sens.sensitivity')} | "
              f"{v('metrics.test.at_high_sens.specificity')} | {int(f(r, 'metrics.test.at_high_sens.fn'))} | "
              f"{v('metrics.test.at_youden.sensitivity')} | {v('metrics.test.at_youden.specificity')} | "
              f"{int(f(r, 'training.epochs_run'))} | {int(f(r, 'training.best_epoch'))} | "
              f"{v('training.best_val_auroc_exact')} | {f(r, 'training.fit_seconds'):.0f} |")
        w("")
        w(f"### Threshold metrics per run{sfx}")
        w("")
        w("Acc = accuracy, BA = balanced accuracy, Prec = precision, Rec = recall (sensitivity), "
          "Spec = specificity, F1 = F1 for Abnormal.")
        w("")
        for key, title, _ in _CUTS:
            w(f"#### {title}")
            w("")
            w("| Run | Cut-off | Acc | BA | Prec | Rec | Spec | F1 | Macro-F1 | NPV | MCC | TP | FP | TN | FN |")
            w("|---|" + "---:|" * 14)
            for r in tr:
                v = lambda k: f(r, f"metrics.test.{key}.{k}")
                w(f"| `{r['run']}` | {v('threshold'):.3f} | " + " | ".join(
                    f"{v(k):.3f}" for k in ("accuracy", "balanced_accuracy", "ppv", "sensitivity", "specificity",
                                            "f1_abnormal", "macro_f1", "npv", "mcc"))
                  + " | " + " | ".join(str(int(v(k))) for k in ("tp", "fp", "tn", "fn")) + " |")
            w("")

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(out), encoding="utf-8")
    _replace(tmp, path)
    print(f"report -> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return path


# Graphs ----------------------------------------------------------------------------------------

GRAPHS_DIR = ROOT / "graphs"
# Family colours from the slide template (validated for colour blindness). Orange and green are close
# for red-blind readers, so every chart also uses a marker shape per family and names every point.
# The pooling baselines are controls, not a model family, so they are a neutral grey.
FAMILY_STYLE = {
    "baseline": {"color": "#6B6B6B", "marker": "s", "label": "No-temporal baseline"},
    "lstm": {"color": "#E69138", "marker": "o", "label": "LSTM"},
    "transformer": {"color": "#3C78D8", "marker": "^", "label": "Transformer"},
    "hybrid": {"color": "#6AA84F", "marker": "D", "label": "Hybrid (LSTM + attention)"},
}
_INK, _MUTED, _GRID = "#222222", "#595959", "#E5E5E5"


def _graph_data(results_dir, num_frames):
    """One record per preset at this T: accuracy (mean, sd over folds), compute, family."""
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
    rows = [r for r in read_csv(Path(results_dir) / "runs.csv") if r.get("status") == "completed"
            and (r.get("variant") or "default") == "default" and int(r["T"]) == num_frames]
    comp = {(r["preset"], r["batch"], r["intra_threads"]): r for r in read_csv(Path(results_dir) / "summary_compute.csv")
            if int(r["T"]) == num_frames}
    order = _preset_order()
    presets = [p for p in order if any(r["preset"] == p for r in rows)]
    data = {}
    for p in presets:
        rs = [r for r in rows if r["preset"] == p]
        col = lambda k: np.array([f(r, k) for r in rs])
        lat = lambda b, th: float(comp[(p, b, th)]["median_ms"]) if (p, b, th) in comp else float("nan")
        data[p] = {"family": rs[0]["family"], "runs": len(rs),
                   "auroc": col("metrics.test.auroc").mean(), "auroc_sd": col("metrics.test.auroc").std(ddof=1),
                   "sens90": col("metrics.test.sens_at_spec90").mean(),
                   "sens90_sd": col("metrics.test.sens_at_spec90").std(ddof=1),
                   "lat1": lat("1", "1"), "lat8": lat("1", "8"),
                   "macs": _macs_of(rs[0]["family"], p, num_frames), "params": int(rs[0]["model.params"])}
    return rows, presets, data


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _family_legend(ax, families, loc="lower right"):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], linestyle="none", marker=FAMILY_STYLE[fam]["marker"], markersize=8,
                      markerfacecolor=FAMILY_STYLE[fam]["color"], markeredgecolor="white",
                      label=FAMILY_STYLE[fam]["label"]) for fam in FAMILY_STYLE if fam in families]
    ax.legend(handles=handles, loc=loc, frameon=False, fontsize=9, labelcolor=_INK)


def _pareto_front(points):
    """Indices of the points no other point beats on both axes (lower x, higher y)."""
    front = [i for i, (x, y) in enumerate(points)
             if not any(x2 <= x and y2 >= y and (x2, y2) != (x, y) for x2, y2 in points)]
    return sorted(front, key=lambda i: points[i][0])


def _scatter(ax, data, xkey, ykey, sdkey, label_offsets=None):
    """Points with sd error bars, a dashed Pareto front, and every point named."""
    label_offsets = label_offsets or {}
    presets = list(data)
    pts = [(data[p][xkey], data[p][ykey]) for p in presets]
    front = _pareto_front(pts)
    ax.plot([pts[i][0] for i in front], [pts[i][1] for i in front], linestyle="--", linewidth=1.2,
            color=_MUTED, zorder=1, label="_front")
    for p, (x, y) in zip(presets, pts):
        st = FAMILY_STYLE[data[p]["family"]]
        ax.errorbar(x, y, yerr=data[p][sdkey], fmt="none", ecolor=st["color"], elinewidth=1.2, capsize=0,
                    alpha=0.6, zorder=2)
        ax.scatter(x, y, s=70, marker=st["marker"], color=st["color"], edgecolors="white", linewidths=1.5,
                   zorder=3)
        dx, dy = label_offsets.get(p, (7, 5))
        ax.annotate(p, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=_INK,
                    ha="center" if dx == 0 else "left" if dx > 0 else "right", va="center", zorder=4,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    return front


def plot_graphs(results_dir=RESULTS_DIR, out_dir=GRAPHS_DIR, num_frames=20):
    """Write the compute-vs-accuracy figures (PNG for slides, SVG for print) to graphs/.

    Built from the same files as the report (runs.csv, summary_compute.csv, benchmark/), so the
    figures and the tables always agree. Rerun after `summary` whenever results change.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "axes.titlecolor": _INK, "axes.labelcolor": _INK,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, _, data = _graph_data(results_dir, num_frames)
    if not data:
        print(f"no completed runs at T={num_frames}; no graphs written")
        return []
    families = {d["family"] for d in data.values()}
    written = []

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)
        written.append(out_dir / f"{name}.png")

    have_latency = all(np.isfinite(d["lat1"]) for d in data.values())
    ref = data.get(REFERENCE_PRESET)
    offsets = {"pool_max": (8, -9), "hybrid_base": (0, 12), "transformer_base": (0, -13),
               "hybrid_lite": (-8, 7), "hybrid_attn": (8, 7), "lstm_base": (8, 7), "pool_mean": (8, 7)}
    plain_ms = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")

    def latency_axis(ax):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.2, 0.3, 0.5, 1, 2, 3, 5]))
        ax.xaxis.set_major_formatter(plain_ms)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # 1. Main figure: accuracy vs measured CPU latency.
    if have_latency:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        _style_axes(ax)
        if ref:
            ax.axhspan(ref["auroc"] - EQUIVALENCE_MARGIN, ref["auroc"] + EQUIVALENCE_MARGIN, color="#F2F2F2",
                       zorder=0)
            ax.text(0.99, ref["auroc"] + EQUIVALENCE_MARGIN - 0.002, f"equivalence band: {REFERENCE_PRESET} "
                    f"+/- {EQUIVALENCE_MARGIN}", transform=ax.get_yaxis_transform(), ha="right", va="top",
                    fontsize=8, color=_MUTED)
        _scatter(ax, data, "lat1", "auroc", "auroc_sd", offsets)
        latency_axis(ax)
        ax.set_xlabel("CPU latency per clip, temporal model + head (ms, batch 1, 1 thread, log scale)")
        ax.set_ylabel("Test AUROC (mean, bars = sd over folds)")
        ax.set_title(f"Accuracy vs compute: every model is equally accurate (T = {num_frames})", loc="left")
        _family_legend(ax, families)
        ax.text(0, -0.2, "Dashed line: Pareto front (no other model is both faster and more accurate). "
                f"{len(rows)} runs, {N_FOLDS}-fold CV x {len({r['seed'] for r in rows})} seeds.",
                transform=ax.transAxes, fontsize=8, color=_MUTED)
        save(fig, "01_pareto_auroc_vs_latency")

    # 2. Screening view: sensitivity at 90% specificity vs latency.
    if have_latency:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        _style_axes(ax)
        _scatter(ax, data, "lat1", "sens90", "sens90_sd", offsets)
        latency_axis(ax)
        ax.set_xlabel("CPU latency per clip, temporal model + head (ms, batch 1, 1 thread, log scale)")
        ax.set_ylabel("Sensitivity at 90% specificity (mean, bars = sd)")
        ax.set_title(f"Screening view: Abnormal clips caught at 90% specificity (T = {num_frames})", loc="left")
        _family_legend(ax, families)
        ax.text(0, -0.2, "Share of Abnormal clips caught while only 1 in 10 Normal clips is flagged. "
                "Differences are within fold noise (not significant).", transform=ax.transAxes, fontsize=8,
                color=_MUTED)
        save(fig, "02_pareto_sensitivity_vs_latency")

    # 3. Hardware-independent compute: accuracy vs analytic MACs.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _style_axes(ax)
    _scatter(ax, data, "macs", "auroc", "auroc_sd",
             {**offsets, "hybrid_base": (0, 12), "transformer_base": (0, -13), "lstm_base": (8, 0)})
    ax.set_xscale("log")
    ax.set_xlabel("Multiply-accumulates per clip, temporal model + head (log scale)")
    ax.set_ylabel("Test AUROC (mean, bars = sd)")
    ax.set_title(f"Accuracy vs analytic compute (MACs, T = {num_frames})", loc="left")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v / 1e6:g} M" if v >= 1e6 else f"{v / 1e3:g} k"))
    _family_legend(ax, families)
    save(fig, "03_auroc_vs_macs")

    # 4. Latency per model, 1 and 8 threads.
    if have_latency:
        order = sorted(data, key=lambda p: data[p]["lat1"])
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
        for ax, key, title in ((axes[0], "lat1", "1 thread"), (axes[1], "lat8", "8 threads")):
            _style_axes(ax)
            ax.grid(axis="y", visible=False)
            vals = [data[p][key] for p in order]
            colors = [FAMILY_STYLE[data[p]["family"]]["color"] for p in order]
            ax.barh(range(len(order)), vals, color=colors, height=0.62, edgecolor="white", linewidth=2)
            for i, v in enumerate(vals):
                ax.text(v + max(vals) * 0.02, i, f"{v:.2f} ms", va="center", fontsize=8.5, color=_INK)
            ax.set_title(f"Batch 1, {title}", loc="left", fontsize=10)
            ax.set_xlim(0, max(vals) * 1.25)
            ax.set_xlabel("ms per clip")
        axes[0].set_yticks(range(len(order)), order, fontsize=9)
        fig.suptitle(f"CPU latency of the temporal model + head (T = {num_frames}, median of 3 sessions)",
                     x=0.01, ha="left", fontsize=11, fontweight="bold", color=_INK)
        _family_legend(axes[1], families, loc="lower right")
        fig.tight_layout()
        save(fig, "04_latency_by_model")

    # 5. End-to-end cost per clip: the shared front end dominates.
    bb = [read_json(q) for q in BENCH_DIR.glob(f"mobilenetv2_backbone_T{num_frames}_*.json")]
    bb = [b for b in bb if b]
    comp = {(r["preset"], r["batch"], r["intra_threads"]): r for r in read_csv(Path(results_dir) / "summary_compute.csv")
            if int(r["T"]) == num_frames}
    if have_latency and bb and ("mobilenetv2_backbone", "1", "1") in comp:
        decode = float(np.median([b["decode_ms_median"] for b in bb]))
        prep = float(np.median([b["preprocess_ms_median"] for b in bb]))
        backbone = float(comp[("mobilenetv2_backbone", "1", "1")]["median_ms"])
        order = sorted(data, key=lambda p: data[p]["lat1"])
        fig, ax = plt.subplots(figsize=(8.4, 4.0))
        _style_axes(ax)
        ax.grid(axis="y", visible=False)
        parts = [("Decode video", decode, "#D9D9D9"), ("Preprocess", prep, "#BDBDBD"),
                 ("MobileNetV2 backbone", backbone, "#9E9E9E")]
        for i, p in enumerate(order):
            left = 0.0
            for name, v, c in parts:
                ax.barh(i, v, left=left, color=c, height=0.62, edgecolor="white", linewidth=2,
                        label=name if i == 0 else None)
                left += v
            st = FAMILY_STYLE[data[p]["family"]]
            ax.barh(i, data[p]["lat1"], left=left, color=st["color"], height=0.62)
            total = left + data[p]["lat1"]
            share = 100 * data[p]["lat1"] / total
            ax.text(total + 6, i, f"{total:.0f} ms (temporal model {data[p]['lat1']:.2f} ms, "
                    f"{'<0.1' if share < 0.1 else f'{share:.1f}'}%)", va="center", fontsize=8, color=_INK)
        ax.set_yticks(range(len(order)), order, fontsize=9)
        ax.set_xlim(0, (decode + prep + backbone) * 1.55)
        ax.set_xlabel("ms per clip, batch 1, 1 thread")
        ax.set_title(f"End-to-end cost per clip: the temporal model is under 1% (T = {num_frames})", loc="left")
        from matplotlib.patches import Patch

        handles = [Patch(color=c, label=n) for n, _, c in parts] + [
            Patch(color=FAMILY_STYLE[fam]["color"], label=f"Temporal model: {FAMILY_STYLE[fam]['label']}")
            for fam in FAMILY_STYLE if fam in families]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=3, frameon=False,
                  fontsize=8, labelcolor=_INK)
        save(fig, "05_end_to_end_breakdown")

    # 6. Equivalence: AUROC difference to the reference with 90% CI and the TOST margin.
    if ref:
        cmp = paired_vs_reference(rows, list(data))
        if cmp:
            fig, ax = plt.subplots(figsize=(7.2, 3.9))
            _style_axes(ax)
            ax.grid(axis="y", visible=False)
            ax.axvspan(-EQUIVALENCE_MARGIN, EQUIVALENCE_MARGIN, color="#F2F2F2", zorder=0)
            ax.axvline(0, color=_MUTED, linewidth=1)
            for i, c in enumerate(cmp):
                st = FAMILY_STYLE[data[c["preset"]]["family"]]
                ax.plot(c["ci90"], [i, i], color=st["color"], linewidth=2.2, solid_capstyle="round")
                ax.scatter(c["mean"], i, s=60, marker=st["marker"], color=st["color"], edgecolors="white",
                           linewidths=1.5, zorder=3)
                verdict = "equivalent" if c["equivalent"] else "not shown equivalent"
                ax.text(max(c["ci90"][1], EQUIVALENCE_MARGIN) + 0.004, i,
                        f"{c['mean']:+.3f}  p = {c['p']:.2f}, {verdict}", va="center", fontsize=8, color=_INK)
            ax.set_yticks(range(len(cmp)), [c["preset"] for c in cmp], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(-0.06, 0.1)
            ax.set_xlabel(f"Test AUROC minus {REFERENCE_PRESET} (mean of paired folds, line = 90% CI)")
            ax.set_title(f"Equivalence with {REFERENCE_PRESET}: shaded band = +/- {EQUIVALENCE_MARGIN} AUROC "
                         f"(T = {num_frames})", loc="left")
            ax.text(0, -0.24, "Corrected resampled t-test (Nadeau and Bengio). A model is equivalent when its "
                    "whole 90% CI lies inside the band (TOST).", transform=ax.transAxes, fontsize=8, color=_MUTED)
            save(fig, "06_equivalence_vs_lstm")

    print(f"{len(written)} graphs -> {out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir}")
    return written


# =============================================================================================
# 9. Command line
# =============================================================================================

def model_cli(family, presets, build, recipe, macs_fn, argv=None):
    """The command line every model file exposes:

        python -m src.<family> --preset NAME --T 20 --seed 0 [--folds 0 1] [--lr] [--epochs]
        python -m src.<family> --preset NAME --T 20 --dry-run          # build + check, no training
        python -m src.<family> --preset NAME --T 20 --benchmark --threads 1
    Without --preset, every preset in the file is used. A fold that fails is recorded and the
    remaining folds still run; the command exits with an error at the end if any fold failed.
    """
    parser = argparse.ArgumentParser(description=f"{family} models: train, check or benchmark")
    parser.add_argument("--preset", nargs="*", choices=list(presets), default=None)
    parser.add_argument("--T", type=int, default=20, choices=SEQUENCE_LENGTHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="default: all folds")
    parser.add_argument("--lr", type=float, default=None, help="override the learning rate (own folder)")
    parser.add_argument("--epochs", type=int, default=None, help="override max epochs (own folder)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-run completed folds (the old files are archived, not deleted)")
    parser.add_argument("--dry-run", action="store_true", help="build and check only; no training")
    parser.add_argument("--benchmark", action="store_true", help="CPU latency/memory benchmark")
    parser.add_argument("--threads", type=int, default=None, help="intra-op threads (benchmark default 1)")
    parser.add_argument("--session", type=int, default=0, help="benchmark repeat index")
    parser.add_argument("--verbose", type=int, default=2)
    args = parser.parse_args(argv)
    if args.benchmark or args.threads:
        set_threads(args.threads or 1)
    names = args.preset or list(presets)
    recipe = full_recipe(recipe)
    mode = "benchmark" if args.benchmark else "dryrun" if args.dry_run else "train"
    start_console_log(f"{family}_{mode}_T{args.T}" + ("" if args.benchmark else f"_s{args.seed}"))

    if args.benchmark:
        for preset in names:
            benchmark_model(build, family, preset, args.T, macs_fn, args.session)
        return

    lr = args.lr if args.lr is not None else recipe["learning_rate"]
    variant = variant_name(recipe, args.lr, args.epochs)
    data, split = load_features(args.T), load_split(args.seed)
    data["split_md5"] = split["md5"]
    if args.dry_run:
        idx = fold_indices(data["ids"], split["folds"][0])
        for preset in names:
            model = compile_model(build, preset, args.T, recipe, lr)
            out = model.predict(data["features"][idx["test"][:4]], verbose=0)
            print(f"ready: {preset} T={args.T} params={model.count_params():,} "
                  f"MACs/clip={macs_fn(preset, args.T):,} optimizer={type(model.optimizer).__name__} "
                  f"lr={lr} variant={variant}; fold 0 sizes { {k: len(v) for k, v in idx.items()} }; "
                  f"forward pass -> {out.shape}")
        return

    folds = args.folds if args.folds is not None else range(len(split["folds"]))
    failed = []
    for preset in names:
        aurocs = []
        for k in folds:
            out_dir = RESULTS_DIR / run_name(preset, args.T, args.seed, k, variant)
            if (read_json(out_dir / "status.json") or {}).get("status") == "completed" and not args.overwrite:
                print(f"skip {out_dir.name}: already completed (use --overwrite)")
                continue
            print(f"=== {out_dir.name} ===", flush=True)
            idx = fold_indices(data["ids"], split["folds"][k])
            try:
                rec = train_fold(build, family, preset, args.T, args.seed, k, data, idx, recipe, out_dir,
                                 learning_rate=lr, max_epochs=args.epochs, verbose=args.verbose,
                                 variant=variant)
            except Exception as exc:
                failed.append(out_dir.name)
                traceback.print_exc()
                print(f"[fail] {out_dir.name}: {type(exc).__name__}: {exc}; continuing with the next fold",
                      file=sys.stderr)
                continue
            aurocs.append(rec["metrics"]["test"]["auroc"])
            print(f"{out_dir.name}: test AUROC {aurocs[-1]:.4f}, best epoch "
                  f"{rec['training']['best_epoch']}, {rec['training']['fit_seconds']:.0f} s")
        if aurocs:
            print(f"{preset}: mean test AUROC over {len(aurocs)} fold(s) {np.mean(aurocs):.4f}")
    if failed:
        print(f"{len(failed)} run(s) failed: {failed}. Their partial files are kept; re-run the same "
              f"command to retry them (the failed attempt is archived first).", file=sys.stderr)
        sys.exit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared data, benchmark and summary commands")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract", help="cache MobileNetV2 features for every video")
    p.add_argument("--frames", type=int, default=CACHE_FRAMES)
    p.add_argument("--limit", type=int, default=None, help="first N videos only (smoke test)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=None,
                   help="cap TensorFlow intra-op threads (leave CPU free for other work)")
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("splits", help="write the shared fold files")
    p.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    p = sub.add_parser("backbone", help="MobileNetV2 + decode cost per clip (end-to-end part)")
    p.add_argument("--T", type=int, default=20, choices=SEQUENCE_LENGTHS)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--session", type=int, default=0)
    p = sub.add_parser("benchmark-all", help="full compute grid, one fresh process per job; resumable")
    p.add_argument("--sessions", type=int, default=BENCHMARK["sessions"])
    p.add_argument("--no-backbone", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run jobs that already have results")
    p.add_argument("--presets", nargs="*", default=None, help="only these presets (default: all)")
    p.add_argument("--T", type=int, nargs="*", default=list(SEQUENCE_LENGTHS), choices=SEQUENCE_LENGTHS)
    sub.add_parser("summary", help="write results/summary_*.csv and the Markdown report")
    p = sub.add_parser("graphs", help="write the compute-vs-accuracy figures to graphs/")
    p.add_argument("--T", type=int, default=20, choices=SEQUENCE_LENGTHS)
    args = parser.parse_args(argv)

    if args.command == "backbone" or (args.command == "extract" and args.threads):
        set_threads(args.threads)
    start_console_log(f"common_{args.command}")
    if args.command == "extract":
        extract_features(args.frames, args.limit, args.workers, args.overwrite)
    elif args.command == "splits":
        write_splits(args.seeds)
    elif args.command == "backbone":
        benchmark_backbone(args.T, session=args.session)
    elif args.command == "benchmark-all":
        benchmark_all(sessions=args.sessions, backbone=not args.no_backbone, force=args.force,
                      presets=args.presets, sequence_lengths=tuple(args.T))
    elif args.command == "summary":
        summarise()
    elif args.command == "graphs":
        plot_graphs(num_frames=args.T)


if __name__ == "__main__":
    main()
