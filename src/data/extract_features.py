"""Cache frozen MobileNetV2 frame embeddings for every video. Spec: docs/data_pipeline.md section 3.

Usage (from repo root):
    python -m src.data.extract_features --dataset oral --frames 40
    python -m src.data.extract_features --dataset oral --frames 40 --limit 8   # quick smoke run

Writes data/features/{dataset}_T{frames}.npz with
    features float32 (N, T, 1280), labels int8 (N,) Abnormal = 1, ids str (N,), ages int16 (N,)
and a {dataset}_T{frames}.meta.json sidecar (backbone, preprocessing, per-video decode log,
per-frame backbone latency).

Why extract once: the backbone is frozen and nothing is augmented, so running it inside every model
on every epoch gives identical outputs at about 100x the cost. T = 20 and T = 10 are taken later as
features[:, ::2] and features[:, ::4] (see src/data/features.py).

Decoding dominates the cost (oral clips are long), so videos are decoded in a small thread pool
(OpenCV releases the GIL) while the main thread runs the backbone.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from src.config import load_config, repo_path
from src.data.labels import load_labels
from src.data.video_io import sample_frames


def _build_backbone(cfg):
    import keras

    size = tuple(cfg["frames"]["size"])
    backbone = keras.applications.MobileNetV2(
        input_shape=(*size, 3),
        include_top=False,
        weights=cfg["backbone"]["weights"],
        pooling=cfg["backbone"]["pooling"],
    )
    backbone.trainable = False
    return backbone, keras


def output_paths(features_dir, dataset, frames, limit=None):
    stem = f"{dataset}_T{frames}" + (f"_smoke{limit}" if limit else "")
    return features_dir / f"{stem}.npz", features_dir / f"{stem}.meta.json"


def extract(dataset="oral", frames=40, limit=None, workers=4, overwrite=False):
    cfg = load_config("data")
    if dataset != cfg["dataset"]:
        raise ValueError(f"configs/data.yaml is set up for {cfg['dataset']!r}, not {dataset!r}")
    raw_dir = repo_path(cfg["raw_dir"])
    features_dir = repo_path(cfg["features_dir"])
    features_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = output_paths(features_dir, dataset, frames, limit)
    if npz_path.exists() and not overwrite:
        raise FileExistsError(f"{npz_path} exists; pass --overwrite to rebuild it")

    table = load_labels(raw_dir / "labels.json", cfg["labels"]["normal_value"])
    ids = table["ids"][:limit] if limit else table["ids"]
    missing = [v for v in ids if not (raw_dir / v).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos listed in labels.json are missing, e.g. {missing[:3]}")

    size = tuple(cfg["frames"]["size"])
    backbone, keras = _build_backbone(cfg)
    preprocess = keras.applications.mobilenet_v2.preprocess_input
    feature_dim = int(cfg["backbone"]["feature_dim"])

    features = np.zeros((len(ids), frames, feature_dim), dtype=np.float32)
    per_video, failures, backbone_ms = {}, {}, []

    # Warm-up so graph tracing is not counted in the per-frame latency.
    backbone.predict_on_batch(np.zeros((frames, *size, 3), dtype=np.float32))

    def load(video_id):
        return sample_frames(raw_dir / video_id, num_frames=frames, size=size)

    print(f"extracting {len(ids)} videos at T={frames} with {workers} decode workers")
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map() keeps input order and prefetches ahead of the backbone.
        results = pool.map(lambda v: (v, _safe(load, v)), ids)
        for i, (video_id, (out, err)) in enumerate(results):
            if err is not None:
                failures[video_id] = err
                print(f"  [fail] {video_id}: {err}", file=sys.stderr)
                continue
            clip, meta = out
            x = preprocess(clip.astype(np.float32))  # [-1, 1]; NOT /255
            t0 = time.perf_counter()
            features[i] = backbone.predict_on_batch(x)
            backbone_ms.append((time.perf_counter() - t0) * 1000 / frames)
            per_video[video_id] = meta.to_dict()
            if meta.padded or meta.passes > 1:
                note = "padded" if meta.padded else "header frame count wrong"
                print(f"  [note] {video_id}: {note} (header {meta.header_frames}, decoded {meta.decoded_frames})")
            if (i + 1) % 25 == 0 or i + 1 == len(ids):
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (i + 1) * (len(ids) - i - 1)
                print(f"  {i + 1}/{len(ids)}  elapsed {elapsed / 60:.1f} min  eta {eta / 60:.1f} min", flush=True)

    if failures:
        # Never drop videos silently; the whole cache must cover labels.json.
        raise RuntimeError(f"{len(failures)} videos failed to decode: {sorted(failures)}; nothing written")

    total_s = time.perf_counter() - t_start
    tmp_path = npz_path.with_suffix(".tmp.npz")
    np.savez(
        tmp_path,
        features=features,
        labels=table["labels"][: len(ids)],
        ids=ids,
        ages=table["ages"][: len(ids)],
    )
    os.replace(tmp_path, npz_path)

    lat = np.array(backbone_ms)
    meta = {
        "dataset": dataset,
        "num_videos": int(len(ids)),
        "num_abnormal": int(table["labels"][: len(ids)].sum()),
        "frames": frames,
        "feature_dim": feature_dim,
        "label_polarity": "Abnormal = 1, Normal = 0",
        "backbone": {
            "name": "MobileNetV2",
            "weights": cfg["backbone"]["weights"],
            "pooling": cfg["backbone"]["pooling"],
            "trainable": False,
            "params": int(backbone.count_params()),
            "keras_version": keras.__version__,
        },
        "preprocessing": {
            "sampling": "uniform linspace over decoded frames, sequential decode",
            "color": "BGR -> RGB",
            "resize": f"{size[0]}x{size[1]} INTER_AREA stretch",
            "normalisation": "mobilenet_v2.preprocess_input ([-1, 1])",
            "short_video": "repeat_pad",
        },
        "timing": {
            "total_minutes": round(total_s / 60, 2),
            "decode_workers": workers,
            "backbone_ms_per_frame_median": round(float(np.median(lat)), 3),
            "backbone_ms_per_frame_p95": round(float(np.percentile(lat, 95)), 3),
            "backbone_batch": frames,
            "note": "measured during extraction while decode threads run; indicative only, "
            "the benchmark grid measures the backbone in isolation",
        },
        "num_padded": int(sum(m["padded"] for m in per_video.values())),
        "num_header_mismatch": int(sum(m["passes"] > 1 for m in per_video.values())),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "videos": per_video,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    print(f"wrote {npz_path} {features.shape} in {total_s / 60:.1f} min")
    print(f"padded {meta['num_padded']}, header mismatches {meta['num_header_mismatch']}, "
          f"backbone {meta['timing']['backbone_ms_per_frame_median']} ms/frame (median)")
    return npz_path


def _safe(fn, arg):
    try:
        return fn(arg), None
    except Exception as exc:  # recorded and reported; extraction aborts before writing
        return None, f"{type(exc).__name__}: {exc}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="oral")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--limit", type=int, default=None, help="only the first N videos (smoke test)")
    parser.add_argument("--workers", type=int, default=4, help="parallel video decode threads")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    extract(args.dataset, args.frames, args.limit, args.workers, args.overwrite)


if __name__ == "__main__":
    main()
