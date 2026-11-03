"""Load the cached feature tensor at any sequence length T, plus the shared split files.

This is the only way training and benchmarking code should read features, so every model sees the
exact same inputs. T = 20 and T = 10 are strided views of the T = 40 cache (features[:, ::2] and
features[:, ::4]), which keeps the same underlying frames across all T.
"""

import json

import numpy as np

from src.config import load_config, repo_path

CACHE_FRAMES = 40


def load_features(num_frames=20, dataset="oral", cache_frames=CACHE_FRAMES):
    """Return dict: features float32 (N, num_frames, 1280), labels int8, ids str, ages int16."""
    if cache_frames % num_frames:
        raise ValueError(f"T={num_frames} must divide the cached T={cache_frames}")
    cfg = load_config("data")
    path = repo_path(cfg["features_dir"]) / f"{dataset}_T{cache_frames}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run: python -m src.data.extract_features --dataset {dataset} --frames {cache_frames}"
        )
    with np.load(path) as data:
        stride = cache_frames // num_frames
        return {
            "features": np.ascontiguousarray(data["features"][:, ::stride]),
            "labels": data["labels"],
            "ids": data["ids"],
            "ages": data["ages"],
        }


def load_split(seed=0, dataset="oral"):
    """Return the saved split dict {"seed", "folds": [{"train", "val", "test"}, ...]}."""
    cfg = load_config("data")
    path = repo_path(cfg["splits_dir"]) / f"{dataset}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run: python -m src.data.splits --dataset {dataset}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fold_indices(ids, fold):
    """Map one fold's id lists to integer row indices into the feature arrays."""
    position = {v: i for i, v in enumerate(ids.tolist())}
    return {part: np.array([position[v] for v in fold[part]]) for part in ("train", "val", "test")}
