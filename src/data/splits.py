"""Write the shared cross-validation splits. Spec: docs/design/data_pipeline.md section 4.

Usage (from repo root):
    python -m src.data.splits --dataset oral            # seeds from configs/data.yaml (0, 1, 2)
    python -m src.data.splits --dataset oral --seeds 0

Writes data/splits/{dataset}_seed{s}.json with video IDs only (safe to commit). Every model reads
the same files, which is what makes the LSTM vs Transformer comparison paired.

Outer: StratifiedKFold(5, shuffle=True, random_state=seed). Inner: a stratified 15% validation
split of each outer training fold, used only for early stopping; the outer test fold is touched once.
Splits depend only on labels.json (not on the feature cache), so they can be created first.
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import load_config, repo_path
from src.data.labels import load_labels


def make_split(ids, labels, seed, n_splits=5, val_fraction=0.15):
    """Return {"seed", "n_splits", "val_fraction", "folds": [{"train", "val", "test"}]}."""
    ids, labels = np.asarray(ids), np.asarray(labels)
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_val_idx, test_idx in outer.split(ids, labels):
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_fraction,
            stratify=labels[train_val_idx],
            random_state=seed,
        )
        folds.append({part: sorted(ids[idx].tolist(), key=_order(ids)) for part, idx in
                      (("train", train_idx), ("val", val_idx), ("test", test_idx))})
    return {"seed": seed, "n_splits": n_splits, "val_fraction": val_fraction, "folds": folds}


def _order(ids):
    position = {v: i for i, v in enumerate(ids.tolist())}
    return position.__getitem__


def summarise(split, ids, labels):
    """One line per fold: sizes and Abnormal counts, for a quick stratification check."""
    label_of = dict(zip(np.asarray(ids).tolist(), np.asarray(labels).tolist()))
    lines = []
    for k, fold in enumerate(split["folds"]):
        parts = [f"{p} {len(fold[p])} ({sum(label_of[v] for v in fold[p])} abn)" for p in ("train", "val", "test")]
        lines.append(f"  fold {k}: " + ", ".join(parts))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="oral")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)

    cfg = load_config("data")
    if args.dataset != cfg["dataset"]:
        raise ValueError(f"configs/data.yaml is set up for {cfg['dataset']!r}, not {args.dataset!r}")
    table = load_labels(repo_path(cfg["raw_dir"]) / "labels.json", cfg["labels"]["normal_value"])
    split_cfg = cfg["splits"]
    out_dir = repo_path(cfg["splits_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds if args.seeds is not None else split_cfg["seeds"]:
        split = make_split(table["ids"], table["labels"], seed,
                           split_cfg["n_splits"], split_cfg["inner_val_fraction"])
        split["dataset"] = args.dataset
        path = out_dir / f"{args.dataset}_seed{seed}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(split, fh, indent=1)
        print(f"wrote {path}")
        print(summarise(split, table["ids"], table["labels"]))


if __name__ == "__main__":
    main()
