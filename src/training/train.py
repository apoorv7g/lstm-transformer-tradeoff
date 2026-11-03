"""Cross-validation training loop: one run per (preset, T, seed, fold).

Usage (from repo root):
    python -m src.training.train --preset lstm_base --T 20 --seed 0             # all 5 folds
    python -m src.training.train --preset transformer_base --T 20 --seed 0 --folds 0
    python -m src.training.train --preset transformer_base --T 20 --dry-run     # build + check, no fit

Every run reads the same cached features (src.data.features) and the same split file, so the LSTM
vs Transformer comparison is paired fold by fold. The recipe comes from configs/lstm.yaml or
configs/transformer.yaml depending on the preset family (the hybrid uses the Transformer recipe,
because it contains attention and needs the warm-up). The outer test fold is predicted exactly once,
after early stopping on the inner validation set has restored the best weights.

Each fold writes results/{preset}_T{T}_s{seed}_f{fold}/ with metrics.json, history.csv,
oof_predictions.csv and model.keras (gitignored).
"""

import argparse
import csv
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np

from src.config import load_config, repo_path
from src.data.features import fold_indices, load_features, load_split
from src.models import PRESET_NAMES, build_model, hybrid, lstm, transformer
from src.training.metrics import evaluate


def recipe_for(preset):
    """Return (family, training recipe dict) for a preset name."""
    if preset in lstm.PRESETS:
        return "lstm", load_config("lstm")["training"]
    if preset in transformer.PRESETS or preset in hybrid.PRESETS:
        return "transformer", load_config("transformer")["training"]
    raise KeyError(f"unknown preset {preset!r}; choose from {PRESET_NAMES}")


def balanced_class_weight(labels):
    """sklearn-style 'balanced' weights: n / (2 * count). About 0.59 Normal / 3.22 Abnormal here."""
    labels = np.asarray(labels).astype(int)
    counts = np.bincount(labels, minlength=2)
    return {c: float(len(labels) / (2 * counts[c])) for c in (0, 1) if counts[c]}


def make_optimizer(recipe, learning_rate):
    """Adam for the LSTM family, AdamW (no decay on biases, LayerNorm or embeddings) for attention."""
    name = recipe["optimizer"].lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=recipe["clipnorm"])
    if name == "adamw":
        opt = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=recipe["weight_decay"],
            clipnorm=recipe["clipnorm"],
        )
        opt.exclude_from_weight_decay(var_names=["bias", "gamma", "beta", "embeddings"])
        return opt
    raise ValueError(f"unsupported optimizer {name!r}")


class LinearWarmup(keras.callbacks.Callback):
    """Ramp the learning rate linearly over the first `epochs` epochs, then leave it alone.

    After the warm-up ReduceLROnPlateau owns the learning rate. Any plateau cut that fires during
    the warm-up is overwritten by the ramp, which is the intended behaviour.
    """

    def __init__(self, target_lr, epochs):
        super().__init__()
        self.target_lr = float(target_lr)
        self.epochs = int(epochs)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.epochs:
            self.model.optimizer.learning_rate = self.target_lr * (epoch + 1) / self.epochs


class EpochLog(keras.callbacks.Callback):
    """Collect one row per epoch (all logged metrics plus the learning rate actually used)."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        row = {"epoch": epoch + 1, "lr": float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))}
        row.update({k: float(v) for k, v in (logs or {}).items() if k not in ("learning_rate", "lr")})
        self.rows.append(row)


def make_callbacks(recipe, learning_rate):
    es, rlr = recipe["early_stopping"], recipe["reduce_lr_on_plateau"]
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=es["monitor"], mode=es["mode"], patience=es["patience"],
            restore_best_weights=es["restore_best_weights"],
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=rlr["monitor"], mode=rlr["mode"], factor=rlr["factor"], patience=rlr["patience"],
        ),
    ]
    if recipe.get("warmup_epochs"):
        callbacks.insert(0, LinearWarmup(learning_rate, recipe["warmup_epochs"]))
    log = EpochLog()
    callbacks.append(log)
    return callbacks, log


def compile_model(preset, num_frames, recipe, learning_rate):
    model = build_model(preset, num_frames=num_frames)
    model.compile(
        optimizer=make_optimizer(recipe, learning_rate),
        loss=recipe["loss"],
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def run_fold(preset, num_frames, seed, fold, data, idx, recipe, out_dir,
             learning_rate=None, max_epochs=None, verbose=2):
    """Train one fold, score its test part once and write the run folder. Returns the metrics dict."""
    keras.utils.set_random_seed(seed)
    lr = float(learning_rate if learning_rate is not None else recipe["learning_rate"])
    epochs = int(max_epochs if max_epochs is not None else recipe["max_epochs"])
    x, y, ids = data["features"], data["labels"].astype("float32"), data["ids"]
    tr, va, te = idx["train"], idx["val"], idx["test"]

    model = compile_model(preset, num_frames, recipe, lr)
    callbacks, log = make_callbacks(recipe, lr)
    start = time.perf_counter()
    model.fit(
        x[tr], y[tr],
        validation_data=(x[va], y[va]),
        epochs=epochs,
        batch_size=recipe["batch_size"],
        class_weight=balanced_class_weight(y[tr]) if recipe["class_weight"] == "balanced" else None,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
    )
    train_seconds = time.perf_counter() - start

    val_prob = model.predict(x[va], batch_size=256, verbose=0).ravel()
    test_prob = model.predict(x[te], batch_size=256, verbose=0).ravel()
    metrics = evaluate(y[te], test_prob, y[va], val_prob)
    best_epoch = int(np.argmax([r.get("val_auc", -1) for r in log.rows])) + 1 if log.rows else 0
    metrics.update(
        preset=preset, T=num_frames, seed=seed, fold=fold,
        params=int(model.count_params()), learning_rate=lr,
        epochs_run=len(log.rows), best_epoch=best_epoch,
        best_val_auc=float(max((r.get("val_auc", float("nan")) for r in log.rows), default=float("nan"))),
        train_seconds=round(train_seconds, 2),
        n_train=int(len(tr)), n_val=int(len(va)),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    _write_csv(out_dir / "history.csv", log.rows)
    _write_csv(out_dir / "oof_predictions.csv", [
        {"id": str(i), "label": int(t), "prob": float(p), "preset": preset, "T": num_frames,
         "seed": seed, "fold": fold}
        for i, t, p in zip(ids[te], y[te], test_prob)
    ])
    model.save(out_dir / "model.keras")
    return metrics


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_dir(preset, num_frames, seed, fold, results_dir=None):
    base = repo_path(results_dir or load_config("data")["results_dir"])
    return base / f"{preset}_T{num_frames}_s{seed}_f{fold}"


def dry_run(preset, num_frames, seed, recipe, learning_rate, data=None, split=None):
    """Build, compile and do one forward pass on real fold shapes. No weights are updated."""
    keras.utils.set_random_seed(seed)
    model = compile_model(preset, num_frames, recipe, learning_rate)
    model.summary(line_length=100)
    if data is not None and split is not None:
        idx = fold_indices(data["ids"], split["folds"][0])
        sizes = {k: len(v) for k, v in idx.items()}
        y_train = data["labels"][idx["train"]]
        out = model.predict(data["features"][idx["test"]][:4], verbose=0)
        print(f"fold 0 sizes {sizes}; class weights {balanced_class_weight(y_train)}")
        print(f"forward pass on 4 test clips -> {out.shape}, probs {np.round(out.ravel(), 3).tolist()}")
    print(f"ready: {preset} T={num_frames} params={model.count_params():,} "
          f"optimizer={type(model.optimizer).__name__} lr={learning_rate}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", required=True, choices=PRESET_NAMES)
    parser.add_argument("--T", type=int, default=20, choices=(10, 20, 40))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="default: all folds")
    parser.add_argument("--lr", type=float, default=None, help="override the recipe learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="override max_epochs")
    parser.add_argument("--dataset", default="oral")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--overwrite", action="store_true", help="re-run folds that already have metrics")
    parser.add_argument("--dry-run", action="store_true", help="build and check only; do not train")
    parser.add_argument("--verbose", type=int, default=2)
    args = parser.parse_args(argv)

    family, recipe = recipe_for(args.preset)
    lr = args.lr if args.lr is not None else recipe["learning_rate"]
    data = load_features(args.T, args.dataset)
    split = load_split(args.seed, args.dataset)

    if args.dry_run:
        print(f"family: {family}; recipe: {recipe}")
        dry_run(args.preset, args.T, args.seed, recipe, lr, data, split)
        return

    folds = args.folds if args.folds is not None else range(len(split["folds"]))
    aurocs = []
    for k in folds:
        out_dir = run_dir(args.preset, args.T, args.seed, k, args.results_dir)
        if (out_dir / "metrics.json").exists() and not args.overwrite:
            print(f"skip {out_dir.name}: metrics.json exists (use --overwrite)")
            continue
        print(f"=== {out_dir.name} ===")
        idx = fold_indices(data["ids"], split["folds"][k])
        m = run_fold(args.preset, args.T, args.seed, k, data, idx, recipe, out_dir,
                     learning_rate=lr, max_epochs=args.epochs, verbose=args.verbose)
        aurocs.append(m["auroc"])
        print(f"{out_dir.name}: test AUROC {m['auroc']:.4f}, best epoch {m['best_epoch']}, "
              f"{m['train_seconds']:.0f} s")
    if aurocs:
        print(f"mean test AUROC over {len(aurocs)} fold(s): {np.mean(aurocs):.4f}")


if __name__ == "__main__":
    main()
