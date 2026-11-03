# src/training

**Status: ready.** The code is written and tested; no real training run has been done yet.

| File | What it does |
|---|---|
| `train.py` | Trains one model on each fold of a split and saves its scores |
| `metrics.py` | AUROC (main score), AUPRC, F1, balanced accuracy, sensitivity, specificity |

Commands (run from the repo root):

```bash
# check a model builds and sees the right data, without training
python -m src.training.train --preset transformer_base --T 20 --dry-run

# train all 5 folds, or just one
python -m src.training.train --preset lstm_base --T 20 --seed 0
python -m src.training.train --preset lstm_base --T 20 --seed 0 --folds 0
```

Other options: `--lr` (for the small learning-rate check), `--epochs`, `--overwrite`
(by default, folds that already have results are skipped).

What it does for each fold:
1. Loads the features (`load_features(T)`) and the split (`load_split(seed)`).
2. Builds the model by name and trains it on the train part. The recipe comes from
   `configs/lstm.yaml` (LSTM models) or `configs/transformer.yaml` (Transformer and hybrid:
   AdamW plus a 5-epoch warm-up).
3. Stops when validation AUROC stops improving and keeps the best weights.
4. Scores the test part once and writes `results/<preset>_T<T>_s<seed>_f<fold>/`.
