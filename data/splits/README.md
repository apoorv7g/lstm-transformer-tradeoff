# data/splits

Which videos are used for training, validation and testing in each fold. These files hold video
IDs only, so they are safe to commit, and **every model uses the same files**. That is what makes
the LSTM vs Transformer comparison fair.

| File | Contents |
|---|---|
| `oral_seed0.json` | 5 folds made with random seed 0 (the main one) |
| `oral_seed1.json`, `oral_seed2.json` | The same with seeds 1 and 2, for repeated runs |

Each fold: 363 train, 65 validation, 107 test videos. The share of Abnormal videos is kept the same
in every part (stratified). Every video is in the test set of exactly one fold.

- train: the model learns from these
- val: used only to decide when to stop training
- test: used once, at the end, to score the model

File format: `{"seed": 0, "folds": [{"train": [ids], "val": [ids], "test": [ids]}, ...]}`.

Rebuild with `python -m src.data.splits --dataset oral`. Read in code with
`load_split(seed)` and `fold_indices(ids, fold)` from `src/data/features.py`.
