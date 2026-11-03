# src

All the Python code. Run everything from the repo root with `python -m ...`.

| Folder / file | What it does | Status |
|---|---|---|
| `config.py` | Finds the repo root and reads the YAML files in `configs/` | Done |
| `data/` | Turns videos into cached features and writes the folds | Done |
| `models/` | Builds the LSTM, Transformer and hybrid models | Done |
| `training/` | Trains a model on each fold and saves its scores | To do |
| `benchmark/` | Measures speed and memory, then makes the final tables and plots | To do |

The order of work is: `data` then `training` then `benchmark`. `models` is used by the last two.
