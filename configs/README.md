# configs

Settings for every step of the study, kept here so no path or number is hard-coded in the code.

| File | What it controls |
|---|---|
| `data.yaml` | Where the videos are, how many frames to take (40), image size, the MobileNetV2 backbone, and how the folds are made (5 folds, seeds 0, 1, 2, 15% validation) |
| `lstm.yaml` | The LSTM model sizes and how they are trained (optimizer, learning rate, epochs) |
| `transformer.yaml` | The same for the Transformer models (and the hybrid) |
| `benchmark.yaml` | How speed and memory are measured (batch sizes, CPU threads, repeat counts) |

All paths in these files are relative to the repo root.

To change a setting, edit the YAML file; the code reads it with `load_config("data")` from
`src/config.py`. Do not change a model size once results exist; add a new model name instead.
