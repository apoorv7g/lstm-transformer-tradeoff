# results

Everything the experiments on the ear dataset produce. Empty until training starts (apart from
`logs/`). Nothing in here is ever deleted by the code: a re-run archives the old files first.

## Training runs

One folder per run, named `<preset>_T<frames>_s<seed>_f<fold>`, for example `lstm_base_T20_s0_f0/`.
Runs with a changed learning rate or epoch budget get a suffix (`..._f0_lr0.0003`) so they never
mix with the main runs.

Written while the run is going, so a crash loses nothing that already happened:

| File | Contents |
|---|---|
| `status.json` | Live state: running / completed / failed / interrupted, attempt number, epochs done, best validation AUROC so far, last epoch's loss and learning rate; on failure the error and full traceback |
| `history.csv` | One row per epoch, saved as soon as the epoch ends: loss, AUROC, AUPRC, accuracy, precision, recall (train and validation, from Keras); exact validation AUROC, AUPRC, Brier, log loss, sensitivity, specificity, balanced accuracy, mean probability per class (from scikit-learn); learning rate used; epoch, train and validation time; step time median / p95 / max; memory; weight norm; time stamp |
| `val_probs_per_epoch.npy` | Validation probabilities after every epoch (epochs x clips) |
| `model_summary.txt`, `model_architecture.json` | Layer table and full Keras config |

Written at the end:

| File | Contents |
|---|---|
| `run.json` | Everything about the run: settings, data sizes, class weights, data fingerprints (feature cache size and date, split file checksum), parameter counts (also per layer) and model size, epochs run, best epoch, learning-rate cuts, time split (fit, train, validation, logging), peak memory, inference time per clip, test / validation / train metrics, hardware and software versions, git commit, command line |
| `predictions.csv` | Predicted probability for every clip (train, validation and test), with its label and predictions at all three cut-offs (0.5, Youden, high-sensitivity) |
| `model.keras` | Best weights |

Metrics in `run.json`, for each of test, validation and train: AUROC (main score), AUPRC, Brier
score, log loss, sensitivity at 90% specificity, specificity at 90% and 95% sensitivity, and at
three cut-offs (0.5; the one that maximises sensitivity + specificity on the validation set; the
highest one that reaches 95% sensitivity on the validation set, `at_high_sens`): accuracy,
balanced accuracy, macro-F1, Abnormal F1, sensitivity, specificity, PPV, NPV, MCC and the
confusion matrix (`fn` = missed Abnormal clips). `history.csv` also logs the validation
sensitivity at 90% specificity for every epoch.

If a run breaks, run the same command again. The old files move to
`attempts/attempt<N>_<status>/` (`failed`, `interrupted`, or `crashed` if the process died) and the
run starts new files. Completed runs are skipped unless you pass `--overwrite` (which also archives).

## Files across all runs

| File | Contents |
|---|---|
| `runs.csv` | One row per completed run with every number from `run.json` flattened into columns |
| `events.jsonl` | Append-only journal, one line per event: run start, every epoch, run end or failure, every benchmark measurement and grid job. Saved to disk immediately, so it survives any crash |
| `logs/<time>_<command>_<pid>.log` | The full console output of every command, one new file per command |
| `results_3fold_T20.md` | Readable report of the 42-run batch (7 models x 2 seeds x 3 folds, T = 20): setup, detailed findings, per-model accuracy and threshold metrics, measured CPU latency and end-to-end cost, paired tests against `lstm_base`, and one row per run. Rebuilt by `python -m src.common summary` from `runs.csv`, `summary_compute.csv` and `benchmark/`. Every table is regenerated; the hand-written findings between `<!-- findings:start -->` and `<!-- findings:end -->` are kept, so edit them in the report itself |

## Benchmarks

| File | Contents |
|---|---|
| `benchmark/<preset>_T<T>_b<batch>_th<threads>_r<session>.json` | One measurement, including every raw timing |
| `benchmark.csv` | One row per measurement: latency median / mean / p5 to p99 / IQR, clips per second, first-call time, memory before and after building and peak while timing, CPU load, parameters, file size, MACs and GFLOPs |
| `benchmark_grid_status.json` | Outcome of the last `benchmark-all`: jobs run, failures, whether the grid is complete |

The backbone rows (`mobilenetv2_backbone`) hold the MobileNetV2, decode and preprocessing cost. Add
these to a model's batch-1 latency to get the end-to-end cost. `benchmark-all` skips measurements
that already exist, so after an interruption just run it again.

## Summaries

`python -m src.common summary` writes `summary_accuracy.csv` (mean and sd per model, T and variant)
and `summary_compute.csv` (median over sessions per benchmark setting).

## What is committed

Only aggregate files: `run.json`, `status.json`, `history.csv`, the model text files, `runs.csv`,
`events.jsonl`, benchmark files and summaries. Kept on disk but not committed: `predictions.csv`
and `val_probs_per_epoch.npy` (per-video), `attempts/`, `logs/`, and model weights.
