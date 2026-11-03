# Design: Benchmarking & Evaluation Protocol

This document is the contract for how every number in the final report is produced. If a
measurement does not follow it, it does not go in the report.

Code: `src/benchmark/` (planned). Config: `configs/benchmark.yaml`.

---

## 1. The experiment grid

| Axis | Values |
|---|---|
| Model | `lstm_small`, `lstm_base`, `bilstm_base`, `transformer_small`, `transformer_base`, `transformer_deep` (+ `hybrid_base` stretch) |
| Sequence length T | 10, 20, 40 |
| CV | stratified 5-fold × seed 0 (mandatory), seeds 1 and 2 (if time) |

- Seed 0 only: 6 × 3 × 5 = **90 training runs**. With all 3 seeds: 270.
- Cached features make each run about seconds to minutes on CPU.

## 2. Accuracy metrics (per outer fold, on the held-out test fold only)

**Positive class = Abnormal.**

| Metric | Role |
|---|---|
| **AUROC** | **Primary.** Threshold-free and robust to the 85/15 imbalance |
| Macro-F1 | Secondary, at threshold 0.5 |
| Balanced accuracy | Secondary |
| Sensitivity (Abnormal recall) and specificity | Clinical reporting |
| AUPRC | Supplementary (imbalance-aware) |
| Accuracy | Reported for comparison with legacy results only. **Majority baseline = 84.5%** |

Also store the out-of-fold probabilities for every video, for pooled ROC curves, McNemar's test and
DeLong's test. Output per run goes to `results/{model}_T{T}_s{seed}_f{fold}/`:
- `metrics.json`
- `oof_predictions.csv`
- `history.csv`
- weights (`*.keras`, gitignored)

## 3. Compute metrics

| Metric | How | Notes |
|---|---|---|
| Parameters | `model.count_params()`; trainable vs non-trainable; head vs backbone | Exact |
| MACs / FLOPs | (a) analytic formulas (`lstm_design.md` §2, `transformer_design.md` §5); (b) TF profiler on a frozen `tf.function` graph (`convert_variables_to_constants_v2_as_graph` + `tf.compat.v1.profiler` `float_operation`) | Report both. The profiler can under-count ops inside the LSTM `while_loop`. State MACs vs FLOPs (FLOPs ≈ 2 × MACs) |
| **Inference latency** | Compiled `tf.function(model, reduce_retracing=True)` called with `training=False`; 50 warm-up calls; **≥ 300 timed calls** with `time.perf_counter()` | Report **median, p95, IQR**. Batch 1 (clinical per-video case) and batch 32 |
| Throughput | clips/s = batch / median latency at batch 32 | |
| Peak memory | Peak RSS via `psutil` (background sampler thread at 5 ms) during inference, minus the RSS baseline after model load | `tracemalloc` cannot see TF's C++ allocator, so use RSS |
| Training cost | Seconds per epoch (Keras callback with `perf_counter`), epochs until early stop, total seconds per fold | Mean ± SD over folds |
| Model size | Bytes of `.keras` file; also `.weights.h5` | Head only. Backbone is reported once (it is shared) |
| End-to-end latency | Backbone per-frame latency × T (measured during extraction, batched) + head latency; also one true end-to-end graph `TimeDistributed(MobileNetV2) → head` at batch 1 for T = 20 | Shows the share of the head in the total |

### Thread configuration (fixed for every measurement)

```python
import os
os.environ["TF_NUM_INTRAOP_THREADS"] = str(k); os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = str(k)
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(k)
tf.config.threading.set_inter_op_parallelism_threads(1)
```

- Run with **k = 1** (single-core, embedded-like) and **k = 8** (all physical cores of the
  i5-12450H, which has 4 P-cores and 4 E-cores).
- Threads must be set **before** TensorFlow executes any op, so the benchmark runs **each model in
  a fresh Python process**. This also stops one model's memory from contaminating another's
  measurement.
- Record `TF_ENABLE_ONEDNN_OPTS` (default on in TF 2.17 on x86) in the run metadata.

### Machine hygiene checklist (before any timing session)

- [ ] Laptop on AC power, Windows power mode **"Best performance"**, same mode every session
- [ ] Close browsers, IDE indexing, OneDrive/sync, antivirus scans where possible
- [ ] Note CPU temperature or just let it idle 2 min first. Run models in a **shuffled order**
- [ ] Repeat the full timing grid in **3 separate sessions** (ideally after reboots); report the
      median across sessions

### Environment metadata saved with every result

CPU model, cores, threads, RAM, OS build, Python / TF / Keras / NumPy versions, thread settings,
oneDNN flag, git commit hash, date and time, and power mode (entered manually).

## 4. Head-only vs end-to-end

The temporal head is about 10–100 M MACs. MobileNetV2 is about 300 M MACs **per frame**. We
therefore always report two numbers:

1. **Head-only**: input is cached `(B, T, 1280)` features. This isolates the architectural
   question (RQ2).
2. **End-to-end**: video frames → backbone → head. This is the deployment answer (RQ4). Decoding
   and resizing time are reported separately, not hidden.

## 5. Statistics

All tests are paired by fold, since every model sees the identical split files.

1. **Primary comparison:** `transformer_base` vs `lstm_base` (parameter-matched) at T = 20, on
   AUROC.
2. **Corrected resampled t-test** (Nadeau & Bengio, 2003). A plain paired t-test on k-fold scores
   understates the variance because the training sets overlap. Use
   `var_corrected = var(d) · (1/n + n_test/n_train)`, where n = number of fold scores (5 or 15).
   With repeated CV (3 × 5) use the Bouckaert & Frank (2004) corrected repeated k-fold version.
3. **Effect size:** mean ΔAUROC with 95% CI (t-based and bootstrap), and Cohen's d_z.
4. **Equivalence:** TOST with margin **Δ = 0.03 AUROC**, fixed now, before the results exist.
5. **Wilcoxon signed-rank only with n ≥ 6 pairs.** With n = 5 the smallest attainable two-sided
   p-value is 0.0625, so it can never reject at 0.05.
6. **Sample-level checks** on pooled out-of-fold predictions: McNemar's test (errors at
   threshold 0.5) and DeLong's test (AUROC).
7. **Latency:** report median and IQR. Test the ratio of medians with a bootstrap CI across
   sessions. Mann-Whitney U on raw timings is supplementary only, because timings within one
   session are not independent.
8. **Multiple comparisons:** Holm–Bonferroni across the family of model-pair comparisons.

## 6. Final outputs

- `results/summary_accuracy.csv`: model × T, mean ± SD of every accuracy metric
- `results/summary_compute.csv`: model × T × threads × batch, every compute metric
- `results/stats.md`: every hypothesis test with statistic, p-value, CI and decision
- Figures in `results/figures/`:
  1. AUROC vs batch-1 latency (head-only), one panel per T, with Pareto front highlighted
  2. Latency vs T per model (k = 1 and k = 8)
  3. Same as (1) but end-to-end
  4. Training seconds per fold per model
  5. Pooled ROC curves for the primary pair
  6. Attention-map examples (qualitative)
