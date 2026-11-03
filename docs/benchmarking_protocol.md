# Design: Benchmarking & Evaluation Protocol

This document is the contract for how every number in the final report is produced. If a
measurement does not follow it, it does not go in the report.

Code: `src/common.py` (section 4 "Metrics", section 6 "Training", section 7 "Compute"). Settings:
the `TRAINING` and `BENCHMARK` constants at the top of that file.

---

## 1. The experiment grid

| Axis | Values |
|---|---|
| Model | `lstm_small`, `lstm_base`, `bilstm_base`, `transformer_small`, `transformer_base`, `transformer_deep`; hybrids `hybrid_base`, `hybrid_attn`, `hybrid_lite`; no-temporal controls `pool_mean`, `pool_max` |
| Sequence length T | 10, 20, 40 |
| CV | stratified 3-fold × seeds 0 and 1 (6 paired fold scores per model) |

- Per model and T: 2 seeds × 3 folds = 6 training runs. The 7 main models at T = 20: **42 runs**.
- Cached features make each run seconds to minutes on CPU.
- Commands: `python -m src.<baseline|lstm|transformer|hybrid> --T <T> --seed <s>` (all presets in the file,
  all folds); completed runs are skipped, so the grid resumes.

## 2. Accuracy metrics (per outer fold, on the held-out test fold only)

**Positive class = Abnormal.** Each ear test fold holds 291–292 clips, 77 of them Abnormal.

| Metric | Role |
|---|---|
| **AUROC** | **Primary.** Threshold-free and robust to the 74/26 imbalance |
| Macro-F1 | Secondary, at threshold 0.5 |
| Balanced accuracy | Secondary |
| **Sensitivity at 90% specificity** | **Main screening number.** Read off the test ROC curve: of the Abnormal clips, how many are caught while only 1 in 10 Normal clips is flagged. Missing an Abnormal clip is the costlier error |
| Sensitivity and specificity at the high-sensitivity cut-off | Cut-off chosen on the inner validation set to reach 95% sensitivity, then applied unchanged to test. Shows how many Abnormal clips a deployable threshold still misses (`fn`) |
| Specificity at 90% / 95% sensitivity | The price, in false alarms, of catching nearly every Abnormal clip |
| Sensitivity (Abnormal recall) and specificity | Clinical reporting |
| AUPRC | Supplementary (imbalance-aware) |
| Accuracy | Reported for comparison with legacy results only. **Majority baseline = 73.6%** |

Threshold metrics are reported at 0.5, at the Youden cut-off and at the 95%-sensitivity cut-off,
both chosen on the inner validation set (never on test). Brier score, log loss, PPV, NPV and MCC are also stored. The probability for every video is
saved, for pooled ROC curves, McNemar's test and DeLong's test. Output per run goes to
`results/{preset}_T{T}_s{seed}_f{fold}/` (full list in `results/README.md`):
- `run.json` (settings, environment, metrics)
- `predictions.csv` (per video; not committed)
- `history.csv`, `status.json`
- `model.keras` (head weights; not committed)

## 3. Compute metrics

| Metric | How | Notes |
|---|---|---|
| Parameters | `model.count_params()`, also per layer | Exact |
| MACs / FLOPs | Analytic formulas per family (`macs()` in each model file; `lstm_design.md` §2, `transformer_design.md` §5). FLOPs = 2 × MACs | A TF-profiler count is not implemented; the profiler can under-count ops inside the LSTM `while_loop` anyway |
| **Inference latency** | Compiled `tf.function` calling the model with `training=False`, one per batch size; 50 warm-up calls; **300 timed calls** with `time.perf_counter()` | Report **median, p95, IQR** (the full percentile set is saved). Batch 1 (clinical per-video case) and batch 32 |
| Throughput | clips/s = batch / median latency | |
| Peak memory | Peak RSS via `psutil` (background sampler thread) while timing, plus RSS before and after building | `tracemalloc` cannot see TF's C++ allocator, so use RSS |
| Training cost | Seconds per epoch, step-time percentiles, epochs until early stop, total fit seconds per fold | From the training logs; mean ± SD over folds |
| Model size | Bytes of the head's `.keras` file | Head only. The backbone is reported once (it is shared) |
| End-to-end latency | `python -m src.common backbone --T <T>`: MobileNetV2 on T frames at batch 1 (10 warm-up, 50 timed calls), plus decode and preprocessing time for 10 videos. Add this to the head's batch-1 latency | Shows the share of the head in the total |

### Thread configuration (fixed for every measurement)

```python
tf.config.threading.set_intra_op_parallelism_threads(k)
tf.config.threading.set_inter_op_parallelism_threads(1)
```

- Run with **k = 1** (single-core, embedded-like) and **k = 8** (all physical cores of the
  i5-12450H, which has 4 P-cores and 4 E-cores).
- Threads must be set **before** TensorFlow executes any op, so `python -m src.common
  benchmark-all` runs **each job in a fresh Python process**. This also stops one model's memory
  from contaminating another's measurement.
- The grid (every preset × T × threads, plus the backbone) runs in a shuffled order and is repeated
  in 3 sessions (`BENCHMARK["sessions"]`). Jobs whose result files exist are skipped, so the grid
  resumes after an interruption.

### Machine hygiene checklist (before any timing session)

- [ ] Laptop on AC power, Windows power mode **"Best performance"**, same mode every session
- [ ] Close browsers, IDE indexing, OneDrive/sync, antivirus scans where possible
- [ ] Let the machine idle for 2 minutes first
- [ ] Ideally reboot between the 3 sessions; report the median across sessions

### Environment metadata saved with every result

CPU model, physical and logical cores, CPU frequency, RAM, OS, Python / TF / Keras / NumPy /
scikit-learn / OpenCV versions, thread settings, git commit (and whether the tree was dirty), the
command line, and the time. Record the power mode by hand.

## 4. Head-only vs end-to-end

The temporal head is about 10–100 M MACs. MobileNetV2 is about 300 M MACs **per frame**. We
therefore always report two numbers:

1. **Head-only**: input is cached `(B, T, 1280)` features. This isolates the architectural
   question (RQ2).
2. **End-to-end**: video decode + MobileNetV2 on T frames + head. This is the deployment answer
   (RQ4). Decoding and preprocessing time are reported separately, not hidden.

## 5. Statistics

Implemented in `src/common.py` (`paired_vs_reference`, used by `summary` and `graphs`): items 1, 2,
the 95% and 90% confidence intervals of item 3, and item 4. Items 5, 6 and 8 and the bootstrap of
item 7 were not run in the main experiment. Results: `docs/05_results.md`.

All tests are paired by fold, since every model sees the identical split files.

1. **Primary comparison:** `transformer_base` vs `lstm_base` (parameter-matched) at T = 20, on
   AUROC.
2. **Corrected resampled t-test** (Nadeau & Bengio, 2003). A plain paired t-test on k-fold scores
   understates the variance because the training sets overlap. Use
   `var_corrected = var(d) · (1/n + n_test/n_train)`, where n = number of fold scores (3 per seed, 6 in total)
   and n_test/n_train ≈ 291/583 = 0.5. With repeated CV (2 × 3) use the Bouckaert & Frank (2004) corrected repeated k-fold version.
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

- `results/summary_accuracy.csv`: model × T × variant, mean ± SD of the accuracy metrics
  (`python -m src.common summary`)
- `results/summary_compute.csv`: model × T × threads × batch, median over sessions
  (`python -m src.common summary`)
- `results/results_3fold_T20.md`: the full report (rebuilt by `summary`): per-model summary,
  paired tests and TOST against `lstm_base`, threshold metrics at three cut-offs, compute and
  end-to-end tables, one row per run, and hand-written findings that survive regeneration.
- `graphs/` (`python -m src.common graphs`): AUROC and sensitivity at 90% specificity against
  latency (Pareto front), AUROC against MACs, latency per model, end-to-end breakdown, equivalence
  forest plot. Of the figures originally planned below, the T panels, training-time bars, ROC
  curves and attention maps were not made:
  1. AUROC vs batch-1 latency (head-only), one panel per T, with Pareto front highlighted
  2. Latency vs T per model (k = 1 and k = 8)
  3. Same as (1) but end-to-end
  4. Training seconds per fold per model
  5. Pooled ROC curves for the primary pair
  6. Attention-map examples (qualitative)
