# 5. Results and Conclusions

Results of the main experiment (4 Nov 2026). The full numbers, including one row per run and the
threshold metrics at every cut-off, are in
[`results/results_3fold_T20.md`](../results/results_3fold_T20.md). The readable A4 report is
[`docs/report/Compute_vs_Accuracy_LSTM_Transformer_Report.pdf`](report/Compute_vs_Accuracy_LSTM_Transformer_Report.pdf)
and the figures are in [`graphs/`](../graphs/README.md).

---

## 5.1 What was run

| Item | Setting |
|---|---|
| Data | 874 ear videos (643 Normal / 231 Abnormal), frozen MobileNetV2 features, **T = 20** frames |
| Models | `lstm_base`, `transformer_base`, `hybrid_base`, `hybrid_attn`, `hybrid_lite`, `pool_mean`, `pool_max` |
| Cross-validation | Stratified **3-fold x 2 seeds** = 6 paired test folds per model (291–292 clips, 77 Abnormal each) |
| Training runs | 42, all completed (about 18 min on 4 CPU threads) |
| Compute | 48 benchmark jobs: 7 models x {1, 8} threads x 3 sessions + the MobileNetV2 front end; batch 1 and 32 |
| Statistics | Corrected resampled t-test (Nadeau and Bengio) and TOST with the pre-registered margin of +/- 0.03 AUROC, paired against `lstm_base` |

Not run (scope decision by the user): T = 10 and T = 40, a third seed, and the presets
`lstm_small`, `bilstm_base`, `transformer_small`, `transformer_deep`.

## 5.2 Accuracy

| Model | AUROC | Sens. at 90% spec. | Difference vs `lstm_base` (90% CI) | Equivalent (+/- 0.03) |
|---|---:|---:|---:|---|
| `hybrid_attn` | **0.909** | **0.760** | +0.002 (-0.018 to +0.022) | Yes |
| `lstm_base` | 0.907 | 0.749 | reference | |
| `transformer_base` | 0.905 | 0.699 | -0.001 (-0.019 to +0.016) | Yes |
| `hybrid_base` | 0.905 | 0.701 | -0.002 (-0.035 to +0.032) | Not shown |
| `hybrid_lite` | 0.900 | 0.684 | -0.007 (-0.040 to +0.027) | Not shown |
| `pool_mean` | 0.900 | 0.665 | -0.007 (-0.024 to +0.010) | Yes |
| `pool_max` | 0.893 | 0.639 | -0.013 (-0.035 to +0.008) | Not shown |

- No model differs significantly from `lstm_base` (all p >= 0.26). "Not shown" means the interval
  is wider than the margin, not that the model is worse.
- At the cut-off chosen on validation (Youden), recall is 0.78–0.82, specificity 0.83–0.87,
  precision 0.63–0.70 and accuracy 0.82–0.85 for every model. Always answering Normal scores 0.736.
- At the high-sensitivity cut-off (chosen on validation for 95% recall), test recall is 0.87–0.92:
  6 to 10 of 77 Abnormal clips are missed per fold, with precision 0.50–0.58 and NPV about 0.95.
- Probabilities are under-confident for Abnormal clips (mean 0.62–0.70), so the 0.5 cut-off is
  not a good operating point; `pool_mean` is the best calibrated (Brier 0.118).

## 5.3 Compute (temporal model and head, CPU, median of 3 sessions)

| Model | Batch 1, 1 thread | Batch 1, 8 threads | Batch 32, 8 threads | MACs per clip | vs `lstm_base` (batch 1, 1 thread) |
|---|---:|---:|---:|---:|---:|
| `lstm_base` | 3.54 ms | 5.82 ms | 18.9 ms | 35.4 M | reference |
| `hybrid_base` | 2.12 ms | 3.83 ms | 11.6 ms | 17.2 M | 1.7x faster |
| `transformer_base` | 1.36 ms | 2.02 ms | 19.1 ms | 28.0 M | **2.6x faster** |
| `hybrid_attn` | 1.11 ms | 1.50 ms | 8.5 ms | 8.6 M | 3.2x faster |
| `hybrid_lite` | 0.92 ms | 1.31 ms | 4.8 ms | 3.0 M | 3.9x faster |
| `pool_max` | 0.25 ms | 0.21 ms | 0.4 ms | 0.08 M | 14x faster |
| `pool_mean` | 0.22 ms | 0.23 ms | 0.4 ms | 0.08 M | 16x faster |

- The Transformer is faster than the LSTM at batch 1 in all 3 sessions; at batch 32 on 8 threads
  they tie. 8 threads make single clips slower than 1 thread (the work is too small to split).
- Peak memory is 330–410 MB for every model, almost all of it TensorFlow.
- **End to end**, every clip first costs about 145 ms of video decoding, 4 ms of preprocessing and
  312 ms of MobileNetV2 on 20 frames (125 ms on 8 threads). The temporal model is **under 1%** of
  the total on 1 thread: about 465 ms per clip with `lstm_base` and 462 ms with `transformer_base`.

## 5.4 Hypotheses

| Hypothesis | Result | Verdict |
|---|---|---|
| **H0-acc:** Transformer AUROC = LSTM AUROC | -0.001, 90% CI inside +/- 0.03 | Supported (equivalent) |
| **H0-cpu:** Transformer batch-1 latency <= LSTM | 1.36 vs 3.54 ms (2.6x faster), all sessions | Supported |
| H0-T: T does not affect AUROC | Not tested; only T = 20 was run | Out of scope |

## 5.5 Conclusions

1. **There is no compute-versus-accuracy trade-off between the LSTM and the Transformer here.**
   They are equally accurate, and the Transformer is cheaper: 2.6 times faster per clip, 21% fewer
   parameters and 21% fewer MACs.
2. **Frame order adds little on this task.** A model that only averages the frame features
   (`pool_mean`) is equivalent in AUROC at about 1/430 of the arithmetic. The only hint in favour
   of temporal models is sensitivity at 90% specificity (about 0.75 vs 0.67), which is not
   significant with 6 folds.
3. **The temporal model is not where the cost is.** Video decoding and MobileNetV2 take over 99% of
   the end-to-end time, so fewer frames or a lighter backbone would save far more than the choice
   of temporal model.
4. **Recommended models:**
   - `hybrid_attn`: best overall (joint-best AUROC, best sensitivity at 90% specificity, 3.2 times
     faster than the LSTM).
   - `pool_mean`: cheapest acceptable model (equivalent AUROC, 16 times faster, best calibrated).
   - Between the two classic models, `transformer_base` is the better choice for CPU deployment.
5. **Accuracy is limited by the frame features, not by the temporal model.** Further accuracy
   gains, including fewer missed Abnormal clips, most likely need better frame features
   (fine-tuning or a domain-specific backbone).

## 5.6 Limitations

- 6 paired folds give wide confidence intervals; three models could not be shown equivalent and the
  sensitivity gap could not be confirmed. A third seed would narrow this.
- Only T = 20 was tested.
- No patient identifier: retakes of the same child may sit in different folds. This affects every
  model equally, but absolute scores may be optimistic.
- Timings come from one laptop (Intel i5-12450H). End-to-end latency adds separately measured
  parts rather than timing one fused pipeline.
- DeLong and McNemar tests on pooled predictions and Holm correction were planned but not run.
- The MobileNetV2 backbone is frozen ImageNet weights; this is an engineering comparison, not a
  validated diagnostic tool.

## 5.7 How the numbers were produced

```
python -m src.<baseline|lstm|transformer|hybrid> --preset <name> --T 20 --seed <0|1>
python -m src.common benchmark-all --T 20 --presets pool_mean pool_max lstm_base \
       transformer_base hybrid_base hybrid_attn hybrid_lite
python -m src.common summary     # results/summary_*.csv and results/results_3fold_T20.md
python -m src.common graphs      # graphs/*.png and *.svg
```
