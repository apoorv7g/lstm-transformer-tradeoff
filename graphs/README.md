# Graphs

Figures for the compute versus accuracy comparison of the LSTM, Transformer, hybrid and
no-temporal baseline models. They cover the 7 trained models at T = 20 frames per clip
(3-fold cross-validation x 2 seeds = 6 test folds per model, 42 runs in total).

Each figure is saved twice: `.png` (200 dpi, for slides) and `.svg` (vector, for print).

## The figures

| File | What it shows | How to read it |
|---|---|---|
| `01_pareto_auroc_vs_latency` | **The main graph.** Test AUROC against CPU latency for all 7 models, with the Pareto front and the +/- 0.03 equivalence band around `lstm_base`. | Every model falls inside the grey band, so the cheaper models lose no accuracy. The dashed line joins the models that no other model beats on both speed and accuracy. |
| `02_pareto_sensitivity_vs_latency` | The screening view: sensitivity at 90% specificity against latency. | Higher is better: the share of Abnormal clips caught while only 1 in 10 Normal clips is flagged. `hybrid_attn` and `lstm_base` lean highest, but the differences are within fold noise. |
| `03_auroc_vs_macs` | Accuracy against multiply-accumulates (MACs) per clip. | The same comparison as figure 1, but with a compute measure that does not depend on the hardware. |
| `04_latency_by_model` | Latency per model as bars, at 1 thread and at 8 threads (batch 1). | The Transformer is 2.6 times faster than the LSTM on 1 thread. More threads make single clips slower, because the work is too small to share. |
| `05_end_to_end_breakdown` | The full cost of one clip: video decoding, preprocessing, the MobileNetV2 backbone, then the temporal model. | The temporal model is under 1% of the total (the thin coloured sliver at the end of each bar). Decoding and the backbone dominate. |
| `06_equivalence_vs_lstm` | Each model's AUROC difference from `lstm_base` with its 90% confidence interval. | The visual form of the equivalence test: a model is equivalent when its whole interval lies inside the grey +/- 0.03 band. |

## Notes

- **Latency** is for the temporal model plus the classifier head, at batch 1, median of 3 timing
  sessions on an Intel i5-12450H CPU. Figure 5 adds the shared front end (decode, preprocess,
  MobileNetV2), which every model pays equally.
- **Error bars** in figures 1 to 3 are the standard deviation over the 6 test folds.
- **Statistics** in figure 6 use the corrected resampled t-test (Nadeau and Bengio). The +/- 0.03
  AUROC margin was fixed before the results existed.
- **Colours:** LSTM orange, Transformer blue, hybrid green, and the no-temporal baselines grey
  (they are controls, not a model family). Each family also has its own marker shape and every
  point is labelled by name, so the figures read without colour.
- The numbers behind every figure are in `results/results_3fold_T20.md`.

## Rebuilding

The figures are generated from `results/runs.csv`, `results/summary_compute.csv` and
`results/benchmark/`. After new runs or benchmarks, from the repo root:

```
python -m src.common summary     # refresh the summary tables and the report
python -m src.common graphs      # redraw every figure in this folder
```
