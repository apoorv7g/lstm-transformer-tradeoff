# 1. Project Summary

**Title:** Compute versus Accuracy Trade-off in LSTM and Transformers for Medical Video Classification
**Type:** Course research project (Deep Learning). Proposal deck: `CacheMemory.pdf`.
**Last updated:** 3 Nov 2026

---

## The problem in one paragraph

Deep-learning video classifiers in medicine (endoscopy, ultrasound, otoscopy, surgical video)
usually pair a **per-frame CNN** with a **temporal model**. Two families dominate the temporal step:

- **LSTMs** process frames one at a time. They are memory-light and simple, but strictly
  sequential, and in principle weaker on long-range dependencies.
- **Transformers** use self-attention, so every frame can look at every other frame in parallel.
  They are often more accurate on large datasets, but carry more parameters, a cost that grows
  with T², and a reputation for needing lots of data.

A clinic or screening programme that deploys on **ordinary CPUs** needs to know whether the
Transformer's extra cost is paid back in accuracy *for its kind of data*: small datasets, short
clips, class imbalance. That is the question this project answers with numbers.

## What we are doing

1. **One dataset, one task.** Binary Normal/Abnormal classification of pediatric **ear
   (otoscopy) screening videos**: 874 clips, 643 Normal / 231 Abnormal. They come from an existing
   curated pipeline (an earlier project that already had a working MobileNetV2 + LSTM model). The
   dataset history is in `docs/data_pipeline.md`.
2. **Freeze the spatial part.** Every frame goes through the same ImageNet MobileNetV2 and becomes a
   1280-d vector. We compute this once and cache it. The CNN is therefore identical for every
   model, so any difference comes from the temporal model alone.
3. **Swap only the temporal block.** We build a small, deliberately designed set of models:
   - LSTM family: `lstm_small`, `lstm_base` (the original legacy head), `bilstm_base`
   - Transformer family: `transformer_small`, `transformer_base` (parameter-matched to
     `lstm_base`), `transformer_deep`
   - Hybrids (LSTM, then attention): `hybrid_base`, `hybrid_attn`, `hybrid_lite`
   - No-temporal controls: `pool_mean`, `pool_max`
4. **Measure both axes on a CPU-only machine:**
   - *Accuracy:* AUROC, macro-F1, balanced accuracy, sensitivity, specificity. Stratified 3-fold
     CV with 2 seeds, identical folds for every model.
   - *Compute:* parameters, FLOPs/MACs, inference latency (batch 1 and batch 32, fixed thread
     counts), throughput, peak RAM, training time per epoch, model file size. Measured for the
     head alone **and** end-to-end including MobileNetV2.
   - Sweep the clip length **T ∈ {10, 20, 40}** to see how each family scales.
5. **Test a null hypothesis** with statistics suited to k-fold results (corrected resampled
   t-test, effect sizes, equivalence margin). Do not just compare two numbers.
6. **Deliver a Pareto chart** of accuracy against CPU cost, and a recommendation of the form
   "for clips like these, use X".

## Why it is a good question (short version)

The literature does not agree:

- Transformers win on large benchmarks.
- Several **small-data video studies** found the LSTM equal or better:
  - violence detection with LSTM vs Transformer (Soeleman et al., 2022)
  - CNN+LSTM reaching about 93% on only 185 lung-ultrasound videos (Barros et al., 2021)
  - the Keras video-Transformer example itself warns that it "works best with a larger dataset"

Medical video is almost always small-data. Our back-of-envelope analysis also shows the "Transformer
is expensive" intuition may simply be **wrong on CPU for short clips**: at T ≤ 40 the T² attention
term is tiny, and a batch-1 LSTM pays a sequential-loop overhead that attention does not. A quick
smoke test on our laptop already pointed that way: in eager mode at T=20 the transformer heads ran
faster than the LSTMs. That is not yet a result, only a reason to measure carefully. See
[04_why_it_matters.md](04_why_it_matters.md).

## What already exists (3 Nov)

- Done: Proposal and literature list (10 papers, now checked; see
  `docs/literature_review.md`)
- Done: Full review of the legacy LSTM code. It found real bugs that we fix here:
  - wrong MobileNetV2 input scaling and BGR channel order
  - data leakage in its confusion-matrix evaluation
  - non-stratified folds on imbalanced data
  - an approximately 21 GB float64 in-memory video tensor

  Details: `docs/legacy_pipeline_audit.md`.
- Done: Model code for every preset (`src/lstm.py`, `src/transformer.py`, `src/hybrid.py`). All
  build under Keras 3.6 / TF 2.17, and their parameter counts are checked.
- Done: Design docs for the data pipeline, LSTM, Transformer and benchmarking protocol.
- Done: Frame sampling, the ear feature cache, the shared splits, the training loop and the
  benchmark code (`src/common.py`).
- Done (4 Nov): 42 training runs (7 models x 3 folds x 2 seeds at T = 20), the CPU benchmark,
  statistics, figures (`graphs/`) and the A4 report (`docs/report/`).
- Result: the LSTM and the Transformer are equally accurate (AUROC 0.907 vs 0.905) and the
  Transformer is 2.6 times faster per clip; the temporal model is under 1% of end-to-end time.
  See `docs/05_results.md`.

## Expected deliverables

1. A reproducible codebase (this repo) that runs the whole study from `labels.json` and videos to
   final tables.
2. A results table and an **accuracy-vs-latency Pareto plot** per T.
3. A hypothesis-test write-up: is the Transformer better, equivalent or worse, and at what cost.
4. A final presentation, built from the report and figures. The hybrid models are done; a short
   paper remains optional.
