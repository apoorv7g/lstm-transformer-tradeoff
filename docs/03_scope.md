# 3. Scope of the Project

This document fixes the boundaries of the study. Anything not listed under **In scope** needs a team
decision before work starts on it.

---

## 3.1 Research questions

| ID | Question |
|----|----------|
| **RQ1 (accuracy)** | With identical frozen MobileNetV2 frame features, does a Transformer-encoder temporal head classify Normal vs Abnormal ear (otoscopy) clips more accurately than an LSTM head? |
| **RQ2 (compute)** | What do the two families cost on a CPU-only machine: parameters, FLOPs, batch-1 latency, throughput, peak RAM, training time, file size? Is the cost of the temporal head even significant next to the CNN backbone? |
| **RQ3 (scaling)** | How do accuracy and cost change as clip length T grows (10 → 20 → 40 frames)? Does the O(T²) attention term matter at these lengths? |
| **RQ4 (trade-off)** | Which models are **Pareto-optimal** in (AUROC, latency)? What do we recommend for CPU deployment? |
| RQ5 (stretch) | Does an LSTM→Transformer hybrid move the Pareto front? |

## 3.2 Hypotheses (fixed before the experiments)

The proposal requires a null hypothesis grounded in existing research.

**H₀-acc (primary).** On short clinical clips (T ≤ 40) encoded by frozen ImageNet MobileNetV2, the
mean cross-validated AUROC of the Transformer head is **not different** from that of the LSTM head.
- H₁-acc: the mean paired per-fold difference is not zero.
- Equivalence margin: **Δ = 0.03 AUROC**, tested with TOST. Failing to reject H₀ is not the same as
  showing equivalence, so equivalence gets its own test.
- *Grounding:*
  - Soeleman et al. (2022): LSTM ≥ Transformer on three small video datasets.
  - Barros et al. (2021): CNN+LSTM reaches about 93% on 185 ultrasound videos.
  - The Keras video-Transformer example notes it needs larger data.
  - LREC 2022 found LSTM and Transformer on par in limited-data regimes.

**H₀-cpu (secondary).** At matched parameter count, the Transformer head's median batch-1 CPU
inference latency is **≤** the LSTM head's.
- H₁-cpu: the Transformer is slower.
- This is a genuine test. Theory (see `docs/transformer_design.md` §5) suggests that on CPU
  at short T the sequential LSTM loop may be the slower one, the opposite of the common belief.

**H₀-T (scaling).** Increasing T from 10 to 40 does not change AUROC for either family.
- Short otoscopy clips may carry most of their information in the few frames where the eardrum is
  in view.

## 3.3 In scope

**Data**
- The only dataset: **pediatric ear (otoscopy) screening**. 874 videos, 643 Normal / 231
  Abnormal, from an earlier curated pipeline. This meets the proposal's "only one type of videos".
- The small external ear test set (50 videos) is not used, for training or for evaluation.
- Why this dataset: see "Dataset history" in `docs/data_pipeline.md`.

**Pipeline**
- Uniform frame sampling, correct MobileNetV2 preprocessing (RGB, `[-1, 1]`), 224×224.
- Features extracted once and cached as `.npz`.
- Stratified 3-fold CV with folds saved to disk and shared by every model, repeated with 2 seeds
  (0 and 1): 6 paired fold scores per model.

**Models.** All share the classifier head and the input features.
- LSTM: `lstm_small`, `lstm_base`, `bilstm_base`
- Transformer: `transformer_small`, `transformer_base`, `transformer_deep`
- Hybrid (stretch): `hybrid_base`
- Optional cheap recurrent baseline: true minLSTM (Feng et al., 2024)

**Metrics**
- Accuracy: AUROC (primary), macro-F1, balanced accuracy, sensitivity (Abnormal recall),
  specificity, plain accuracy. Plain accuracy is reported but not used for decisions because of
  class imbalance: the majority-class baseline is already **73.6%**.
- Compute: params, MACs/FLOPs (analytic and profiler), latency (median and p95; batch 1 and
  batch 32; 1 thread and all physical cores), throughput, peak RSS, seconds per epoch, epochs to
  converge, file size.
- Reported **head-only** and **end-to-end** (including MobileNetV2 on T frames).

**Hardware.** One CPU-only laptop: Intel Core i5-12450H (8 cores / 12 threads), 16 GB RAM,
Windows 11. Python 3.11, TensorFlow 2.17, Keras 3.6. No GPU anywhere in the measured path.

**Analysis**
- Corrected resampled t-test (Nadeau–Bengio), mean difference with 95% CI, TOST equivalence.
- Holm correction across comparisons.
- Pareto plots per T, and a recommendation table.

## 3.4 Out of scope (and why)

| Excluded | Reason |
|---|---|
| End-to-end fine-tuning of the CNN backbone | It would mix spatial and temporal effects. It is also infeasible on CPU in our time budget. |
| Video-native Transformers (ViViT full, TimeSformer, VideoMAE) | They need GPUs and large-scale pretraining. They are cited as context only. |
| GPU benchmarks | Deliberate constraint: the deployment target is CPU. |
| Multi-class diagnosis (separate finding types) | Labels are binary in the curated set. |
| Clinical validation or regulatory claims | This is an engineering comparison, not a diagnostic device study. |
| Data augmentation research, hyper-parameter search beyond the small preset grid | Both would add confounders. Each model gets the same modest, fixed budget. |
| Quantisation / pruning / TFLite deployment | Nice future work. At most we may report one INT8 export size. |
| Patient-level de-duplication beyond what metadata allows | `labels.json` has no patient ID. The risk is documented, not solved (see Risks). |

## 3.5 Deliverables

1. **Codebase** (this repo): reproducible from raw videos to final tables.
2. **Results table**: every model × T, accuracy metrics (mean ± SD over folds) and compute
   metrics.
3. **Figures**:
   - AUROC vs latency Pareto front for each T
   - latency vs T curves
   - training-time bars
   - ROC curves
4. **Hypothesis-test report** for H₀-acc, H₀-cpu and H₀-T.
5. **Final presentation** (week of Nov 2). **Stretch:** hybrid model and a short paper draft
   (week of Nov 9).

## 3.6 Timeline (re-planned on 23 Oct)

| Dates | Work | Exit criterion |
|---|---|---|
| Oct 23–25 | Feature extraction (T=40, subsampled to 20 and 10), saved splits, training loop | `data/features/ear_T40.npz` exists; one fold of `lstm_base` trains end-to-end |
| Oct 26–29 | Full training grid (models × 3 T × 2 seeds × 3 folds, a few minutes each on cached features) | `results/*/run.json` complete |
| Oct 29–Nov 1 | Benchmark harness and the full compute grid | `results/benchmark.csv` complete |
| Nov 2–6 | Statistics, plots, write-up, slides | Final presentation ready |
| Nov 9 (buffer) | Hybrid, paper draft | Optional |

## 3.7 Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Patient leakage**: several videos from the same child (retakes) landing in different folds | Inflated accuracy for *both* families. The comparison stays mostly fair, but the absolute numbers become optimistic | Document it. If any grouping signal exists (consecutive IDs with the same age), test a grouped split as a sensitivity analysis |
| Class imbalance (26.4% Abnormal; 231 positives) | A model can look good on accuracy by predicting all-Normal | AUROC and balanced metrics, class weights, stratified folds |
| Small n gives noisy fold scores (77 Abnormal per test fold, 23 in the inner val) | Underpowered tests | Repeated CV (2 seeds), report CIs and effect sizes, use an equivalence test |
| CPU timing noise (laptop turbo, background apps) | Unreliable latency | Plugged in, fixed threads, warm-up, ≥300 timed runs, fresh process per model, report median and IQR |
| Transformer underfits or overfits on small data | Unfair comparison | Pre-LN, dropout, AdamW and warm-up. Both families get the same early-stopping budget |
| Schedule slip (we are behind the original timeline) | Missing benchmark week | Caching features cuts training from hours to minutes per run. Cut the hybrid first |
