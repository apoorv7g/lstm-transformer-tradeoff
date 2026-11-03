# 4. Why This Matters

## 4.1 The clinical reason: AI has to run where the patients are

Screening programmes for school children, primary-care clinics and community health camps record
short videos with low-cost devices such as digital otoscopes and other handheld cameras. The data in
this project comes from exactly that kind of pipeline: pediatric ear (otoscopy) screening, with
children aged 3–18 (median 11).

Places like these typically have:

- **no GPU**: a laptop, a mini-PC or a phone
- **many videos per day and little patience**: a result is needed in seconds, at batch size 1
- **imbalanced data**: most children are normal, and the abnormal minority is what matters

If a model only becomes accurate after it is too slow or too large to run there, it does not help.
Choosing the temporal architecture is therefore a **deployment decision**, not just an accuracy
decision. Practitioners need evidence about the trade-off measured on the hardware they actually
have. That evidence is surprisingly rare, because most papers report accuracy on GPUs.

## 4.2 The compute reason: the "Transformers are expensive" intuition may be wrong here

The usual story says LSTMs are cheap and sequential, while Transformers are powerful but cost
O(T²). For **our** regime that story is incomplete, for three reasons.

1. **Short sequences.** Attention-score FLOPs (≈ 2·T²·d) only exceed projection and FFN FLOPs once
   T > 2d + d_ff, which is about **512 frames** for d = 128. Our clips use T = 10–40. The quadratic
   term is a rounding error.
2. **CPU execution model.**
   - An LSTM at batch 1 runs **T dependent small matrix–vector steps** in a loop. Each step waits
     for the previous one, so it parallelises poorly over cores and pays loop overhead per step.
   - A Transformer runs **a few large matrix multiplies** that oneDNN/MKL spreads well across
     cores.
   - A quick smoke test on our machine (eager mode, T = 20, batch 1) gave LSTM heads of about
     190–375 ms and Transformer heads of about 115–220 ms. Eager timings are noisy and *not* a
     result, but they show why we must measure rather than assume.
3. **The backbone dominates anyway.**
   - MobileNetV2 costs about 300 M multiply-adds per 224² frame, so about 6 G per 20-frame clip.
   - The temporal head costs about 10–40 M.
   - End to end, the head may be **under 1%** of the compute. If so, the clinically relevant
     conclusion is "choose the head on accuracy alone, and optimise the backbone if you need
     speed". That is also a useful finding.

A study that measures this properly on CPU, at both head-only and end-to-end level, tests a
widely held assumption.

## 4.3 The research reason: the literature disagrees, and small-data medical video sits in the gap

| Evidence | What it suggests |
|---|---|
| Large-scale video benchmarks (ViViT, TimeSformer, …) | Attention wins *with lots of data and pretraining* |
| Soeleman et al. 2022: violence detection, InceptionV3 features + LSTM vs Transformer | **LSTM ≥ Transformer** on three small datasets |
| Barros et al. 2021: lung ultrasound, 185 videos, Xception + LSTM, 20 frames | CNN+LSTM already reaches ~93% accuracy on tiny medical sets |
| Keras "Video Classification with Transformers" example | Its own authors note it "works best with a larger dataset" |
| OperA 2021, Trans-SVNet 2021 (surgical video) | Transformers help on **long** surgical videos with thousands of frames |
| LREC 2022, limited-data LM study | Transformer gains vanish on the smaller corpus |

Short, small, imbalanced medical clips fall between these results. Nobody has settled which side of
the line they are on. A **controlled** comparison answers it:

- frozen identical features
- identical head, folds and budget
- parameter-matched variants
- proper statistics

Many published comparisons change the backbone, the preprocessing *and* the temporal model all at
once, so they cannot attribute the difference.

## 4.4 The methodological reason: doing the comparison properly is itself a contribution

Our audit of the previous LSTM pipeline (`docs/legacy_pipeline_audit.md`) found problems
that are common in student and early-stage medical-AI work:

- ImageNet models fed with the wrong input scaling and channel order
- confusion matrices computed on data that included the training set
- accuracy reported on a 74/26 imbalanced set, where 73.6% is free
- non-stratified folds
- no fixed seeds and no compute measurement at all

This project makes the following standard: one feature cache, saved folds, AUROC and balanced
metrics, fixed-thread latency protocol, corrected resampled tests. The resulting codebase is
reusable for *any* "which temporal model?" question on clinical video.

## 4.5 Who benefits

- **Practitioners and health-tech teams** deploying on CPUs. They get a concrete recommendation
  ("for ≤40-frame clips with frozen features, use X; expect Y ms per clip on a mid-range laptop").
- **Researchers.** They get a reproducible, controlled baseline and a measured answer on where
  attention starts paying off.
- **Us (the team).** We get a rigorous end-to-end research workflow: hypothesis → controlled
  experiment → statistics → write-up. A future paper and a better screening model for the original
  pipeline can build on it.
