# 2. Abstract

> Draft abstract, written before the experiments. Values in ⟨angle brackets⟩ are placeholders to
> fill in once the results exist. Do not report them as findings until then.

---

## Structured abstract

**Background.** Clinical video classification systems usually combine a 2D convolutional network,
which encodes each frame, with a temporal model, which aggregates the frames. Long Short-Term Memory
(LSTM) networks and Transformer encoders are the two most common temporal models. Transformers
dominate large-scale video benchmarks. Medical video datasets, however, are small, imbalanced and
often deployed on commodity CPUs, and for that setting it is unclear whether the Transformer's added
capacity becomes better accuracy, or what it costs.

**Objective.** To quantify the trade-off between compute and classification accuracy for LSTM and
Transformer temporal models on short medical screening videos, under a CPU-only deployment
constraint.

**Methods.**

- *Data:* 874 pediatric otoscopy videos (643 Normal, 231 Abnormal; 480×640; median length 13.8 s).
  From each video, T ∈ {10, 20, 40} frames are sampled uniformly and encoded once by a frozen
  ImageNet MobileNetV2, giving 1280-d per-frame embeddings.
- *Models:* on these identical inputs we train
  - three LSTM variants (0.73–1.78 M parameters)
  - three pre-LayerNorm Transformer-encoder variants (0.44–2.46 M parameters, one
    parameter-matched to the LSTM baseline)
  - optionally, an LSTM→Transformer hybrid

  All share the same classification head, optimiser budget and early-stopping rule.
- *Accuracy:* evaluated with stratified 5-fold cross-validation (AUROC, macro-F1, balanced
  accuracy, sensitivity, specificity).
- *Compute:* measured on an Intel i5-12450H CPU with fixed thread counts (parameters, FLOPs, median
  and p95 batch-1 latency, throughput, peak resident memory, training time, model size), both for
  the temporal head alone and end-to-end.
- *Statistics:* differences are tested with the corrected resampled t-test, with effect sizes and a
  pre-registered equivalence margin of Δ = 0.03 AUROC.

**Results.** ⟨Transformer vs LSTM AUROC: x ± y vs x ± y; Δ = …, p = …⟩.
⟨Head-only batch-1 latency: … ms vs … ms; end-to-end latency dominated by MobileNetV2 (…%)⟩.
⟨Effect of T⟩. ⟨Pareto-optimal models⟩.

**Conclusions.** ⟨Whether H₀ is rejected. A practical recommendation for model choice in small,
short-clip medical video tasks on CPU⟩.

**Keywords:** medical video classification, LSTM, Transformer, self-attention, compute efficiency,
CPU inference, otoscopy, MobileNetV2, accuracy–efficiency trade-off.

---

## Plain-language abstract

Doctors and screening programmes increasingly use AI to read short medical videos, for example a
camera looking inside a child's ear. Such a system first looks at each frame on its own, then
combines what it saw across the frames to reach a decision. There are two popular ways to do the
"combining" step:

- an **LSTM**, which reads the frames one after another like reading a sentence
- a **Transformer**, which compares all the frames with each other at once

Transformers are the newer and more fashionable choice. They are also thought to be more powerful
but heavier to run.

We test whether that reputation holds when the dataset is small (hundreds of videos, not millions)
and the computer is an ordinary laptop with no graphics card. We give both models exactly the same
frame information and measure how accurate each one is, and how fast, how big and how
memory-hungry. The result tells a clinic with modest hardware which model is worth running.

---

## One-sentence version

*On short, small-sample medical videos and a CPU-only budget, does a Transformer temporal model
beat an LSTM by enough to justify its cost? We measure both sides of that question under controlled
conditions.*
