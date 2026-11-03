# 2. Abstract

> Written before the experiments and completed with the results of the main experiment
> (4 Nov 2026). Full numbers: `docs/05_results.md` and `results/results_3fold_T20.md`.

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

- *Data:* 874 pediatric ear (otoscopy) screening videos (643 Normal, 231 Abnormal; 480×640;
  median length 13.1 s).
  From each video, T = 20 frames are sampled uniformly and encoded once by a frozen ImageNet
  MobileNetV2, giving 1280-d per-frame embeddings.
- *Models:* on these identical inputs we train seven temporal models:
  - an LSTM (`lstm_base`, 1.78 M parameters)
  - a pre-LayerNorm Transformer encoder parameter-matched to it (`transformer_base`, 1.40 M)
  - three LSTM→attention hybrids, two of them with attention plus max pooling (0.16–0.86 M)
  - two no-temporal controls that average or max-pool the frames (0.08 M)

  All share the same classification head, optimiser budget and early-stopping rule.
- *Accuracy:* evaluated with stratified 3-fold cross-validation repeated with 2 seeds (AUROC, macro-F1, balanced
  accuracy, sensitivity, specificity).
- *Compute:* measured on an Intel i5-12450H CPU with fixed thread counts (parameters, FLOPs, median
  and p95 batch-1 latency, throughput, peak resident memory, training time, model size), both for
  the temporal head alone and end-to-end.
- *Statistics:* differences are tested with the corrected resampled t-test, with effect sizes and a
  pre-registered equivalence margin of Δ = 0.03 AUROC.

**Results.** Over 6 paired test folds, the Transformer and the LSTM reached AUROC 0.905 ± 0.011
and 0.907 ± 0.013 (Δ = -0.001, p = 0.87; 90% CI -0.019 to +0.016, inside the equivalence margin).
All seven models scored AUROC 0.893–0.909 and none differed significantly from the LSTM; even the
frame-averaging control was equivalent (0.900). Batch-1 latency of the temporal head on one CPU
thread was 1.36 ms for the Transformer and 3.54 ms for the LSTM (2.6 times faster); at batch 32 on
8 threads the two tied. End to end, video decoding and MobileNetV2 took over 99% of the time per
clip (about 462 of 465 ms). The Pareto-optimal models were `pool_mean`, `hybrid_lite` and
`hybrid_attn`. The effect of T was not tested.

**Conclusions.** H₀-acc is supported (the families are equivalent in accuracy) and H₀-cpu holds
(the Transformer is faster, not slower). There is no compute-versus-accuracy trade-off between the
two families on this task: the cheaper Transformer loses nothing. Because the temporal head is
under 1% of end-to-end cost, deployments should save compute through fewer frames or a lighter
backbone; an attention-pooling hybrid (`hybrid_attn`) gave the best accuracy and screening
sensitivity at a third of the LSTM's latency.

**Keywords:** medical video classification, LSTM, Transformer, self-attention, compute efficiency,
CPU inference, otoscopy, ear screening, MobileNetV2, accuracy–efficiency trade-off.

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

*On short, small-sample medical videos and a CPU-only budget, a Transformer temporal model is as
accurate as an LSTM and 2.6 times faster, but both are under 1% of the end-to-end cost.*
