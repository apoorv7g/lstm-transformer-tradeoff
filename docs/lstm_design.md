# Design: LSTM Temporal Heads

Code: `src/models/lstm.py`. Shared head: `src/models/heads.py`. Config: `configs/lstm.yaml`.

---

## 1. Lineage

The LSTM side of this study **is** the DL_test1 model, with the bugs fixed. The legacy model was:

```python
Sequential([
    TimeDistributed(MobileNetV2(include_top=False, weights="imagenet", frozen)),  # per frame
    TimeDistributed(GlobalAveragePooling2D()),                                     # → (20, 1280)
    LSTM(256, return_sequences=True),
    LSTM(128),
    Dense(64, relu), Dropout(0.5), Dense(1, sigmoid),
])  # Adam(1e-3), BCE, batch 16, 30 epochs, KFold(5)
```

Our `lstm_base` preset reproduces the temporal block and head **exactly**. The only difference is
that the frozen backbone runs once offline, with correct preprocessing (see `data_pipeline.md`).

DL_test1 also contained a "MiniLSTM" (`minLSTM/minLSTM.py`, `miniLSTM_miniGRU.py`). It is **not**
the minLSTM of Feng et al. (2024). It is a hand-written *standard* LSTM cell: it keeps the output
gate, the tanh, and the h_{t−1} term inside every gate. Its parameter count equals a Keras LSTM's,
and it is slower because it runs in a Python-level RNN loop. It is not carried over. A *true*
minLSTM is listed below as an optional extra baseline.

## 2. The LSTM cell (what we pay for)

For input x_t ∈ ℝ^{d_in} and hidden size h:

```
i_t = σ(W_i x_t + U_i h_{t-1} + b_i)        input gate
f_t = σ(W_f x_t + U_f h_{t-1} + b_f)        forget gate
g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)     candidate
o_t = σ(W_o x_t + U_o h_{t-1} + b_o)        output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

- **Parameters per layer:** `4 · (d_in·h + h² + h)`.
- **MACs per time step:** about `4 · (d_in·h + h²)`. **Per clip:** `T` times that.
- **Sequential depth:** T steps per layer. Step t cannot start before step t−1 finishes. This
  causes poor multi-core use at batch 1 on CPU.
- **Activation memory at inference:** O(h) per layer, the smallest of any model here.

## 3. Presets

| Preset | Layers | Params (verified) | MACs/clip T=10 | T=20 | T=40 | Role |
|---|---|---:|---:|---:|---:|---|
| `lstm_small` | LSTM(128) | 729,729 | 7.2 M | 14.4 M | 28.8 M | Cheapest reasonable recurrent model |
| `lstm_base` | LSTM(256,seq) → LSTM(128) | 1,779,329 | 17.7 M | 35.4 M | 70.8 M | **Legacy baseline**, parameter reference |
| `bilstm_base` | BiLSTM(128,seq) → BiLSTM(64) | 1,615,489 | 16.1 M | 32.1 M | 64.2 M | Does seeing "future" frames help? |

Worked parameter count for `lstm_base`:
- LSTM1: 4·(1280·256 + 256² + 256) = **1,573,888**
- LSTM2: 4·(256·128 + 128² + 128) = **197,120**
- Head: (128·64 + 64) + (64 + 1) = **8,321**
- Total **1,779,329** (matches `model.count_params()`)

The MAC numbers are analytic: they count the matrix multiplies and ignore element-wise ops. About
**89% of `lstm_base`'s parameters and compute sit in the first layer's 1280→4·256 input kernel.**
The input dimension, not the recurrence, drives the cost. The Transformer has the same property
(its 1280→d projection).

Parameters do not depend on T, while MACs grow linearly with T.

### Optional extra: true minLSTM (Feng et al., 2024, arXiv:2410.01201)

```
f_t = σ(Linear(x_t)),  i_t = σ(Linear(x_t)),  h̃_t = Linear(x_t)
f'_t, i'_t = f_t/(f_t+i_t), i_t/(f_t+i_t)
h_t = f'_t ⊙ h_{t-1} + i'_t ⊙ h̃_t
```

- There is no h_{t−1} inside the gates, so parameters are about `3·(d_in·h + h)` and there is no
  h² term.
- It could be computed with a parallel scan, though at T ≤ 40 a simple loop is fine.
- It would add a "maximally cheap recurrence" point on the Pareto plot. Build it only if the main
  grid finishes early.

## 4. Implementation notes (things that silently change CPU timing)

- Use the built-in `keras.layers.LSTM` with default activations (`tanh` / `sigmoid`),
  `recurrent_dropout=0`, `unroll=False`, `use_bias=True`.
  - Any deviation disables the fused kernel and makes the LSTM look artificially slow.
  - This matters even on CPU: TF has an optimised standard-LSTM path that is selected under these
    defaults.
- Input dropout (`dropout=`) is allowed and keeps the fast path. Default 0.0 to match legacy. The
  head's Dropout(0.5) is the main regulariser.
- Do **not** wrap the model in `TimeDistributed(backbone)` for training. The features are
  precomputed. The end-to-end latency benchmark builds a separate backbone+head graph.
- Save models with `model.save("*.keras")` (Keras 3 format), not legacy `.h5`.

## 5. Training recipe (shared budget; see also `transformer_design.md` §6)

| Setting | Value | Note |
|---|---|---|
| Loss | Binary cross-entropy | Abnormal = 1 |
| Class weights | `balanced` (≈ 0.59 Normal / 3.22 Abnormal on oral) | Legacy had none |
| Optimiser | Adam, lr 1e-3, `clipnorm=1.0` | Clipping is standard for RNNs |
| Batch size | 32 | Cached features make this cheap |
| Max epochs | 100 | |
| Early stopping | monitor `val_auc` (inner val), patience 15, `restore_best_weights=True` | |
| LR schedule | ReduceLROnPlateau(factor 0.5, patience 5) on `val_auc` | |
| Seed | `keras.utils.set_random_seed(seed)` per run | |
| Threshold | 0.5 for reported metrics. Also report the Youden-optimal threshold chosen on inner val | |

**Small LR check (optional, equal for both families):** try lr ∈ {1e-3, 3e-4} on the inner
validation set of seed 0 only, then fix the value for all runs.

## 6. Expected behaviour / what to watch

- On about 364 training clips per fold, `lstm_base` (1.78 M params) can overfit within about 10–20 epochs.
  Watch train vs inner-val AUC, and expect early stopping to fire early.
- BiLSTM doubles the sequential work per layer (two directions). Expect the **worst batch-1 CPU
  latency** among the LSTMs, despite fewer parameters than `lstm_base`.
- Smoke test (eager mode, batch 1, T = 20, untuned; **not a result**): `lstm_small` ≈ 190 ms,
  `lstm_base` ≈ 375 ms, `bilstm_base` ≈ 685 ms. Eager mode adds large per-op overhead. The real
  benchmark uses compiled `tf.function` calls (see `benchmarking_protocol.md`).
