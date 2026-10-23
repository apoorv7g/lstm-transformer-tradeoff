# Design: Transformer Temporal Heads (+ Hybrid)

Code: `src/models/transformer.py`, `src/models/hybrid.py`. Config: `configs/transformer.yaml`.

---

## 1. Lineage and reference design

We follow the **"factorised encoder" / late-fusion** pattern:

- ViViT model 2 (Arnab et al., 2021) uses this pattern: encode each frame spatially, then run a
  *temporal* Transformer over the per-frame tokens.
- So do VTN (Neimark et al., 2021), Trans-SVNet (Gao et al., 2021) and OperA (Czempiel et al.,
  2021) for surgical video.
- So does the Keras example **"Video Classification with Transformers"** (Sayak Paul;
  keras.io/examples/vision/video_transformers). Our code is closest to this one.

The Keras example in brief:

- DenseNet121 features (1024-d), `MAX_SEQ_LENGTH = 20`
- learned `PositionalEmbedding`
- 1 post-LN `TransformerEncoder(embed_dim=1024, dense_dim=4, num_heads=1)`
- `GlobalMaxPooling1D`, `Dropout(0.5)`, softmax
- about 4.23 M params, ≈ 89% test accuracy on 5 UCF101 classes, 594 training videos

We deliberately change four things:

| Keras example | Ours | Why |
|---|---|---|
| No projection; tokens are raw 1024-d CNN features | `Dense(1280 → d_model)` projection | Lets us size the model independently of the backbone and parameter-match the LSTM. In the example, `key_dim = embed_dim = 1024`, so the 4 attention matrices hold about 4.2 M of its 4.23 M params |
| `key_dim = embed_dim` per head | `key_dim = d_model / num_heads` | Standard multi-head split, so params do not depend on the head count |
| Post-LN (`x = LN(x + f(x))`) | **Pre-LN** (`x = x + f(LN(x))`) plus a final LN | Trains more stably without warm-up gymnastics on small data |
| `GlobalMaxPooling1D` | `GlobalAveragePooling1D` (max available as an option) | Lower variance on small data; a CLS token would be one more thing to learn from ~600 clips |
| FFN width 4 (a bottleneck) | FFN = 2 × d_model, GELU | Conventional width |

Positional encoding: **learned** absolute embeddings (T × d params), as in the Keras example and
ViT. At T ≤ 40 sinusoidal encoding is equally reasonable. We fix *learned* and note it. A swap is
a one-line ablation if time permits.

## 2. Block definition

```
tokens  = Dense(d)(features) + PosEmb[0:T]          # (T, d)
tokens  = Dropout(p)(tokens)
repeat N times:
    tokens = tokens + Dropout(MHA(LN(tokens)))      # h heads, key_dim = d/h, attn-dropout p
    tokens = tokens + Dropout(W2·GELU(W1·LN(tokens)))  # W1: d→d_ff, W2: d_ff→d
tokens  = LN(tokens)
pooled  = mean_t(tokens)                            # (d,)
output  = Dense(64,relu) → Dropout(0.5) → Dense(1,sigmoid)     # shared head
```

There is no attention mask, because every clip has exactly T real frames (short clips are
repeat-padded during sampling, not zero-padded).

## 3. Parameter maths

For one encoder block with model width d, FFN width d_ff:

- MHA (Q, K, V, O weights and biases): `4d² + 4d`
- FFN: `2·d·d_ff + d_ff + d`
- 2 × LayerNorm: `4d`
- **Block ≈ 4d² + 2·d·d_ff + 9d + d_ff**

Extras:
- projection `1280·d + d`
- positional `T·d`
- final LN `2d`
- head `d·64 + 64 + 65`

## 4. Presets (verified with `model.count_params()`, Keras 3.6)

| Preset | d | heads | d_ff | layers | Params T=10 | T=20 | T=40 | Role |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `transformer_small` | 128 | 4 | 256 | 2 | 438,785 | 440,065 | 442,625 | Cheapest attention model |
| `transformer_base` | 256 | 4 | 512 | 2 | 1,401,729 | 1,404,289 | 1,409,409 | **Parameter-matched** to `lstm_base` (1.40 M vs 1.78 M) |
| `transformer_deep` | 256 | 8 | 512 | 4 | 2,455,937 | 2,458,497 | 2,463,617 | Does depth buy accuracy? |
| `hybrid_base` | 128 (LSTM) | 4 | 256 | 1 | 862,465 | 862,465 | 862,465 | Stretch: LSTM(128,seq) → 1 block |

Parameters grow slightly with T, only through the learned positional table (T·d).

## 5. Compute maths: why "attention is O(T²)" barely matters here

Analytic MACs per clip, counting matmuls only:

- projection: `T·1280·d`
- per layer:
  - `4·T·d²` (QKVO)
  - `2·T²·d` (QKᵀ and A·V)
  - `2·T·d·d_ff` (FFN)

| Preset | T=10 | T=20 | T=40 | of which attention-score (T²) at T=40 |
|---|---:|---:|---:|---:|
| `transformer_small` | 4.3 M | 8.7 M | 17.9 M | 0.8 M (4.6%) |
| `transformer_base` | 13.9 M | 27.9 M | 56.7 M | 1.6 M (2.9%) |
| `transformer_deep` | 24.5 M | 49.3 M | 100.3 M | 3.3 M (3.3%) |
| `hybrid_base` | 8.5 M | 17.1 M | 34.5 M | 0.4 M |
| *cf.* `lstm_base` | 17.7 M | 35.4 M | 70.8 M | n/a |
| *cf.* MobileNetV2 backbone (≈300 M per frame) | ≈3.0 G | ≈6.0 G | ≈12.0 G | n/a |

Takeaways for the write-up:

1. **The T² term crosses the linear terms only when T > 2d + d_ff**, which is 512 frames at
   d = 128 and 1024 at d = 256. Our T ≤ 40 is far below that.
2. At matched parameters, the Transformer does **fewer** MACs per clip than the LSTM
   (27.9 M vs 35.4 M at T = 20).
3. More importantly for CPU **latency**:
   - The Transformer's work is a handful of large GEMMs with no dependency across time.
   - The LSTM's work is T dependent small GEMVs.
   - Prediction to test: the Transformer is *not* slower at batch 1 on CPU. This is H₀-cpu in
     `docs/03_scope.md`.
4. Both heads are **< 1%** of end-to-end MACs once MobileNetV2 is included. End-to-end latency
   differences between heads may be within noise. That is a finding to report, not hide.

## 6. Training recipe

| Setting | Value | Differs from LSTM? |
|---|---|---|
| Loss / labels / class weights / batch / max epochs / early stopping / seeds | **identical** to `lstm_design.md` §5 | No |
| Optimiser | AdamW, lr 3e-4, weight_decay 1e-2 (excluding LN and bias if easy), `clipnorm=1.0` | Yes. Standard Transformer recipe |
| LR schedule | linear warm-up for 5 epochs, then ReduceLROnPlateau as for the LSTM | Yes. Warm-up is standard for attention |
| Dropout | 0.1 in embedding, attention and FFN; 0.5 in the shared head | n/a |

**Fairness argument.** Each family gets its *conventional* optimiser and the *same* epoch budget,
stopping rule and LR-sensitivity check (lr ∈ {1e-3, 3e-4} on seed-0 inner validation). Forcing
Adam(1e-3) on a Transformer would handicap it, so that would not be fair either. The paper should
state this choice explicitly.

## 7. Hybrid (stretch goal, `hybrid_base`)

`LSTM(128, return_sequences=True) → TransformerEncoderBlock(d=128, 4 heads, d_ff=256) × 1 → LN → mean-pool → head`

- The LSTM supplies order and recency information, so **no positional embedding** is used.
- Attention then re-weights the time steps globally, in the spirit of OperA's frame-importance
  attention.
- It sits between `lstm_small` and `transformer_small` in cost. It is interesting only if it
  improves on the Pareto front.
- The literature has related designs, though not identical ones: TSTNet uses a Swin spatial
  encoder with an LSTM, and 3DTransLSTM interleaves the two.

## 8. Expected behaviour / what to watch

- Small data plus attention gives a high variance across folds. Expect a larger SD than the LSTM.
  Repeated CV (3 seeds) matters most here.
- If `transformer_deep` ≤ `transformer_base` on AUROC, that supports "capacity is not the
  bottleneck", which is consistent with H₀.
- Inspect attention maps for a few clips (mean over heads). Do they focus on a few frames, for
  example when the eardrum is in view? This is a qualitative figure for the presentation.
- Smoke test (eager, batch 1, T = 20; **not a result**): `transformer_small` ≈ 116 ms,
  `transformer_base` ≈ 130 ms, `transformer_deep` ≈ 217 ms, `hybrid_base` ≈ 245 ms.
