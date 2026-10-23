# Compute vs. Accuracy Trade-off in LSTMs and Transformers for Medical Video Classification

A controlled study of **how much accuracy you buy, and at what CPU cost**, when you replace an
LSTM temporal model with a Transformer encoder for classifying short clinical screening videos as
**Normal vs. Abnormal**.

> **Status (23 Oct 2026):** Design phase complete. Repo structure, research docs and model
> definitions (LSTM / Transformer / Hybrid) are in place and build correctly. Next up are feature
> extraction, the training loop and the benchmark harness. See [Roadmap](#roadmap--status).

---

## TL;DR

- **Data.** Pediatric otoscopy (ear) videos from an existing curated screening pipeline: **874 clips**,
  74% Normal / 26% Abnormal, 480×640, 5–43 s long. A 535-clip oral-cavity set is available as a
  secondary dataset. This is private medical data and is **never committed**.
- **Controlled comparison.** Every model gets the *same* input: T frame embeddings from a **frozen
  ImageNet MobileNetV2** (1280-d per frame), extracted once and cached. Only the **temporal block**
  differs (LSTM, Transformer encoder, or a hybrid). The classifier head is the same everywhere.
- **Measurements.** On a **CPU-only** laptop (i5-12450H, 16 GB):
  - accuracy metrics: AUROC, macro-F1, balanced accuracy, sensitivity/specificity
  - compute metrics: params, FLOPs, batch-1 latency, throughput, peak RAM, training time, model size
  - all of the above across sequence lengths T ∈ {10, 20, 40}
- **Hypothesis.** H₀: with frozen CNN features on short clips, a Transformer head is **not** more
  accurate than an LSTM head. The compute side is tested rather than assumed (see
  [docs/03_scope.md](docs/03_scope.md)).

## Project documents

| # | Doc | What's in it |
|---|-----|--------------|
| 1 | [docs/01_summary.md](docs/01_summary.md) | One-page summary of the whole project |
| 2 | [docs/02_abstract.md](docs/02_abstract.md) | Paper-style abstract (structured + plain) |
| 3 | [docs/03_scope.md](docs/03_scope.md) | Research questions, hypotheses, in/out of scope, deliverables, risks |
| 4 | [docs/04_why_it_matters.md](docs/04_why_it_matters.md) | Motivation: clinical, compute and research reasons |
| Design | [docs/design/data_pipeline.md](docs/design/data_pipeline.md) | Video → frames → MobileNetV2 features → cached `.npz`, plus splits |
| Design | [docs/design/lstm_design.md](docs/design/lstm_design.md) | LSTM family: equations, presets, param maths, training recipe |
| Design | [docs/design/transformer_design.md](docs/design/transformer_design.md) | Transformer family and hybrid: blocks, presets, param/FLOP maths |
| Design | [docs/design/benchmarking_protocol.md](docs/design/benchmarking_protocol.md) | Exactly how compute and accuracy are measured, plus statistics |
| Research | [docs/research/legacy_pipeline_audit.md](docs/research/legacy_pipeline_audit.md) | Review of the old `DL_test1` LSTM code: what we reuse and what we fix |
| Research | [docs/research/literature_review.md](docs/research/literature_review.md) | The 10 proposal papers (checked) plus background (ViViT, minLSTM, …) |
| Context | [CLAUDE.md](CLAUDE.md) / [AGENTS.md](AGENTS.md) | Full working context for AI assistants and new teammates |
| Slides | [docs/presentations/Progress_Review_2026-10-23.pdf](docs/presentations/Progress_Review_2026-10-23.pdf) | 12-slide progress review (source: `progress_review/slides.html`, build with `build.sh`) |

## Architecture at a glance

```
 video (.mp4, 480x640, ~14 s)
   │  uniform-sample T frames, BGR→RGB, resize 224², mobilenet_v2.preprocess_input
   ▼
 MobileNetV2 (ImageNet, frozen, pooling="avg")          ← run ONCE, cached to data/features/
   ▼
 frame embeddings  (T, 1280)
   ▼
 ┌──────────────────────── temporal block (the ONLY thing that varies) ───────────────────────┐
 │ LSTM:         LSTM(256, seq) → LSTM(128)                                                    │
 │ Transformer:  Dense(1280→d) + learned pos-emb → N × [pre-LN MHA + FFN] → LN → mean-pool     │
 │ Hybrid:       LSTM(128, seq) → 1 × Transformer block → mean-pool                            │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
   ▼
 Dense(64, relu) → Dropout(0.5) → Dense(1, sigmoid)          (same head everywhere)
```

### Model presets (built and checked, Keras 3.6 / TF 2.17)

| Preset | Temporal block | Params (T=20) |
|---|---|---:|
| `lstm_small` | LSTM(128) | 729,729 |
| `lstm_base` | LSTM(256) → LSTM(128) *(the original DL_test1 head)* | 1,779,329 |
| `bilstm_base` | BiLSTM(128) → BiLSTM(64) | 1,615,489 |
| `transformer_small` | d=128, 4 heads, FFN 256, 2 layers | 440,065 |
| `transformer_base` | d=256, 4 heads, FFN 512, 2 layers | 1,404,289 |
| `transformer_deep` | d=256, 8 heads, FFN 512, 4 layers | 2,458,497 |
| `hybrid_base` *(stretch)* | LSTM(128) → 1 Transformer block | 862,465 |

Regenerate the table with `python -m scripts.model_summary`.

## Repository structure

```
lstm-transformer-tradeoff/
├── README.md                    ← you are here
├── CLAUDE.md                    ← deep project context for Claude Code / AI assistants
├── AGENTS.md                    ← same context, generic agent format (Codex, Cursor, …)
├── requirements.txt
├── .gitignore
├── configs/                     ← one YAML per experiment family (data, model presets, training, benchmark)
│   ├── data.yaml
│   ├── lstm.yaml
│   ├── transformer.yaml
│   └── benchmark.yaml
├── docs/
│   ├── 01_summary.md  02_abstract.md  03_scope.md  04_why_it_matters.md
│   ├── design/                  ← data_pipeline, lstm_design, transformer_design, benchmarking_protocol
│   └── research/                ← legacy_pipeline_audit, literature_review
├── src/
│   ├── models/                  ← (done) heads.py, lstm.py, transformer.py, hybrid.py, registry (__init__)
│   ├── data/                    ← (todo) video_io.py, extract_features.py, splits.py
│   ├── training/                ← (todo) train.py (k-fold loop, callbacks, metrics)
│   └── benchmark/               ← (todo) latency.py, flops.py, memory.py, report.py
├── scripts/                     ← CLI entry points (model_summary.py, done)
├── data/                        ← GITIGNORED: raw/ (videos + labels.json), features/ (cached .npz)
├── results/                     ← run outputs: metrics JSON/CSV committed, weights gitignored
├── notebooks/                   ← analysis and plots
└── tests/                       ← smoke tests (shapes, param counts, determinism)
```

## Quick start

```bash
# 1. Environment (Python 3.11, CPU-only TensorFlow)
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Check that the models build
python -m scripts.model_summary

# 3. (coming next) put data in place and extract features
#    data/raw/ear/labels.json + data/raw/ear/video-*.mp4
#    python -m src.data.extract_features --dataset ear --frames 40
```

## Roadmap & status

The proposal's timeline, adjusted to where we actually are on 23 Oct:

| Week of | Phase (proposal) | Status / plan |
|---|---|---|
| Oct 5 | Architecture finalisation | Done: design docs and model code in this repo |
| Oct 12 | Data preprocessing | Partial: Data is curated (DL_test1). The fixed preprocessing and feature cache still need code |
| Oct 19 | Training | **Current focus**: feature extraction → splits → k-fold training loop |
| Oct 26 | Benchmarking | Pending: Latency, FLOPs, memory, train-time harness. Run the full grid |
| Nov 2 | Final analysis | Pending: Stats tests, Pareto plots, write-up, presentation |
| Nov 9 | Buffer / paper | Pending: Hybrid model and paper draft if time permits |

## Constraints (from the proposal)

1. **One video type only** (ear/otoscopy is primary; oral is optional replication).
2. **CPU-only** benchmarking.
3. Built around a **null hypothesis** grounded in prior work.
4. **Hybrid model** only if time permits.

## Data & ethics note

The videos are pediatric clinical screening recordings. They stay **off GitHub**: `data/`, `*.mp4`,
model weights and cached features are all gitignored. Only aggregate metrics are committed.
