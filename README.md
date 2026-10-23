# Compute vs. Accuracy Trade-off in LSTMs and Transformers for Medical Video Classification

A controlled study of the trade-off between classification accuracy and CPU compute cost when an
LSTM temporal model is replaced by a Transformer encoder, applied to short clinical screening
videos labelled **Normal** or **Abnormal**.

**Status (23 Oct 2026):**
- Done: design phase, and the model definitions are implemented and verified.
- In progress: feature extraction and the training pipeline.
- See [Project status](#project-status).

---

## Overview

Medical video classifiers usually combine a per-frame convolutional network with a temporal model
that aggregates information across frames. The two dominant temporal models are:

- **LSTMs**, which process frames sequentially and are compact.
- **Transformers**, which relate all frames to each other through self-attention. They are
  generally more accurate on large datasets, but are assumed to be more expensive.

Clinical screening deployments face three constraints that make that comparison unclear:

- datasets are small and imbalanced
- clips are short
- inference often runs on ordinary CPUs

This project measures both sides of the trade-off under controlled conditions:

- **Accuracy:** AUROC, macro-F1, balanced accuracy, sensitivity and specificity, under
  stratified cross-validation.
- **Compute:** parameters, FLOPs, inference latency, throughput, peak memory, training time and
  model size, measured on a CPU-only machine.

The study is built around a null hypothesis taken from prior work. On short clips represented by
frozen CNN features, a Transformer temporal head is **not** more accurate than an LSTM head. The
compute side is measured rather than assumed. Our analysis suggests that, at these sequence
lengths, the usual "Transformers are slower" intuition may not hold on CPU.

Full motivation and scope:
[summary](docs/01_summary.md), [abstract](docs/02_abstract.md), [scope](docs/03_scope.md) and
[why it matters](docs/04_why_it_matters.md).

## Research questions

| ID | Question |
|---|---|
| RQ1 | With identical frozen frame features, is a Transformer temporal head more accurate than an LSTM head? |
| RQ2 | What does each family cost on CPU (parameters, FLOPs, latency, memory, training time)? Is the temporal head even a significant share of end-to-end cost? |
| RQ3 | How do accuracy and cost scale with clip length T ∈ {10, 20, 40}? |
| RQ4 | Which models are Pareto-optimal in accuracy vs. latency, and which should be deployed on CPU? |

## Dataset

The primary dataset is pediatric otoscopy (ear) screening video from an existing curated pipeline.

| Property | Value |
|---|---|
| Videos | 874 |
| Labels | 643 Normal / 231 Abnormal (majority-class accuracy 73.6%) |
| Resolution | 480 × 640 (portrait) |
| Duration | median 13.8 s (range 5 to 43 s), mostly ~30 fps |
| Metadata | `labels.json`: `{id, interpretation, age}` |

Two more sets are available for optional replication and robustness checks:
- an oral-cavity set (535 videos)
- a small external ear set (50 videos)

Abnormal is treated as the positive class throughout. The data is private clinical material and
**is not included in this repository**. Details: [data pipeline design](docs/design/data_pipeline.md).

## Method

### Pipeline

```
video (.mp4)
  │  uniform sampling of T frames, sequential decode, BGR → RGB, resize to 224 × 224
  ▼
MobileNetV2 (ImageNet weights, frozen, global average pooling)     run once, features cached
  ▼
frame embeddings (T, 1280)
  ▼
temporal block (the only component that varies between models)
  │  LSTM:         LSTM(256) → LSTM(128)
  │  Transformer:  Dense(1280 → d) + learned positional embedding
  │                → N × [pre-LN multi-head attention + feed-forward] → LayerNorm → mean pooling
  │  Hybrid:       LSTM(128) → one Transformer block → mean pooling
  ▼
classifier head (identical for all models): Dense(64, ReLU) → Dropout(0.5) → Dense(1, sigmoid)
```

### Controls

The comparison is only valid if every model sees the same inputs and the same training budget.
The following are held fixed across all models:

- **Features.** MobileNetV2 runs once at T = 40, and the features are cached. T = 20 and T = 10
  are uniform subsamples of the same frames.
- **Folds.** Stratified 5-fold cross-validation with a stratified 15% inner validation split for
  early stopping. Folds are saved to disk and shared by all models. Up to three seeds are used.
- **Head and training.** Identical classifier head, binary cross-entropy with balanced class
  weights, batch size, epoch budget, and early stopping on validation AUROC.
- **Preprocessing.** `mobilenet_v2.preprocess_input` scaling and RGB channel order, matching the
  ImageNet weights.

### Models

Every model below is implemented and builds correctly. Parameter counts are verified with
TensorFlow 2.17 / Keras 3.6.

| Preset | Temporal block | Parameters (T = 20) | MACs per clip (T = 20) |
|---|---|---:|---:|
| `lstm_small` | LSTM(128) | 729,729 | 14.4 M |
| `lstm_base` | LSTM(256) → LSTM(128) | 1,779,329 | 35.4 M |
| `bilstm_base` | BiLSTM(128) → BiLSTM(64) | 1,615,489 | 32.1 M |
| `transformer_small` | d = 128, 4 heads, FFN 256, 2 layers | 440,065 | 8.7 M |
| `transformer_base` | d = 256, 4 heads, FFN 512, 2 layers | 1,404,289 | 27.9 M |
| `transformer_deep` | d = 256, 8 heads, FFN 512, 4 layers | 2,458,497 | 49.3 M |
| `hybrid_base` (stretch goal) | LSTM(128) → 1 Transformer block | 862,465 | 17.1 M |

- `transformer_base` is the parameter-matched counterpart of `lstm_base`; they form the primary
  comparison.
- For reference, the frozen MobileNetV2 backbone costs about 300 M MACs per frame.

Design rationale, equations and training recipes:
- [LSTM design](docs/design/lstm_design.md)
- [Transformer design](docs/design/transformer_design.md)

### Evaluation

| Aspect | Protocol |
|---|---|
| Accuracy metrics | AUROC (primary), macro-F1, balanced accuracy, sensitivity, specificity, AUPRC; computed only on held-out folds |
| Latency | Compiled `tf.function`, 50 warm-up and ≥ 300 timed calls, median and p95; batch 1 and 32; 1 and 8 threads; one process per model |
| Other compute metrics | Parameters, analytic and profiled FLOPs, peak resident memory, training time per epoch, model file size; head-only and end-to-end |
| Statistics | Corrected resampled t-test, 95% confidence intervals and effect sizes, TOST equivalence (margin 0.03 AUROC), McNemar and DeLong tests, Holm correction |
| Hardware | Intel Core i5-12450H (8 cores, 12 threads), 16 GB RAM, Windows 11, no GPU |

Full protocol: [benchmarking protocol](docs/design/benchmarking_protocol.md).

## Getting started

### Requirements

- Python 3.11
- CPU-only TensorFlow 2.17 / Keras 3.6 (pinned in `requirements.txt`)
- Edge or Chrome, only if you want to rebuild the slides

### Installation

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Git Bash / Linux / macOS:

```bash
python -m venv .venv
source .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Verify the models

From the repository root:

```bash
python -m scripts.model_summary
```

This builds every preset at T = 10, 20 and 40 and prints the parameter counts. They should match
the table above.

### Data placement

Place the dataset locally. The `data/` directory is excluded from version control.

```
data/raw/ear/labels.json
data/raw/ear/video-1.mp4 ... video-874.mp4
```

## Workflow

The study runs as a sequence of steps, each driven by a config file in `configs/`.

| Step | Purpose | Command | Status |
|---|---|---|---|
| 1. Verify models | Build all presets and check parameter counts | `python -m scripts.model_summary` | Implemented |
| 2. Extract features | Sample frames and cache MobileNetV2 embeddings to `data/features/ear_T40.npz` | `python -m src.data.extract_features --dataset ear --frames 40` | Planned |
| 3. Create splits | Write stratified folds to `data/splits/ear_seed{0,1,2}.json` | `python -m src.data.splits --dataset ear` | Planned |
| 4. Train | One run per (preset, T, seed, fold); writes metrics and out-of-fold predictions to `results/runs/` | `python -m src.training.train --preset lstm_base --T 20 --seed 0` | Planned |
| 5. Benchmark | Measure latency, memory and FLOPs on CPU | `python -m src.benchmark.run_grid` | Planned |
| 6. Analyse | Hypothesis tests, summary tables and Pareto plots | `notebooks/analysis.ipynb` | Planned |

Commands marked *Planned* show the intended interface. Those modules are not implemented yet.

Configuration files:

| File | Contents |
|---|---|
| `configs/data.yaml` | Frame sampling, preprocessing, backbone and split settings |
| `configs/lstm.yaml` | LSTM presets and training recipe |
| `configs/transformer.yaml` | Transformer presets and training recipe |
| `configs/benchmark.yaml` | Latency, memory and thread settings for the compute benchmark |

## Repository structure

```
lstm-transformer-tradeoff/
├── configs/                 Experiment configuration (data, LSTM, Transformer, benchmark)
├── data/                    Local only: raw/ (videos), features/ (cache), splits/ (fold IDs)
├── docs/
│   ├── 01_summary.md        Project summary
│   ├── 02_abstract.md       Abstract (results to be filled in)
│   ├── 03_scope.md          Research questions, hypotheses, scope, risks
│   ├── 04_why_it_matters.md Motivation
│   ├── design/              Data pipeline, LSTM, Transformer, benchmarking protocol
│   ├── research/            Literature review, audit of the earlier LSTM baseline
│   └── presentations/       Progress review slides (PDF and HTML source)
├── notebooks/               Analysis notebooks
├── results/                 Metrics and figures (model weights are not committed)
├── scripts/                 Command-line utilities
├── src/
│   ├── models/              Temporal heads: LSTM, Transformer, hybrid, shared head, registry
│   ├── data/                Frame sampling, feature extraction, splits
│   ├── training/            Cross-validation training loop
│   └── benchmark/           Latency, memory and FLOP measurement
├── tests/                   Unit and regression tests
└── requirements.txt
```

## Project status

| Phase | Planned | Status |
|---|---|---|
| Architecture finalisation | Week of Oct 5 | Complete |
| Data preprocessing | Week of Oct 12 | Dataset profiled and audited; extraction code in progress |
| Training | Week of Oct 19 | In progress |
| Benchmarking | Week of Oct 26 | Not started |
| Final analysis | Week of Nov 2 | Not started |
| Paper and hybrid model (buffer) | Week of Nov 9 | Optional |

Progress review slides:
[docs/presentations/Progress_Review_2026-10-23_CacheMemory.pdf](docs/presentations/Progress_Review_2026-10-23_CacheMemory.pdf).

## Scope and limitations

- A single video type (otoscopy) is studied. The oral-cavity set is an optional replication.
- All benchmarks are CPU-only by design. GPU performance is out of scope.
- The CNN backbone is frozen. End-to-end fine-tuning and video-native Transformers (ViViT,
  TimeSformer) are out of scope.
- The metadata has no patient identifier, so leakage between related clips cannot be fully ruled
  out. A grouped-split sensitivity analysis is planned.
- This is an engineering comparison, not a validated diagnostic tool.

## Documentation

| Document | Description |
|---|---|
| [Summary](docs/01_summary.md) | One-page overview |
| [Abstract](docs/02_abstract.md) | Structured abstract |
| [Scope](docs/03_scope.md) | Research questions, hypotheses, deliverables, risks |
| [Why it matters](docs/04_why_it_matters.md) | Clinical, computational and research motivation |
| [Data pipeline](docs/design/data_pipeline.md) | Frame sampling, feature extraction, splits |
| [LSTM design](docs/design/lstm_design.md) | Architecture, parameter and compute analysis, training recipe |
| [Transformer design](docs/design/transformer_design.md) | Architecture, hybrid model, compute analysis |
| [Benchmarking protocol](docs/design/benchmarking_protocol.md) | Measurement procedure and statistics |
| [Literature review](docs/research/literature_review.md) | Related work with verification status |
| [Baseline audit](docs/research/legacy_pipeline_audit.md) | Review of the earlier LSTM baseline and the fixes applied |

## Data and ethics

The videos are pediatric clinical screening recordings. They are never committed. The following
are all excluded by `.gitignore`:
- raw videos
- cached features
- model weights

Only aggregate metrics and fold assignment files (video IDs only) are tracked.
