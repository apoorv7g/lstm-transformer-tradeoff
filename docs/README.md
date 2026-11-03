# docs

All written documentation, in one flat folder. Read in this order:

**Start here**

| File | What it covers |
|---|---|
| `01_summary.md` | One-page overview of the project |
| `02_abstract.md` | Structured abstract, completed with the results |
| `03_scope.md` | Research questions, hypotheses, what is in and out of scope, risks |
| `04_why_it_matters.md` | Why the comparison is useful |
| `05_results.md` | **Results and conclusions**: what was run, the numbers, hypothesis verdicts |

**How it is built**

| File | What it covers |
|---|---|
| `data_pipeline.md` | Videos to frames to cached features to folds |
| `lstm_design.md` | LSTM models: equations, sizes, parameter counts, training recipe |
| `transformer_design.md` | Transformer models and the LSTM + Transformer hybrid |
| `benchmarking_protocol.md` | How every accuracy and speed number is measured, and the statistics |

**Background**

| File | What it covers |
|---|---|
| `literature_review.md` | Related papers, each marked verified or unverified |
| `legacy_pipeline_audit.md` | Bugs found in the earlier LSTM project and how this repo fixes them |

**Slides and report**

- `slides/` holds the 23 Oct progress review deck (PDF) and the files used to build it.
- `report/` holds the A4 project report
  (`Compute_vs_Accuracy_LSTM_Transformer_Report.pdf`), its source `report.html` (same theme as the
  deck) and `build.py` (headless Playwright; run with `py -3.11 docs/report/build.py`).
