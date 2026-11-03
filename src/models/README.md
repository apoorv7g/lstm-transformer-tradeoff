# src/models

The models being compared. Every model takes the same input, `(T, 1280)` frame features, and
ends with the same classifier head. Only the middle part (LSTM or Transformer) changes.

| File | What it holds |
|---|---|
| `heads.py` | The shared input and classifier head: Dense(64) then Dropout(0.5) then Dense(1, sigmoid) |
| `lstm.py` | LSTM models: `lstm_small`, `lstm_base`, `bilstm_base` |
| `transformer.py` | Transformer models: `transformer_small`, `transformer_base`, `transformer_deep` |
| `hybrid.py` | LSTM followed by one Transformer block: `hybrid_base` (stretch goal) |
| `__init__.py` | Registry, so any model is built by name |
| `summary.py` | Command: builds every model and prints its parameter count |

Build a model:

```python
from src.models import build_model
model = build_model("transformer_base", num_frames=20)
```

Check parameter counts (should match the table in the main README):

```bash
python -m src.models.summary
```

To add a model, add an entry to the `PRESETS` dict in its file; the registry picks it up. Do not
change an existing preset once results exist. Design notes: `docs/lstm_design.md` and
`docs/transformer_design.md`.
