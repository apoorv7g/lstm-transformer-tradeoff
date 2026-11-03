# tests

Automated checks that the code still does what it should. They use small made-up videos, so no
real medical data is needed.

| File | What it checks |
|---|---|
| `test_data_pipeline.py` | Frames come out in order, at 224 x 224, as RGB; short videos are padded; folds never share a video and keep the Abnormal share even |

Run them from the repo root:

```bash
python -m pytest tests -q
```

Still to add: model output shapes and a parameter-count check for every model.
