# data/features

The cached input that every model trains on. Not committed (it is derived from private videos).

| File | What it is |
|---|---|
| `oral_T40.npz` | `features` (535, 40, 1280) float32, `labels` (535,) with Abnormal = 1, `ids`, `ages`. About 110 MB |
| `oral_T40.meta.json` | How it was made: backbone, preprocessing, and a log per video (frames decoded, padding) |

How it is made: for each video, 40 evenly spaced frames are taken, converted to RGB, resized to
224 x 224 and passed through a frozen ImageNet MobileNetV2. Each frame becomes a 1280-number
vector. Because the backbone never changes, doing this once is the same as running it inside every
model, and about 100 times cheaper.

Build it:

```bash
python -m src.data.extract_features --dataset oral --frames 40
```

Use it in code (T = 10, 20 or 40; shorter lengths keep every 2nd or 4th frame):

```python
from src.data.features import load_features
data = load_features(20)        # data["features"].shape == (535, 20, 1280)
```
