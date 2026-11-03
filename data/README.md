# data

The dataset, in three folders that are used in order:

```
raw/  (videos)  ->  features/  (one cached tensor)  ->  splits/  (which video goes in which fold)
```

| Folder | Holds | Committed to git? |
|---|---|---|
| `raw/` | The 535 oral screening videos and `labels.json` | No, private medical data |
| `features/` | `oral_T40.npz`: MobileNetV2 embeddings of 40 frames per video, plus `oral_T40.meta.json` (how each video was decoded) | No, derived from the videos |
| `splits/` | `oral_seed0.json`, `oral_seed1.json`, `oral_seed2.json`: fold assignments | Yes, video IDs only |

To rebuild from the videos (from the repo root):

```bash
python -m src.common extract    # about 4 minutes
python -m src.common splits     # a few seconds
```

Never commit videos, features or model weights. `.gitignore` already blocks them.
