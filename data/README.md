# data

The dataset, in three folders that are used in order:

```
raw/  (videos)  ->  features/  (one cached tensor)  ->  splits/  (which video goes in which fold)
```

| Folder | Holds | Committed to git? |
|---|---|---|
| `raw/` | The 874 ear (otoscopy) screening videos and `labels.json` | No, private medical data |
| `features/` | `ear_T40.npz`: MobileNetV2 embeddings of 40 frames per video (about 179 MB), plus `ear_T40.meta.json` (how each video was decoded) | No, derived from the videos |
| `splits/` | `ear_seed0.json`, `ear_seed1.json`, `ear_seed2.json`: fold assignments | Yes, video IDs only |

To rebuild from the videos (from the repo root):

```bash
python -m src.common extract    # about 6 minutes; --workers sets decode threads
python -m src.common splits     # a few seconds
```

Never commit videos, features or model weights. `.gitignore` already blocks them.
