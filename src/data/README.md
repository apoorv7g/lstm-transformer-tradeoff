# src/data

Turns the raw videos into the inputs every model uses. Design notes: `docs/data_pipeline.md`.

| File | What it does |
|---|---|
| `labels.py` | Reads `labels.json`; sets Abnormal = 1, Normal = 0 |
| `video_io.py` | `sample_frames(path, 40)`: takes 40 evenly spaced frames from a video, RGB, 224 x 224 |
| `extract_features.py` | Command: runs every video's frames through MobileNetV2 and saves `data/features/oral_T40.npz` |
| `splits.py` | Command: writes the 5-fold splits to `data/splits/` |
| `features.py` | What training code calls: `load_features(T)`, `load_split(seed)`, `fold_indices(ids, fold)` |

Commands (from the repo root):

```bash
python -m src.data.extract_features --dataset oral --frames 40
python -m src.data.extract_features --dataset oral --frames 40 --limit 8   # quick test on 8 videos
python -m src.data.splits --dataset oral
```

Things the sampler handles on purpose:
- Short videos (fewer than 40 frames) are padded by repeating frames, never dropped.
- The frame count in a video's header can be wrong, so the real count is used.
- A single broken frame does not stop decoding (one video, `video-487`, has one).
- If any video fails completely, nothing is written, so no video is silently lost.
