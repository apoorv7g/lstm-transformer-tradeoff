# data/raw

The original videos. **Private pediatric medical data: never commit anything here except this
README.**

Expected contents:

```
data/raw/labels.json
data/raw/video-1.mp4 ... video-535.mp4
```

- 535 oral-cavity screening videos: 452 Normal, 83 Abnormal.
- Portrait 480 x 640, 1 to 64 seconds long (median 26 s), 3.2 GB in total.
- `labels.json` is a list of `{"id": "video-N.mp4", "interpretation": "Normal" or "Abnormal", "age": N}`.
  In the code, Abnormal = 1 and Normal = 0.
- There is no patient ID, so two clips of the same child could land in different folds.

A fresh clone does not include the videos. Copy them in from the team's private storage, then
run the feature extraction (see `data/README.md`).
