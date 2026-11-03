"""Data pipeline. Spec: docs/data_pipeline.md.

- video_io.py: sample_frames(path, T) -> uint8 (T, 224, 224, 3) RGB + decode log
- labels.py: labels.json -> ids / labels (Abnormal = 1) / ages
- extract_features.py: CLI that caches frozen MobileNetV2 embeddings to data/features/
- splits.py: CLI that writes the shared stratified folds to data/splits/
- features.py: load_features(T), load_split(seed), fold_indices(): what training code calls
"""
