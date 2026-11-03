"""Read labels.json into aligned id / label / age arrays.

Label polarity is Abnormal = 1 (the legacy code had Normal = 1). Any interpretation other than the
configured normal value is treated as Abnormal, so free-text findings cannot slip through as Normal.
IDs are sorted by their video number so every downstream file has a deterministic order.
"""

import json
import re

import numpy as np


def _video_number(video_id):
    match = re.search(r"(\d+)", video_id)
    return int(match.group(1)) if match else -1


def load_labels(labels_path, normal_value="Normal"):
    """Return dict with ids (str), labels (int8, Abnormal = 1) and ages (int16), sorted by id."""
    with open(labels_path, encoding="utf-8") as fh:
        records = json.load(fh)
    records = sorted(records, key=lambda r: (_video_number(r["id"]), r["id"]))
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate ids in {labels_path}")
    labels = [0 if str(r["interpretation"]).strip() == normal_value else 1 for r in records]
    ages = [int(r.get("age", -1)) for r in records]
    return {
        "ids": np.array(ids),
        "labels": np.array(labels, dtype=np.int8),
        "ages": np.array(ages, dtype=np.int16),
    }
