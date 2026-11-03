"""No-temporal baselines: pool the frame features over time, then the shared head.

    python -m src.baseline --preset pool_mean --T 20 --seed 0
    python -m src.baseline --preset pool_max --T 20 --dry-run

These models ignore frame order completely. They answer the question the LSTM and Transformer
results cannot answer alone: does temporal modelling help at all on this task? If a pooled
baseline matches them within the equivalence margin, the cheapest model wins on compute.
    pool_mean: average of the frame embeddings (the clip's overall appearance)
    pool_max:  per-feature maximum over frames (keeps a finding that is visible in only a few frames)
Uses the LSTM recipe (Adam 1e-3, no warm-up): there is no attention to stabilise.
"""

from src import common  # first: sets TF_CPP_MIN_LOG_LEVEL before TensorFlow loads

import keras
from keras import layers

FAMILY = "baseline"

RECIPE = dict(optimizer="adam", learning_rate=1e-3)

# Frozen once results exist: add a new name instead of editing one.
PRESETS = {
    "pool_mean": dict(pooling="mean"),   # 82,049 params at any T
    "pool_max": dict(pooling="max"),     # 82,049 params at any T
}


def build_pool(num_frames=20, pooling="mean", name="pool"):
    inputs = common.feature_input(num_frames)
    pool = layers.GlobalAveragePooling1D if pooling == "mean" else layers.GlobalMaxPooling1D
    x = pool(name=f"{name}_{pooling}")(inputs)
    return keras.Model(inputs, common.classifier_head(x), name=name)


def build(preset, num_frames=20):
    return build_pool(num_frames, name=preset, **PRESETS[preset])


def macs(preset, num_frames=20):
    """Pooling is additions or comparisons only (T x 1280, no multiplies); the head dominates."""
    return common.head_macs(common.FEATURE_DIM)


if __name__ == "__main__":
    common.model_cli(FAMILY, PRESETS, build, RECIPE, macs)
