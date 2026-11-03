"""LSTM -> Transformer hybrid (stretch goal). Design: docs/transformer_design.md, hybrid section.

    python -m src.hybrid --preset hybrid_base --T 20 --seed 0
    python -m src.hybrid --preset hybrid_base --T 20 --dry-run

The LSTM reads the frames in order, so its outputs already carry position and recency (no
positional embedding needed). Attention blocks then let every time step look at every other one
before mean pooling. Uses the Transformer recipe, because it contains attention.
More hybrid designs can be added here as new PRESETS entries or new build functions.
"""

from src import common  # first: sets TF_CPP_MIN_LOG_LEVEL before TensorFlow loads

import keras
from keras import layers
from src.transformer import RECIPE, TransformerEncoderBlock, block_macs

FAMILY = "hybrid"

PRESETS = {
    "hybrid_base": dict(lstm_units=128, num_heads=4, ff_dim=256, num_layers=1),  # 862,465 @T20
}


def build_hybrid(num_frames=20, lstm_units=128, num_heads=4, ff_dim=256, num_layers=1, dropout=0.1,
                 name="hybrid"):
    inputs = common.feature_input(num_frames)
    x = layers.LSTM(lstm_units, return_sequences=True, name=f"{name}_lstm")(inputs)
    for i in range(num_layers):
        x = TransformerEncoderBlock(lstm_units, num_heads, ff_dim, dropout, name=f"{name}_block{i + 1}")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_final_ln")(x)
    x = layers.GlobalAveragePooling1D(name=f"{name}_pool")(x)
    return keras.Model(inputs, common.classifier_head(x), name=name)


def build(preset, num_frames=20):
    return build_hybrid(num_frames, name=preset, **PRESETS[preset])


def macs(preset, num_frames=20):
    cfg = PRESETS[preset]
    h = cfg["lstm_units"]
    lstm = num_frames * 4 * h * (common.FEATURE_DIM + h)
    return lstm + cfg["num_layers"] * block_macs(num_frames, h, cfg["ff_dim"]) + common.head_macs(h)


if __name__ == "__main__":
    common.model_cli(FAMILY, PRESETS, build, RECIPE, macs)
