"""LSTM -> Transformer hybrid (stretch goal). Design: docs/transformer_design.md, hybrid section.

    python -m src.hybrid --preset hybrid_base --T 20 --seed 0
    python -m src.hybrid --preset hybrid_base --T 20 --dry-run

The LSTM reads the frames in order, so its outputs already carry position and recency (no
positional embedding needed). Attention blocks then let every time step look at every other one
before mean pooling. Uses the Transformer recipe, because it contains attention.

hybrid_attn / hybrid_lite (added 3 Nov) are built for a finding that is mostly a matter of how the
canal looks, and may be visible in only some frames as the camera moves:
    Dense(1280 -> d) projection    shrinks each frame first, so the LSTM is cheap (4 d * 2d per step
                                   instead of 4 h * (1280 + h))
    LSTM(d, sequences)             local context and order; no positional embedding needed
    1 pre-LN attention block       every frame can compare itself with every other frame
    attention pooling + max pool   learned per-frame weights pick the informative frames; the max
                                   keeps strong evidence from a single frame; mean pooling would
                                   dilute it across the frames where the finding is not visible
    shared head
"""

from src import common  # first: sets TF_CPP_MIN_LOG_LEVEL before TensorFlow loads

import keras
from keras import layers, ops
from src.transformer import RECIPE, TransformerEncoderBlock, block_macs

FAMILY = "hybrid"

PRESETS = {
    "hybrid_base": dict(lstm_units=128, num_heads=4, ff_dim=256, num_layers=1),  # 862,465 @T20
    "hybrid_attn": dict(design="attnpool", d_model=128, num_heads=4, ff_dim=256, num_layers=1),
    "hybrid_lite": dict(design="attnpool", d_model=64, num_heads=2, ff_dim=128, num_layers=1),
}


@keras.saving.register_keras_serializable(package="lstm_tf")
class AttentionPooling(layers.Layer):
    """Weighted average over time with learned weights: softmax_t(w . x_t + b) * x_t, summed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = layers.Dense(1)

    def build(self, input_shape):
        self.score.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        weights = ops.softmax(self.score(x), axis=1)
        return ops.sum(weights * x, axis=1)


def build_hybrid(num_frames=20, lstm_units=128, num_heads=4, ff_dim=256, num_layers=1, dropout=0.1,
                 name="hybrid"):
    inputs = common.feature_input(num_frames)
    x = layers.LSTM(lstm_units, return_sequences=True, name=f"{name}_lstm")(inputs)
    for i in range(num_layers):
        x = TransformerEncoderBlock(lstm_units, num_heads, ff_dim, dropout, name=f"{name}_block{i + 1}")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_final_ln")(x)
    x = layers.GlobalAveragePooling1D(name=f"{name}_pool")(x)
    return keras.Model(inputs, common.classifier_head(x), name=name)


def build_hybrid_attnpool(num_frames=20, d_model=128, num_heads=4, ff_dim=256, num_layers=1,
                          dropout=0.1, name="hybrid_attnpool"):
    inputs = common.feature_input(num_frames)
    x = layers.Dense(d_model, name=f"{name}_proj")(inputs)
    x = layers.LSTM(d_model, return_sequences=True, name=f"{name}_lstm")(x)
    for i in range(num_layers):
        x = TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout, name=f"{name}_block{i + 1}")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_final_ln")(x)
    x = layers.Concatenate(name=f"{name}_pool")([AttentionPooling(name=f"{name}_attnpool")(x),
                                                  layers.GlobalMaxPooling1D(name=f"{name}_maxpool")(x)])
    return keras.Model(inputs, common.classifier_head(x), name=name)


def build(preset, num_frames=20):
    cfg = dict(PRESETS[preset])
    if cfg.pop("design", None) == "attnpool":
        return build_hybrid_attnpool(num_frames, name=preset, **cfg)
    return build_hybrid(num_frames, name=preset, **cfg)


def macs(preset, num_frames=20):
    cfg = PRESETS[preset]
    if cfg.get("design") == "attnpool":
        t, d = num_frames, cfg["d_model"]
        return (t * common.FEATURE_DIM * d                       # projection
                + t * 4 * d * (d + d)                            # LSTM on projected frames
                + cfg["num_layers"] * block_macs(t, d, cfg["ff_dim"])
                + t * d                                          # attention-pooling scores
                + common.head_macs(2 * d))                       # head on [attn pool, max pool]
    h = cfg["lstm_units"]
    lstm = num_frames * 4 * h * (common.FEATURE_DIM + h)
    return lstm + cfg["num_layers"] * block_macs(num_frames, h, cfg["ff_dim"]) + common.head_macs(h)


if __name__ == "__main__":
    common.model_cli(FAMILY, PRESETS, build, RECIPE, macs)
