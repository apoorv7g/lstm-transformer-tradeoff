"""Transformer encoder over cached MobileNetV2 frame features. Design: docs/transformer_design.md.

    python -m src.transformer --preset transformer_base --T 20 --seed 0
    python -m src.transformer --preset transformer_base --T 20 --dry-run
    python -m src.transformer --preset transformer_base --T 20 --benchmark --threads 1

Each frame embedding is one token: Dense(1280 -> d) projection, learned positional embedding,
pre-LayerNorm encoder blocks (GELU FFN of width 2d, dropout 0.1), final LayerNorm, mean pooling,
then the shared head. No CLS token and no mask: clips are repeat-padded, never zero-padded.
This is a temporal Transformer over frames, not a ViT over pixels (the CNN backbone is frozen).
"""

from src import common  # first: sets TF_CPP_MIN_LOG_LEVEL before TensorFlow loads

import keras
from keras import layers, ops

FAMILY = "transformer"

# Standard attention recipe: AdamW plus a 5-epoch linear warm-up. Rest is common.TRAINING.
RECIPE = dict(optimizer="adamw", learning_rate=3e-4, weight_decay=1e-2, warmup_epochs=5)

# Frozen once results exist: add a new name instead of editing one.
PRESETS = {
    "transformer_small": dict(d_model=128, num_heads=4, ff_dim=256, num_layers=2),  # 440,065 @T20
    "transformer_base": dict(d_model=256, num_heads=4, ff_dim=512, num_layers=2),   # 1,404,289 @T20, ~lstm_base
    "transformer_deep": dict(d_model=256, num_heads=8, ff_dim=512, num_layers=4),   # 2,458,497 @T20
}


@keras.saving.register_keras_serializable(package="lstm_tf")
class PositionalEmbedding(layers.Layer):
    """Learned absolute position embedding added to the frame tokens."""

    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length, self.d_model = sequence_length, d_model
        self.position_embeddings = layers.Embedding(sequence_length, d_model)

    def build(self, input_shape):
        # Explicit build so a reloaded .keras file restores weights eagerly.
        self.position_embeddings.build((self.sequence_length,))
        super().build(input_shape)

    def call(self, x):
        return x + self.position_embeddings(ops.arange(0, self.sequence_length, 1))

    def get_config(self):
        return {**super().get_config(), "sequence_length": self.sequence_length, "d_model": self.d_model}


@keras.saving.register_keras_serializable(package="lstm_tf")
class TransformerEncoderBlock(layers.Layer):
    """Pre-LN block: x + MHA(LN(x)); x + FFN(LN(x)). Pre-LN trains more stably on small data."""

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads, self.ff_dim, self.dropout = d_model, num_heads, ff_dim, dropout
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                              dropout=dropout)
        self.drop1 = layers.Dropout(dropout)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ff1 = layers.Dense(ff_dim, activation="gelu")
        self.ff2 = layers.Dense(d_model)
        self.drop2 = layers.Dropout(dropout)

    def build(self, input_shape):
        self.ln1.build(input_shape)
        self.attn.build(input_shape, input_shape)
        self.ln2.build(input_shape)
        self.ff1.build(input_shape)
        self.ff2.build((*input_shape[:-1], self.ff_dim))
        super().build(input_shape)

    def call(self, x, training=None):
        h = self.ln1(x)
        x = x + self.drop1(self.attn(h, h, training=training), training=training)
        return x + self.drop2(self.ff2(self.ff1(self.ln2(x))), training=training)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "num_heads": self.num_heads,
                "ff_dim": self.ff_dim, "dropout": self.dropout}


def build_transformer(num_frames=20, d_model=128, num_heads=4, ff_dim=256, num_layers=2,
                      dropout=0.1, pooling="mean", name="transformer"):
    if d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    inputs = common.feature_input(num_frames)
    x = layers.Dense(d_model, name=f"{name}_proj")(inputs)
    x = PositionalEmbedding(num_frames, d_model, name=f"{name}_pos")(x)
    x = layers.Dropout(dropout, name=f"{name}_emb_dropout")(x)
    for i in range(num_layers):
        x = TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout, name=f"{name}_block{i + 1}")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_final_ln")(x)
    pool = {"mean": layers.GlobalAveragePooling1D, "max": layers.GlobalMaxPooling1D}[pooling]
    x = pool(name=f"{name}_pool")(x)
    return keras.Model(inputs, common.classifier_head(x), name=name)


def build(preset, num_frames=20):
    return build_transformer(num_frames, name=preset, **PRESETS[preset])


def block_macs(num_frames, d_model, ff_dim):
    """One encoder block: Q, K, V and output projections (4 T d^2), scores and weighted sum
    (2 T^2 d), and the FFN (2 T d ff). The T^2 term is why attention cost grows faster with T."""
    t, d = num_frames, d_model
    return 4 * t * d * d + 2 * t * t * d + 2 * t * d * ff_dim


def macs(preset, num_frames=20):
    cfg = PRESETS[preset]
    d = cfg["d_model"]
    return (num_frames * common.FEATURE_DIM * d
            + cfg["num_layers"] * block_macs(num_frames, d, cfg["ff_dim"])
            + common.head_macs(d))


if __name__ == "__main__":
    common.model_cli(FAMILY, PRESETS, build, RECIPE, macs)
