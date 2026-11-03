"""Transformer temporal heads. Design rationale: docs/transformer_design.md.

Structure follows the Keras "Video Classification with Transformers" example
(PositionalEmbedding + TransformerEncoder over per-frame CNN features), with
two changes: a linear projection 1280 -> d_model so the model size can be
controlled independently of the backbone, and pre-LayerNorm blocks, which train
more stably on small datasets.
"""

import keras
from keras import layers, ops

from .heads import FEATURE_DIM, classifier_head, feature_input


@keras.saving.register_keras_serializable(package="lstm_tf")
class PositionalEmbedding(layers.Layer):
    """Learned absolute position embedding added to the frame tokens."""

    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.position_embeddings = layers.Embedding(sequence_length, d_model)

    def call(self, x):
        positions = ops.arange(0, self.sequence_length, 1)
        return x + self.position_embeddings(positions)

    def get_config(self):
        config = super().get_config()
        config.update(sequence_length=self.sequence_length, d_model=self.d_model)
        return config


@keras.saving.register_keras_serializable(package="lstm_tf")
class TransformerEncoderBlock(layers.Layer):
    """Pre-LN encoder block: x + MHA(LN(x)); x + FFN(LN(x))."""

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
        )
        self.drop1 = layers.Dropout(dropout)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ff1 = layers.Dense(ff_dim, activation="gelu")
        self.ff2 = layers.Dense(d_model)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=None):
        h = self.ln1(x)
        h = self.attn(h, h, training=training)
        x = x + self.drop1(h, training=training)
        h = self.ln2(x)
        h = self.ff2(self.ff1(h))
        return x + self.drop2(h, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            d_model=self.d_model,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
        )
        return config


def build_transformer(
    num_frames=20,
    feature_dim=FEATURE_DIM,
    d_model=128,
    num_heads=4,
    ff_dim=256,
    num_layers=2,
    dropout=0.1,
    pooling="mean",
    head_units=64,
    head_dropout=0.5,
    name="transformer",
):
    """Transformer encoder over frame embeddings.

    pooling: "mean" (global average over frames) or "max" (as in the Keras
    example). Mean pooling is the default: on a few hundred clips a CLS token is
    one more thing to learn from scratch and rarely helps.
    """
    if d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads")

    inputs = feature_input(num_frames, feature_dim)
    x = layers.Dense(d_model, name=f"{name}_proj")(inputs)
    x = PositionalEmbedding(num_frames, d_model, name=f"{name}_pos")(x)
    x = layers.Dropout(dropout, name=f"{name}_emb_dropout")(x)
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            d_model, num_heads, ff_dim, dropout, name=f"{name}_block{i + 1}"
        )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_final_ln")(x)
    if pooling == "mean":
        x = layers.GlobalAveragePooling1D(name=f"{name}_pool")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling1D(name=f"{name}_pool")(x)
    else:
        raise ValueError(f"unknown pooling: {pooling}")
    outputs = classifier_head(x, head_units, head_dropout)
    return keras.Model(inputs, outputs, name=name)


PRESETS = {
    # Small: cheapest reasonable attention model (~0.44M params).
    "transformer_small": dict(d_model=128, num_heads=4, ff_dim=256, num_layers=2),
    # Base: roughly parameter-matched to lstm_base (~1.4M vs ~1.78M).
    "transformer_base": dict(d_model=256, num_heads=4, ff_dim=512, num_layers=2),
    # Deep: same width as base, more layers, to see if depth buys accuracy.
    "transformer_deep": dict(d_model=256, num_heads=8, ff_dim=512, num_layers=4),
}
