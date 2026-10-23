"""Stretch goal: LSTM -> Transformer hybrid. See docs/design/transformer_design.md §Hybrid.

The LSTM injects local order/recency information (so no positional embedding
is needed), and a single attention block then lets every time step look at
every other one before pooling.
"""

import keras
from keras import layers

from .heads import FEATURE_DIM, classifier_head, feature_input
from .transformer import TransformerEncoderBlock


def build_hybrid(
    num_frames=20,
    feature_dim=FEATURE_DIM,
    lstm_units=128,
    num_heads=4,
    ff_dim=256,
    num_layers=1,
    dropout=0.1,
    head_units=64,
    head_dropout=0.5,
    name="hybrid",
):
    inputs = feature_input(num_frames, feature_dim)
    x = layers.LSTM(lstm_units, return_sequences=True, name=f"{name}_lstm")(inputs)
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            lstm_units, num_heads, ff_dim, dropout, name=f"{name}_block{i + 1}"
        )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_final_ln")(x)
    x = layers.GlobalAveragePooling1D(name=f"{name}_pool")(x)
    outputs = classifier_head(x, head_units, head_dropout)
    return keras.Model(inputs, outputs, name=name)


PRESETS = {
    "hybrid_base": dict(lstm_units=128, num_heads=4, ff_dim=256, num_layers=1),
}
