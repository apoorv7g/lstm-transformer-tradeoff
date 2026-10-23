"""LSTM temporal heads. Design rationale: docs/design/lstm_design.md."""

import keras
from keras import layers

from .heads import FEATURE_DIM, classifier_head, feature_input


def build_lstm(
    num_frames=20,
    feature_dim=FEATURE_DIM,
    units=(256, 128),
    dropout=0.0,
    recurrent_dropout=0.0,
    bidirectional=False,
    head_units=64,
    head_dropout=0.5,
    name="lstm",
):
    """Stacked (Bi)LSTM over frame embeddings.

    Defaults reproduce the legacy DL_test1 temporal head exactly:
    LSTM(256, return_sequences=True) -> LSTM(128) -> Dense(64) -> Dropout(0.5) -> Dense(1).

    recurrent_dropout must stay 0.0 if we want the fused (cuDNN/oneDNN-friendly)
    LSTM kernel; a non-zero value silently falls back to the slow generic loop
    and would distort the CPU latency benchmark.
    """
    inputs = feature_input(num_frames, feature_dim)
    x = inputs
    for i, u in enumerate(units):
        last = i == len(units) - 1
        rnn = layers.LSTM(
            u,
            return_sequences=not last,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name=f"{name}_lstm{i + 1}",
        )
        if bidirectional:
            rnn = layers.Bidirectional(rnn, name=f"{name}_bilstm{i + 1}")
        x = rnn(x)
    outputs = classifier_head(x, head_units, head_dropout)
    return keras.Model(inputs, outputs, name=name)


# Named presets referenced by configs/ and the design doc.
PRESETS = {
    # Legacy-faithful baseline (~1.78M params).
    "lstm_base": dict(units=(256, 128)),
    # Single small layer: the "cheapest reasonable RNN" point on the curve.
    "lstm_small": dict(units=(128,)),
    # BiLSTM: tests whether seeing the future frames helps (2x compute).
    "bilstm_base": dict(units=(128, 64), bidirectional=True),
}
