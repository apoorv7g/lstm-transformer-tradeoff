"""LSTM models over cached MobileNetV2 frame features. Design: docs/lstm_design.md.

    python -m src.lstm --preset lstm_base --T 20 --seed 0              # train all 5 folds
    python -m src.lstm --preset lstm_base --T 20 --seed 0 --folds 0    # one fold
    python -m src.lstm --preset lstm_base --T 20 --dry-run             # build + check, no training
    python -m src.lstm --preset lstm_base --T 20 --benchmark --threads 1

recurrent_dropout stays 0 and unroll stays False so Keras uses the fused LSTM kernel; anything else
falls back to a slow generic loop and would distort the CPU latency comparison.
"""

from src import common  # first: sets TF_CPP_MIN_LOG_LEVEL before TensorFlow loads

import keras
from keras import layers

FAMILY = "lstm"

# Optimiser settings for this family; everything else is common.TRAINING.
RECIPE = dict(optimizer="adam", learning_rate=1e-3)

# Frozen once results exist: add a new name instead of editing one.
PRESETS = {
    "lstm_small": dict(units=(128,)),                          # cheapest reasonable RNN (729,729 @T20)
    "lstm_base": dict(units=(256, 128)),                       # exact legacy head (1,779,329 @T20)
    "bilstm_base": dict(units=(128, 64), bidirectional=True),  # sees future frames (1,615,489 @T20)
}


def build_lstm(num_frames=20, units=(256, 128), bidirectional=False, dropout=0.0, name="lstm"):
    """Stacked (Bi)LSTM -> shared head. Default = legacy LSTM(256, seq) -> LSTM(128)."""
    inputs = common.feature_input(num_frames)
    x = inputs
    for i, u in enumerate(units):
        rnn = layers.LSTM(u, return_sequences=i < len(units) - 1, dropout=dropout,
                          recurrent_dropout=0.0, name=f"{name}_lstm{i + 1}")
        if bidirectional:
            rnn = layers.Bidirectional(rnn, name=f"{name}_bilstm{i + 1}")
        x = rnn(x)
    return keras.Model(inputs, common.classifier_head(x), name=name)


def build(preset, num_frames=20):
    return build_lstm(num_frames, name=preset, **PRESETS[preset])


def macs(preset, num_frames=20):
    """Analytic multiply-accumulates per clip (the TF profiler under-counts inside the LSTM loop).

    One LSTM step does 4 gate matmuls: 4 * h * (input + h). A BiLSTM runs two of them and passes
    2h features on.
    """
    cfg = PRESETS[preset]
    directions = 2 if cfg.get("bidirectional") else 1
    total, in_dim = 0, common.FEATURE_DIM
    for h in cfg["units"]:
        total += directions * num_frames * 4 * h * (in_dim + h)
        in_dim = directions * h
    return total + common.head_macs(in_dim)


if __name__ == "__main__":
    common.model_cli(FAMILY, PRESETS, build, RECIPE, macs)
