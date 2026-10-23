"""Shared building blocks used by every temporal head.

Every model in this project consumes the SAME input: a sequence of pre-extracted,
frozen MobileNetV2 frame embeddings of shape (T, D) = (NUM_FRAMES, 1280).
Only the temporal block in the middle changes between models; the classifier
head below is identical everywhere so accuracy differences can be attributed to
the temporal block alone.
"""

import keras
from keras import layers

FEATURE_DIM = 1280  # MobileNetV2 GlobalAveragePooling2D output


def classifier_head(x, hidden_units=64, dropout=0.5, name="head"):
    """Dense(64, relu) -> Dropout(0.5) -> Dense(1, sigmoid).

    Same head as the legacy DL_test1 models, kept identical across every
    architecture in this study.
    """
    x = layers.Dense(hidden_units, activation="relu", name=f"{name}_dense")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return layers.Dense(1, activation="sigmoid", name=f"{name}_out")(x)


def feature_input(num_frames, feature_dim=FEATURE_DIM):
    return keras.Input(shape=(num_frames, feature_dim), name="frame_features")
