"""Tests for the model presets and the training harness (no real medical data needed).

Run from the repo root:  python -m pytest tests -q

The parameter counts are the frozen table in docs/lstm_design.md and docs/transformer_design.md
(Keras 3.6). If one changes, a preset definition changed, which is not allowed once results exist.
The training-loop test fits two epochs on random numbers only to prove the plumbing works.
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import pytest

from src.models import PRESET_NAMES, build_model
from src.training.metrics import evaluate, youden_threshold
from src.training.train import balanced_class_weight, recipe_for, run_fold

PARAMS_T20 = {
    "lstm_small": 729_729,
    "lstm_base": 1_779_329,
    "bilstm_base": 1_615_489,
    "transformer_small": 440_065,
    "transformer_base": 1_404_289,
    "transformer_deep": 2_458_497,
    "hybrid_base": 862_465,
}


def test_every_preset_is_covered():
    assert set(PRESET_NAMES) == set(PARAMS_T20)


@pytest.mark.parametrize("preset", sorted(PARAMS_T20))
def test_param_count_and_output_shape(preset):
    model = build_model(preset, num_frames=20)
    assert model.count_params() == PARAMS_T20[preset]
    out = model.predict(np.zeros((3, 20, 1280), "float32"), verbose=0)
    assert out.shape == (3, 1) and np.all((out >= 0) & (out <= 1))


@pytest.mark.parametrize("preset", ["transformer_small", "hybrid_base"])
def test_custom_layers_round_trip(tmp_path, preset):
    model = build_model(preset, num_frames=10)
    x = np.random.default_rng(0).normal(size=(2, 10, 1280)).astype("float32")
    path = tmp_path / "m.keras"
    model.save(path)
    again = keras.models.load_model(path)
    np.testing.assert_allclose(model.predict(x, verbose=0), again.predict(x, verbose=0), atol=1e-5)


def test_recipe_family():
    assert recipe_for("lstm_base")[1]["optimizer"] == "adam"
    assert recipe_for("transformer_base")[1]["optimizer"] == "adamw"
    assert recipe_for("hybrid_base")[1]["optimizer"] == "adamw"


def test_balanced_class_weight_matches_dataset():
    w = balanced_class_weight([0] * 452 + [1] * 83)
    assert w[0] == pytest.approx(0.59, abs=0.01) and w[1] == pytest.approx(3.22, abs=0.01)


def test_metrics_perfect_and_youden():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    m = evaluate(y, p, y, p)
    assert m["auroc"] == 1.0 and m["at_0.5"]["sensitivity"] == 1.0 and m["at_0.5"]["specificity"] == 1.0
    assert 0.3 < youden_threshold(y, p) <= 0.8


@pytest.mark.parametrize("preset", ["lstm_small", "transformer_small"])
def test_run_fold_writes_run_folder(tmp_path, preset):
    rng = np.random.default_rng(0)
    n, t = 40, 10
    labels = np.array([0] * 32 + [1] * 8, dtype="int8")
    data = {
        "features": rng.normal(size=(n, t, 1280)).astype("float32"),
        "labels": labels,
        "ids": np.array([f"video-{i}.mp4" for i in range(n)]),
    }
    order = rng.permutation(n)
    idx = {"train": order[:24], "val": order[24:32], "test": order[32:]}
    data["labels"][idx["val"][:2]] = 1
    data["labels"][idx["test"][:2]] = 1
    _, recipe = recipe_for(preset)
    out = tmp_path / "run"
    metrics = run_fold(preset, t, 0, 0, data, idx, recipe, out, max_epochs=2, verbose=0)
    for name in ("metrics.json", "history.csv", "oof_predictions.csv", "model.keras"):
        assert (out / name).exists()
    assert metrics["epochs_run"] == 2 and metrics["n"] == 8
    assert json.loads((out / "metrics.json").read_text())["preset"] == preset
