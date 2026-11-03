"""All automated checks. No real medical data is needed (synthetic videos and random features).

    python -m pytest src/tests.py -q

The parameter counts are the frozen table in docs/lstm_design.md and docs/transformer_design.md
(Keras 3.6). If one changes, a preset definition changed, which is not allowed once results exist.
The training test fits two epochs on random numbers only to prove the logging plumbing works.
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import keras
import numpy as np
import pytest

from src import baseline, common, hybrid, lstm, transformer

MODULES = {"baseline": baseline, "lstm": lstm, "transformer": transformer, "hybrid": hybrid}
PARAMS_T20 = {
    "lstm_small": 729_729,
    "lstm_base": 1_779_329,
    "bilstm_base": 1_615_489,
    "transformer_small": 440_065,
    "transformer_base": 1_404_289,
    "transformer_deep": 2_458_497,
    "hybrid_base": 862_465,
    "hybrid_attn": 444_930,
    "hybrid_lite": 156_994,
    "pool_mean": 82_049,
    "pool_max": 82_049,
}
ALL = [(fam, p) for fam, mod in MODULES.items() for p in mod.PRESETS]


# ---------------------------------------------------------------- data

def _write_video(path, n_frames, size=(64, 48)):
    """Synthetic clip whose frame i is a flat grey of brightness 5 * i, so order is checkable."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), 5 * i, dtype=np.uint8))
    writer.release()


def test_target_indices_uniform_and_padded():
    assert common.target_indices(100, 5).tolist() == [0, 25, 50, 74, 99]
    short = common.target_indices(10, 40)
    assert len(short) == 40 and short.min() == 0 and short.max() == 9 and np.all(np.diff(short) >= 0)


@pytest.mark.parametrize("n_frames,padded", [(50, False), (12, True)])
def test_sample_frames_shape_dtype_order(tmp_path, n_frames, padded):
    path = tmp_path / "clip.mp4"
    _write_video(path, n_frames)
    frames, meta = common.sample_frames(path, num_frames=40)
    assert frames.shape == (40, 224, 224, 3) and frames.dtype == np.uint8
    assert meta["decoded_frames"] == n_frames and meta["padded"] is padded
    brightness = frames.mean(axis=(1, 2, 3))
    assert np.all(np.diff(brightness) > -3) and brightness[-1] > brightness[0] + 20


def test_sample_frames_bgr_to_rgb(tmp_path):
    path = tmp_path / "red.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(5):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[..., 2] = 255  # pure red in OpenCV's BGR order
        writer.write(frame)
    writer.release()
    frames, _ = common.sample_frames(path, num_frames=4)
    assert frames[..., 0].mean() > 200 and frames[..., 1].mean() < 50 and frames[..., 2].mean() < 50


def test_sample_frames_missing_file(tmp_path):
    with pytest.raises(IOError):
        common.sample_frames(tmp_path / "nope.mp4")


def test_split_integrity():
    rng = np.random.default_rng(0)
    ids = np.array([f"video-{i}.mp4" for i in range(874)])
    labels = (rng.random(874) < 231 / 874).astype(np.int8)
    split = common.make_split(ids, labels, seed=0)
    assert len(split["folds"]) == common.N_FOLDS == 3
    label_of = dict(zip(ids.tolist(), labels.tolist()))
    all_test = []
    for fold in split["folds"]:
        train, val, test = (set(fold[p]) for p in ("train", "val", "test"))
        assert not (train & val) and not (train & test) and not (val & test)
        assert len(train | val | test) == len(ids)
        for part in (test, val):
            assert abs(np.mean([label_of[v] for v in part]) - labels.mean()) < 0.05
        all_test.extend(test)
    assert sorted(all_test) == sorted(ids.tolist())          # every id tested exactly once
    assert common.make_split(ids, labels, seed=0) == split   # deterministic
    assert common.make_split(ids, labels, seed=1) != split


def test_committed_splits_match_labels_if_present():
    """The committed split files must be exactly what make_split produces from labels.json."""
    if not (common.RAW_DIR / "labels.json").exists():
        pytest.skip("labels.json not available")
    table = common.load_labels()
    for seed in common.SEEDS:
        saved = common.load_split(seed)
        fresh = common.make_split(table["ids"], table["labels"], seed)
        assert saved["folds"] == fresh["folds"]


# ---------------------------------------------------------------- models

def test_every_preset_is_covered():
    assert {p for _, p in ALL} == set(PARAMS_T20)


@pytest.mark.parametrize("family,preset", ALL)
def test_param_count_shape_and_macs(family, preset):
    mod = MODULES[family]
    model = mod.build(preset, 20)
    assert model.count_params() == PARAMS_T20[preset]
    out = model.predict(np.zeros((3, 20, 1280), "float32"), verbose=0)
    assert out.shape == (3, 1) and np.all((out >= 0) & (out <= 1))
    if family == "baseline":   # pooling has no multiplies, so cost does not grow with T
        assert mod.macs(preset, 40) == mod.macs(preset, 10) == 1280 * 64 + 64
    else:
        assert mod.macs(preset, 40) > mod.macs(preset, 20) > mod.macs(preset, 10) > 0


def test_macs_match_hand_count():
    # lstm_small at T=10: 10 * 4 * 128 * (1280 + 128) + head (128 * 64 + 64)
    assert lstm.macs("lstm_small", 10) == 10 * 4 * 128 * 1408 + 128 * 64 + 64


@pytest.mark.parametrize("family,preset", [("transformer", "transformer_small"), ("hybrid", "hybrid_base"),
                                           ("hybrid", "hybrid_attn"), ("baseline", "pool_max")])
def test_custom_layers_round_trip(tmp_path, family, preset):
    model = MODULES[family].build(preset, 10)
    x = np.random.default_rng(0).normal(size=(2, 10, 1280)).astype("float32")
    model.save(tmp_path / "m.keras")
    again = keras.models.load_model(tmp_path / "m.keras")
    np.testing.assert_allclose(model.predict(x, verbose=0), again.predict(x, verbose=0), atol=1e-5)


def test_recipes():
    assert common.full_recipe(lstm.RECIPE)["optimizer"] == "adam"
    assert common.full_recipe(lstm.RECIPE)["warmup_epochs"] == 0
    assert common.full_recipe(transformer.RECIPE)["optimizer"] == "adamw"
    assert common.full_recipe(transformer.RECIPE)["warmup_epochs"] == 5
    assert hybrid.RECIPE is transformer.RECIPE
    assert baseline.RECIPE == lstm.RECIPE


def test_pool_baselines_ignore_frame_order():
    x = np.random.default_rng(0).normal(size=(2, 10, 1280)).astype("float32")
    for preset in baseline.PRESETS:
        model = baseline.build(preset, 10)
        np.testing.assert_allclose(model.predict(x, verbose=0), model.predict(x[:, ::-1], verbose=0),
                                   atol=1e-5)


# ---------------------------------------------------------------- metrics and logging

def test_balanced_class_weight_matches_dataset():
    w = common.balanced_class_weight([0] * 643 + [1] * 231)
    assert w[0] == pytest.approx(0.68, abs=0.01) and w[1] == pytest.approx(1.89, abs=0.01)


def test_metrics_perfect_and_youden():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    m = common.evaluate(y, p, common.youden_threshold(y, p))
    assert m["auroc"] == 1.0 and m["auprc"] == 1.0
    assert m["at_0.5"]["sensitivity"] == 1.0 and m["at_0.5"]["specificity"] == 1.0 and m["at_0.5"]["mcc"] == 1.0
    assert 0.3 < m["at_youden"]["threshold"] <= 0.8


def test_screening_operating_points():
    # 10 Normal, 10 Abnormal; one Abnormal (0.05) scores below every Normal.
    y = np.array([0] * 10 + [1] * 10)
    p = np.array([0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.60, 0.70]
                 + [0.05, 0.65, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.95])
    m = common.evaluate(y, p)
    # 90% specificity allows 1 false alarm (0.70), so the cut-off can fall to 0.65: catches 9 of 10
    assert m["sens_at_spec90"] == pytest.approx(0.9)
    # 95% sensitivity needs all 10, so the cut-off drops to 0.05 and every Normal is flagged
    assert m["spec_at_sens95"] == pytest.approx(0.0)
    assert m["spec_at_sens90"] == pytest.approx(0.9)
    thr = common.sensitivity_threshold(y, p, target=0.9)
    assert thr == pytest.approx(0.65)
    hs = common.evaluate(y, p, high_sens=thr)["at_high_sens"]
    assert hs["sensitivity"] == pytest.approx(0.9) and hs["fn"] == 1


def test_write_json_survives_brief_windows_lock(tmp_path, monkeypatch):
    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("[WinError 5] Access is denied")
        real(src, dst)

    monkeypatch.setattr(common.os, "replace", flaky)
    common.write_json(tmp_path / "s.json", {"a": 1})
    assert common.read_json(tmp_path / "s.json") == {"a": 1} and len(calls) == 3


def test_report_keeps_handwritten_findings(tmp_path):
    """summary regenerates every table but must copy the findings block over unchanged."""
    import shutil

    if not (common.RESULTS_DIR / "runs.csv").exists():
        pytest.skip("no results yet")
    shutil.copy(common.RESULTS_DIR / "runs.csv", tmp_path / "runs.csv")
    first = common.write_report(tmp_path)
    text = first.read_text(encoding="utf-8")
    assert common.FINDINGS_START in text and "## All runs" in text
    mine = "\n".join([common.FINDINGS_START, "## Findings", "", "My notes.", common.FINDINGS_END])
    start, end = text.index(common.FINDINGS_START), text.index(common.FINDINGS_END) + len(common.FINDINGS_END)
    first.write_text(text[:start] + mine + text[end:], encoding="utf-8")
    again = common.write_report(tmp_path).read_text(encoding="utf-8")
    assert mine in again and again.count(common.FINDINGS_START) == 1


def test_upsert_csv_replaces_same_key(tmp_path):
    path = tmp_path / "x.csv"
    common.upsert_csv(path, {"run": "a", "v": 1}, key="run")
    common.upsert_csv(path, {"run": "b", "v": 2, "extra": 3}, key="run")
    common.upsert_csv(path, {"run": "a", "v": 9}, key="run")
    rows = common.read_csv(path)
    assert [r["run"] for r in rows] == ["b", "a"] and rows[1]["v"] == "9" and "extra" in rows[0]


def test_evaluate_does_not_clip_ranking_metrics():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.99999994, 0.99999997, 1e-9, 1e-8])
    # 3 of 4 positive/negative pairs are ranked correctly; clipping at 1e-7 would tie them all (0.5)
    assert common.evaluate(y, p)["auroc"] == pytest.approx(0.75)


def test_plain_turns_numpy_nan_into_null():
    assert common._plain({"a": np.float64("nan"), "b": np.float32(1.5), "c": [np.inf]}) == {"a": None, "b": 1.5, "c": [None]}


def test_variant_names_keep_override_runs_apart():
    recipe = common.full_recipe(lstm.RECIPE)
    assert common.variant_name(recipe) == "default"
    assert common.variant_name(recipe, lr=1e-3) == "default"
    assert common.variant_name(recipe, lr=3e-4) == "lr0.0003"
    assert common.variant_name(recipe, lr=3e-4, epochs=20) == "lr0.0003_ep20"
    assert common.run_name("lstm_base", 20, 0, 1) == "lstm_base_T20_s0_f1"
    assert common.run_name("lstm_base", 20, 0, 1, "lr0.0003") == "lstm_base_T20_s0_f1_lr0.0003"


def test_log_event_appends_json_lines(tmp_path):
    common.log_event(tmp_path, "a", x=1)
    common.log_event(tmp_path, "b", y=np.float32(2))
    lines = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [l["kind"] for l in lines] == ["a", "b"] and lines[1]["y"] == 2.0


def _toy_fold(n=40, t=10):
    rng = np.random.default_rng(0)
    labels = np.zeros(n, dtype="int8")
    order = rng.permutation(n)
    idx = {"train": order[:24], "val": order[24:32], "test": order[32:]}
    for part in idx.values():
        labels[part[:3]] = 1
    data = {"features": rng.normal(size=(n, t, 1280)).astype("float32"), "labels": labels,
            "ids": np.array([f"video-{i}.mp4" for i in range(n)]), "ages": np.full(n, 10, "int16")}
    return data, idx


@pytest.mark.parametrize("family,preset", [("lstm", "lstm_small"), ("transformer", "transformer_small")])
def test_train_fold_writes_everything(tmp_path, family, preset):
    data, idx = _toy_fold()
    mod = MODULES[family]
    out = tmp_path / "run"
    rec = common.train_fold(mod.build, family, preset, 10, 0, 0, data, idx, mod.RECIPE, out,
                            max_epochs=3, verbose=0)
    for name in ("run.json", "status.json", "history.csv", "predictions.csv", "model.keras",
                 "model_summary.txt", "model_architecture.json", "val_probs_per_epoch.npy"):
        assert (out / name).exists(), name
    saved = json.loads((out / "run.json").read_text())
    assert saved["training"]["epochs_run"] == 3 and saved["metrics"]["test"]["n"] == 8
    assert {"auroc", "auprc", "brier", "at_0.5", "at_youden"} <= saved["metrics"]["test"].keys()
    assert saved["memory"]["rss_peak_mb"] > 0 and saved["environment"]["tensorflow"]
    assert saved["model"]["params_by_layer"]
    assert json.loads((out / "status.json").read_text())["status"] == "completed"
    history = common.read_csv(out / "history.csv")
    assert len(history) == 3
    assert {"loss", "val_auc", "val_auprc", "val_auroc_exact", "lr", "epoch_seconds", "train_seconds",
            "val_seconds", "step_ms_median", "rss_mb", "weight_norm"} <= history[0].keys()
    if family == "transformer":   # warm-up: the LR logged for epoch 1 is the one used in epoch 1
        assert float(history[0]["lr"]) == pytest.approx(3e-4 / 5)
    assert np.load(out / "val_probs_per_epoch.npy").shape == (3, 8)
    assert len(common.read_csv(out / "predictions.csv")) == 40
    assert common.read_csv(tmp_path / "runs.csv")[0]["run"] == rec["run"] == "run"
    kinds = [json.loads(l)["kind"] for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert kinds == ["run_start", "epoch_end", "epoch_end", "epoch_end", "run_end"]


def test_crashed_run_keeps_its_data_and_next_attempt_starts_fresh(tmp_path, monkeypatch):
    data, idx = _toy_fold()
    out = tmp_path / "run"
    real_evaluate = common.evaluate
    calls = {"n": 0}

    def flaky(*args, **kwargs):           # epochs 1-2 log fine, then the run "breaks"
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated crash")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(common, "evaluate", flaky)
    with pytest.raises(RuntimeError):
        common.train_fold(lstm.build, "lstm", "lstm_small", 10, 0, 0, data, idx, lstm.RECIPE, out,
                          max_epochs=4, verbose=0)
    status = json.loads((out / "status.json").read_text())
    assert status["status"] == "failed" and "simulated crash" in status["traceback"]
    assert len(common.read_csv(out / "history.csv")) == 2          # data up to the crash is kept

    monkeypatch.setattr(common, "evaluate", real_evaluate)
    common.train_fold(lstm.build, "lstm", "lstm_small", 10, 0, 0, data, idx, lstm.RECIPE, out,
                      max_epochs=2, verbose=0)
    old = out / "attempts" / "attempt1_failed"
    assert len(common.read_csv(old / "history.csv")) == 2           # crashed attempt archived intact
    assert json.loads((out / "status.json").read_text())["attempt"] == 2
    assert len(common.read_csv(out / "history.csv")) == 2           # new attempt, new file
    kinds = [json.loads(l)["kind"] for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "run_failed" in kinds and kinds[-1] == "run_end"


def test_dead_process_is_archived_as_crashed(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    common.write_json(out / "status.json", {"status": "running", "attempt": 1})
    (out / "history.csv").write_text("epoch,loss\n1,0.5\n")
    assert common.archive_previous_attempt(out) == 2
    assert (out / "attempts" / "attempt1_crashed" / "history.csv").exists()
    assert [p.name for p in out.iterdir()] == ["attempts"]


def test_benchmark_model_logs(tmp_path):
    rows = common.benchmark_model(lstm.build, "lstm", "lstm_small", 10, lstm.macs,
                                  batch_sizes=(1, 2), warmup=2, timed=5, out_dir=tmp_path / "bench")
    assert [r["batch"] for r in rows] == [1, 2]
    assert rows[0]["median_ms"] > 0 and rows[0]["clips_per_second"] > 0
    assert rows[0]["params"] == lstm.build("lstm_small", 10).count_params()
    assert len(common.read_csv(tmp_path / "benchmark.csv")) == 2
    assert len(list((tmp_path / "bench").glob("*.json"))) == 2
    raw = json.loads(next((tmp_path / "bench").glob("*_b1_*.json")).read_text())["raw_ms"]
    assert len(raw) == 5


def test_benchmark_grid_jobs_name_their_outputs():
    jobs = common.benchmark_jobs(sessions=1)
    presets = sum(len(m.PRESETS) for m in MODULES.values())
    assert len(jobs) == presets * 3 * 2 + 3 * 2
    assert all(j["outputs"] and j["args"][0] == "-m" for j in jobs)
