"""Tests for the frame sampler and the split generator (no real medical data needed).

Run from the repo root:  python -m pytest tests -q
"""

import cv2
import numpy as np
import pytest

from src.data.splits import make_split
from src.data.video_io import sample_frames, target_indices


def _write_video(path, n_frames, size=(64, 48)):
    """Synthetic clip whose frame i is a flat grey of brightness 5 * i, so order is checkable."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), 5 * i, dtype=np.uint8))
    writer.release()


def test_target_indices_uniform_and_padded():
    assert target_indices(100, 5).tolist() == [0, 25, 50, 74, 99]
    short = target_indices(10, 40)
    assert len(short) == 40 and short.min() == 0 and short.max() == 9
    assert np.all(np.diff(short) >= 0)


@pytest.mark.parametrize("n_frames,padded", [(50, False), (12, True)])
def test_sample_frames_shape_dtype_order(tmp_path, n_frames, padded):
    path = tmp_path / "clip.mp4"
    _write_video(path, n_frames)
    frames, meta = sample_frames(path, num_frames=40)
    assert frames.shape == (40, 224, 224, 3)
    assert frames.dtype == np.uint8
    assert meta.decoded_frames == n_frames
    assert meta.padded is padded
    brightness = frames.mean(axis=(1, 2, 3))
    assert np.all(np.diff(brightness) > -3)  # non-decreasing up to codec noise
    assert brightness[-1] > brightness[0] + 20


def test_sample_frames_bgr_to_rgb(tmp_path):
    path = tmp_path / "red.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(5):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[..., 2] = 255  # pure red in OpenCV's BGR order
        writer.write(frame)
    writer.release()
    frames, _ = sample_frames(path, num_frames=4)
    r, g, b = frames[..., 0].mean(), frames[..., 1].mean(), frames[..., 2].mean()
    assert r > 200 and g < 50 and b < 50


def test_sample_frames_missing_file(tmp_path):
    with pytest.raises(IOError):
        sample_frames(tmp_path / "nope.mp4")


def test_split_integrity():
    rng = np.random.default_rng(0)
    ids = np.array([f"video-{i}.mp4" for i in range(535)])
    labels = (rng.random(535) < 83 / 535).astype(np.int8)
    split = make_split(ids, labels, seed=0)
    assert len(split["folds"]) == 5

    all_test = []
    rate = labels.mean()
    label_of = dict(zip(ids.tolist(), labels.tolist()))
    for fold in split["folds"]:
        train, val, test = (set(fold[p]) for p in ("train", "val", "test"))
        assert not (train & val) and not (train & test) and not (val & test)
        assert len(train | val | test) == len(ids)
        for part in (test, val):
            assert abs(np.mean([label_of[v] for v in part]) - rate) < 0.05
        all_test.extend(test)
    assert sorted(all_test) == sorted(ids.tolist())  # every id tested exactly once

    assert make_split(ids, labels, seed=0) == split  # deterministic
    assert make_split(ids, labels, seed=1) != split
