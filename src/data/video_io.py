"""Uniform frame sampling from a video. Spec: docs/data_pipeline.md section 2.

Why this is written the way it is (each point fixes a legacy DL_test1 bug):
- Sequential decode with grab()/retrieve() instead of cap.set(POS_FRAMES): seeking in H.264 is
  slow and can land on the wrong frame. grab() skips a frame cheaply; retrieve() decodes only the
  frames we keep.
- CAP_PROP_FRAME_COUNT can be wrong, so we count the frames that actually decode. If the count
  differs from the header, we resample over the true count with a second pass.
- Short clips are repeat-padded (linspace over fewer than T frames repeats indices), never dropped.
- BGR -> RGB, INTER_AREA resize, and the result stays uint8 (no /255, no float64).
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

FRAME_SIZE = (224, 224)
MAX_CONSECUTIVE_FAILURES = 25


@dataclass
class SampleMeta:
    """What happened while sampling one video; written to the feature-cache sidecar."""

    header_frames: int
    decoded_frames: int
    fps: float
    width: int
    height: int
    padded: bool
    passes: int
    indices: list

    def to_dict(self):
        return asdict(self)


def target_indices(n_frames, num_frames):
    """Uniformly spaced frame indices; repeats indices when n_frames < num_frames."""
    return np.linspace(0, n_frames - 1, num_frames).round().astype(int)


def _prepare(frame_bgr, size):
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _decode_pass(path, wanted, size):
    """Decode the whole file sequentially, keeping frames whose index is in `wanted`.

    Returns (kept: dict index -> uint8 frame, decoded frame count). The full file is always read to
    the end so the true frame count is known even when the header lies.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    kept, idx, misses = {}, 0, 0
    try:
        # A single corrupt packet makes grab() return False mid-file (seen in video-487), so only a
        # run of consecutive failures counts as end of stream. Undecodable packets get no index.
        while misses < MAX_CONSECUTIVE_FAILURES:
            if not cap.grab():
                misses += 1
                continue
            misses = 0
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    kept[idx] = _prepare(frame, size)
            idx += 1
    finally:
        cap.release()
    return kept, idx


def sample_frames(path, num_frames=40, size=FRAME_SIZE):
    """Sample `num_frames` uniformly spaced RGB frames from a video.

    Returns (frames: uint8 array (num_frames, H, W, 3), SampleMeta).
    Raises IOError if the file cannot be opened or no frame decodes.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    passes = 1
    indices = target_indices(max(header, 1), num_frames)
    kept, decoded = _decode_pass(path, set(indices.tolist()), size)
    if decoded == 0:
        raise IOError(f"no frames decoded: {path}")

    # Header was wrong (or a wanted frame failed to retrieve): resample over the true count.
    if decoded != header or len(kept) < len(set(indices.tolist())):
        passes = 2
        indices = target_indices(decoded, num_frames)
        kept, decoded2 = _decode_pass(path, set(indices.tolist()), size)
        missing = set(indices.tolist()) - kept.keys()
        if missing:
            raise IOError(f"{len(missing)} frames failed to decode on pass 2 ({decoded2} decoded): {path}")

    frames = np.stack([kept[i] for i in indices])
    meta = SampleMeta(
        header_frames=header,
        decoded_frames=decoded,
        fps=round(fps, 3),
        width=width,
        height=height,
        padded=bool(decoded < num_frames),
        passes=passes,
        indices=indices.tolist(),
    )
    return frames, meta
