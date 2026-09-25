"""Reading frames fast, ignoring shimmer, and timing the flight from the release."""

import cv2
import numpy as np
import pytest

from shottracker import video
from shottracker.config import Config
from shottracker.puck_detect import PuckDetector, build_background_and_noise
from shottracker.speed import in_flight


def _opencv_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def test_the_fast_reader_gives_the_same_frames_as_opencv(angled_clip):
    ref = _opencv_frames(angled_clip["path"])
    got = list(video.iter_frames(angled_clip["path"]))
    assert len(got) == len(ref)
    for a, b in zip(got[::17], ref[::17]):
        assert a.shape == b.shape
        assert np.abs(a.astype(int) - b.astype(int)).mean() < 2.0


def test_the_fast_reader_scales_while_decoding(angled_clip):
    h, w = _opencv_frames(angled_clip["path"])[0].shape[:2]
    size = (w // 2, h // 2)
    frames = list(video.iter_frames(angled_clip["path"], size))
    assert frames and all(f.shape[:2] == (size[1], size[0]) for f in frames)


def test_opencv_is_the_fallback_without_pyav(angled_clip, monkeypatch):
    monkeypatch.setattr(video, "av", None)
    frames = list(video.iter_frames(angled_clip["path"]))
    assert len(frames) == angled_clip["frames"]


def test_keyframe_sampling_declines_when_a_clip_has_too_few(angled_clip):
    assert video.keyframes(angled_clip["path"], 10_000) is None


def test_shimmering_ground_is_ignored_and_a_puck_on_it_is_not():
    rng = np.random.default_rng(0)
    base = np.full((200, 300, 3), 150, np.uint8)

    def frame():
        f = base.copy()
        # The right half is gravel near the lens: it shimmers by +-25 levels.
        f[:, 150:] = np.clip(150 + rng.normal(0, 12, (200, 150, 1)), 0, 255).astype(np.uint8)
        return f

    plate, noise = build_background_and_noise([frame() for _ in range(40)])
    cfg = Config()
    test = frame()
    cv2.circle(test, (225, 100), 5, (20, 20, 20), -1)   # a puck over the gravel
    flat = PuckDetector(cfg, plate, puck_px=10.0, scale=1.0)
    aware = PuckDetector(cfg, plate, puck_px=10.0, scale=1.0, noise=noise)
    many = flat.detect(test, 0, limit=1000)
    few = aware.detect(test, 0, limit=1000)
    assert len(many) > 5 * max(len(few), 1)
    assert any(abs(c.x - 225) < 3 and abs(c.y - 100) < 3 for c in few)


def test_the_flight_starts_where_the_stick_lets_go():
    # 40 frames of the blade carrying the puck slowly, then steady flight.
    frames = np.arange(100, dtype=float)
    s = np.where(frames < 40, -0.2 + 0.001 * frames, -0.16 + 0.02 * (frames - 40))
    s2, f2 = in_flight(s, frames)
    assert 38 <= f2[0] <= 41
    slope = np.polyfit(f2, s2, 1)[0]
    assert slope == pytest.approx(0.02, rel=0.05)


def test_a_track_that_is_all_flight_is_left_whole():
    frames = np.arange(30, dtype=float)
    s = 0.03 * frames
    s2, f2 = in_flight(s, frames)
    assert len(s2) == 30
