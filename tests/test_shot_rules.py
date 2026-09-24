"""Slow motion baked into the file, and telling a shot from what follows it.

Both came from the first real slow-motion clip: an iPhone handed over one
second filmed at 240 fps as eight seconds of 30 fps video, and after the puck
hit, the netting kept moving long enough to be tracked as two more "shots".
"""

import struct

import numpy as np
import pytest

from shottracker import pipeline
from shottracker.config import Config
from shottracker.container import ContainerTiming, TrackTiming, detect_slow_motion, read_timing
from shottracker.geometry import GoalPlane, GoalSpec, outer_rect
from shottracker.puck_detect import Candidate
from shottracker.shots import Shot, approaches_goal, dedupe_shots
from shottracker.tracking import Track

# --- reading a file's own timing -------------------------------------------


def _box(kind: bytes, payload: bytes, *, large: bool = False) -> bytes:
    if large:
        return struct.pack(">I4sQ", 1, kind, 16 + len(payload)) + payload
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _trak(handler: bytes, timescale: int, duration: int, stts: list[tuple[int, int]] | None,
          *, v1: bool = False) -> bytes:
    if v1:
        mdhd = bytes([1, 0, 0, 0]) + struct.pack(">QQIQ", 0, 0, timescale, duration) + b"\0\0\0\0"
    else:
        mdhd = bytes([0, 0, 0, 0]) + struct.pack(">IIII", 0, 0, timescale, duration) + b"\0\0\0\0"
    hdlr = b"\0\0\0\0" + b"\0\0\0\0" + handler + b"\0" * 12 + b"\0"
    stbl = b""
    if stts is not None:
        stbl = _box(b"stts", b"\0\0\0\0" + struct.pack(">I", len(stts))
                    + b"".join(struct.pack(">II", c, d) for c, d in stts))
    minf = _box(b"minf", _box(b"stbl", stbl))
    return _box(b"trak", _box(b"mdia", _box(b"mdhd", mdhd) + _box(b"hdlr", hdlr) + minf))


def _movie(tmp_path, *traks: bytes, moov_last: bool = True, large_mdat: bool = False) -> str:
    ftyp = _box(b"ftyp", b"qt  \0\0\0\0qt  ")
    mdat = _box(b"mdat", b"\0" * 64, large=large_mdat)
    moov = _box(b"moov", b"".join(traks))
    body = ftyp + (mdat + moov if moov_last else moov + mdat)
    path = tmp_path / "clip.mov"
    path.write_bytes(body)
    return str(path)


def test_reads_video_and_sound_lengths_from_a_phone_style_file(tmp_path):
    # The real clip: 251 frames at 80 ticks of 1/2400 s, and 1.09 s of sound.
    path = _movie(
        tmp_path,
        _trak(b"vide", 2400, 251 * 80, [(251, 80)]),
        _trak(b"soun", 48000, 52_320, None),
        large_mdat=True,
    )
    timing = read_timing(path)
    assert timing is not None
    assert timing.video.frames == 251
    assert timing.video.duration_s == pytest.approx(8.3667, abs=1e-3)
    assert timing.audio.duration_s == pytest.approx(1.09, abs=1e-3)


def test_reads_version_one_headers_and_a_movie_box_at_the_front(tmp_path):
    path = _movie(tmp_path, _trak(b"vide", 600, 6000, [(300, 20)], v1=True), moov_last=False)
    timing = read_timing(path)
    assert timing.video.duration_s == pytest.approx(10.0)
    assert timing.video.frames == 300
    assert timing.audio is None


def test_anything_that_is_not_a_movie_file_gives_no_timing(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"not a video at all, just some text that is long enough")
    assert read_timing(str(path)) is None


def test_reads_the_files_opencv_writes(angled_clip):
    timing = read_timing(angled_clip["path"])
    assert timing is not None and timing.video is not None
    assert timing.video.frames == angled_clip["frames"]


# --- deciding whether it is slow motion ------------------------------------


def _timing(video_s: float, audio_s: float | None) -> ContainerTiming:
    tracks = [TrackTiming("vide", video_s)]
    if audio_s is not None:
        tracks.append(TrackTiming("soun", audio_s))
    return ContainerTiming(tracks)


def test_eight_seconds_of_picture_over_one_of_sound_is_240_fps():
    slow = detect_slow_motion(_timing(8.37, 1.09), 30.0)
    assert slow.capture_fps == 240.0
    assert "240" in slow.note


def test_120_fps_slow_motion_is_recognised_too():
    assert detect_slow_motion(_timing(4.0, 1.0), 30.0).capture_fps == 120.0


def test_sound_as_long_as_the_picture_means_real_time():
    slow = detect_slow_motion(_timing(157.0, 157.0), 30.0)
    assert slow.capture_fps is None and slow.note is None


def test_without_sound_there_is_nothing_to_go_on():
    assert detect_slow_motion(_timing(8.37, None), 30.0).capture_fps is None


def test_a_clip_that_is_only_partly_slowed_is_flagged_not_guessed():
    # A second at normal speed then a second at 240 fps: 9 s of picture over
    # 2 s of sound implies 135 fps, which no phone films at.
    slow = detect_slow_motion(_timing(9.0, 2.0), 30.0)
    assert slow.capture_fps is None
    assert slow.note and "normal speed" in slow.note


def test_probe_times_by_the_filming_rate_and_seeks_by_the_playback_rate(angled_clip, monkeypatch):
    monkeypatch.setattr(pipeline, "read_timing", lambda path: _timing(8.0, 1.0))
    real = pipeline.probe(angled_clip["path"], Config())
    playback = angled_clip["fps"]
    assert real.fps == pytest.approx(playback * 8)
    assert real.fps_source == "slow-motion"
    assert real.playback_fps == pytest.approx(playback)
    assert real.notes


def test_an_ordinary_clip_keeps_its_own_rate(angled_clip):
    info = pipeline.probe(angled_clip["path"], Config())
    assert info.fps_source == "container"
    assert info.playback_fps is None and not info.notes


def test_a_rate_typed_by_hand_still_seeks_at_the_files_own_rate(angled_clip):
    cfg = Config()
    cfg.fps_override = angled_clip["fps"] * 2
    info = pipeline.probe(angled_clip["path"], cfg)
    assert info.fps_source == "override"
    assert info.playback_fps == pytest.approx(angled_clip["fps"])


# --- a shot has to come from somewhere -------------------------------------


@pytest.fixture
def square_goal():
    """A goal seen square-on, 200 px wide, feet at y = 500."""
    spec = GoalSpec()
    rect = outer_rect(spec)
    px_per_in = 200.0 / (rect[:, 0].max() - rect[:, 0].min())
    quad = np.column_stack([400 + rect[:, 0] * px_per_in, 500 - rect[:, 1] * px_per_in])
    return GoalPlane(quad, spec)


def _track(points, start_frame=0) -> Track:
    return Track([
        Candidate(frame=start_frame + i, x=float(x), y=float(y), area_px=20, score=0.8,
                  aspect=1.0, angle_deg=0.0, length_px=5.0, darkness=0.8)
        for i, (x, y) in enumerate(points)
    ])


def test_a_puck_coming_from_the_stick_is_a_shot(square_goal):
    # From well off to the left, arriving at the middle of the net.
    pts = np.linspace([-300, 520], [400, 440], 30)
    assert approaches_goal(_track(pts), square_goal, Config())


def test_movement_that_starts_on_the_post_is_not(square_goal):
    # Flicker hopping from one post across to the other, as the netting sways.
    pts = np.linspace([302, 460], [498, 450], 6)
    assert not approaches_goal(_track(pts), square_goal, Config())


def test_a_rebound_leaving_the_net_is_not_a_new_shot(square_goal):
    pts = np.linspace([400, 460], [150, 520], 12)
    assert not approaches_goal(_track(pts), square_goal, Config())


def test_a_camera_low_behind_the_shooter_still_sees_shots_arrive(square_goal):
    # From there a whole shot covers well under half a goal width on screen.
    pts = np.linspace([420, 580], [410, 470], 40)
    assert approaches_goal(_track(pts), square_goal, Config())


# --- one shot, not three ----------------------------------------------------


def _shot(first: int, impact: int, quality: float) -> Shot:
    return Shot(index=0, first_tracked_frame=first, impact_frame=impact, impact_image=(0.0, 0.0),
                impact_goal_in=(0.0, 24.0), outcome="on_net", zone_key=None, zone_label=None,
                miss_detail=None, speed=None, track_length=impact - first, quality=quality)


def test_movement_just_after_an_impact_never_replaces_the_shot():
    # The real clip: the shot lands at 208, the netting is tracked 216-222
    # and 241-250.  Neither is a shot, even when it tracked more cleanly.
    shots = dedupe_shots([_shot(109, 208, 0.79), _shot(216, 222, 0.95), _shot(241, 250, 0.72)],
                         Config(), fps=240.0)
    assert [(s.first_tracked_frame, s.impact_frame) for s in shots] == [(109, 208)]
    assert any("after the impact" in n for n in shots[0].notes)


def test_the_gap_between_shots_is_measured_in_seconds_not_frames():
    pair = [_shot(0, 100, 0.8), _shot(105, 120, 0.8)]
    # 20 frames is a twelfth of a second at 240 fps: the aftermath of one shot.
    assert len(dedupe_shots([_shot(s.first_tracked_frame, s.impact_frame, s.quality) for s in pair],
                            Config(), fps=240.0)) == 1
    # At 30 fps the same 20 frames is two thirds of a second: two shots.
    assert len(dedupe_shots(pair, Config(), fps=30.0)) == 2


def test_a_track_that_came_apart_mid_flight_keeps_the_better_piece():
    shots = dedupe_shots([_shot(0, 50, 0.4), _shot(30, 52, 0.9)], Config(), fps=240.0)
    assert len(shots) == 1 and shots[0].quality == 0.9
    assert shots[0].notes.count("merged with a near-simultaneous trajectory (likely a rebound)") == 1
