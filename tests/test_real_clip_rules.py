"""Rules that came from the second batch of real clips.

Four clips filmed from the ground 26 ft out, on the 0.5x lens, with sunlit
trees and backstop netting behind the goal: 4K at 60 fps and 1080p slow
motion.  Each rule here fixed something those clips got wrong.
"""

import struct

import numpy as np
import pytest

from shottracker import net_detect, pipeline
from shottracker.camera import calibrate_from_homography
from shottracker.config import Config, GoalSpec
from shottracker.container import Lens, read_lens
from shottracker.geometry import GoalPlane, outer_rect
from shottracker.puck_detect import Candidate, demote_recurring, recurring
from shottracker.synth import camera_preset
from shottracker.tracking import Track, build_tracks, clear_sightings, filter_by_clutter

# --- the lens, as the phone recorded it -------------------------------------


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _meta(items: dict[str, str]) -> bytes:
    keys = b"".join(_box(b"mdta", k.encode()) for k in items)
    ilst = b"".join(
        _box(struct.pack(">I", i + 1), _box(b"data", struct.pack(">II", 1, 0) + v.encode()))
        for i, v in enumerate(items.values())
    )
    hdlr = _box(b"hdlr", b"\0" * 8 + b"mdta" + b"\0" * 12 + b"\0")
    return _box(b"meta", hdlr + _box(b"keys", b"\0\0\0\0" + struct.pack(">I", len(items)) + keys)
                + _box(b"ilst", ilst))


def _movie(tmp_path, video_meta: dict[str, str] | None) -> str:
    trak = _box(b"trak", _meta(video_meta) if video_meta else b"")
    movie_meta = _meta({"com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 17"})
    path = tmp_path / "clip.mov"
    path.write_bytes(_box(b"ftyp", b"qt  \0\0\0\0qt  ") + _box(b"mdat", b"\0" * 32)
                     + _box(b"moov", movie_meta + trak))
    return str(path)


def test_reads_the_lens_an_iphone_writes_into_the_video_track(tmp_path):
    path = _movie(tmp_path, {
        "com.apple.quicktime.camera.lens_model": "iPhone 17 back dual wide camera 2.22mm f/2.2",
        "com.apple.quicktime.camera.focal_length.35mm_equivalent": "14",
    })
    lens = read_lens(path)
    assert lens.model.startswith("iPhone 17 back dual wide")
    assert lens.focal_35mm == 14.0


def test_slow_motion_records_no_lens(tmp_path):
    assert read_lens(_movie(tmp_path, None)) is None


def test_a_35mm_equivalent_is_scaled_by_the_frame_diagonal():
    # The 0.5x lens held upright at 4K: about 74 degrees across the frame.
    f = Lens(focal_35mm=14.0).focal_px(2160, 3840)
    assert f == pytest.approx(1425.5, abs=1.0)
    assert np.degrees(2 * np.arctan(1080 / f)) == pytest.approx(74.3, abs=0.2)


def test_a_known_lens_places_a_square_on_camera_that_the_outline_cannot():
    spec = GoalSpec()
    cam = camera_preset("head_on")
    quad = cam.project(np.hstack([outer_rect(spec), np.zeros((4, 1))]))
    plane = GoalPlane(quad, spec)
    solved = calibrate_from_homography(plane.H, (cam.width, cam.height), outer_rect(spec), plane.image_quad)
    known = calibrate_from_homography(plane.H, (cam.width, cam.height), outer_rect(spec), plane.image_quad,
                                      known_focal_px=cam.fx, known_focal_spread=0.04)
    assert solved is None  # the outline alone says nothing about this lens
    assert known.focal_source == "known" and not known.focal_assumed
    assert np.linalg.norm(known.position - cam.position) < 1.0


def test_the_pipeline_prefers_a_field_of_view_given_over_the_files_lens():
    info = pipeline.VideoInfo(path="x", width=2160, height=3840, fps=60.0, frame_count=10,
                              lens="0.5x", focal_35mm=14.0)
    cfg = Config()
    f, _, source = pipeline.known_focal_px(info, cfg)
    assert f == pytest.approx(1425.5, abs=1.0) and "0.5x" in source
    cfg.camera.hfov_deg = 90.0
    f, _, source = pipeline.known_focal_px(info, cfg)
    assert f == pytest.approx(1080.0) and source == "the field of view given"


def test_a_lens_chosen_by_hand_is_used_only_when_the_file_is_silent():
    cfg = Config()
    cfg.camera.focal_35mm = Config().camera.LENS_CHOICES["1x"]
    silent = pipeline.VideoInfo(path="x", width=1080, height=1920, fps=240.0, frame_count=10)
    f, _, source = pipeline.known_focal_px(silent, cfg)
    assert f == pytest.approx(28.0 * np.hypot(1080, 1920) / 43.27) and "chosen" in source
    recorded = pipeline.VideoInfo(path="x", width=1080, height=1920, fps=60.0, frame_count=10,
                                  lens="0.5x", focal_35mm=14.0)
    f, _, source = pipeline.known_focal_px(recorded, cfg)
    assert "as the phone recorded it" in source


def test_the_rate_comes_from_the_frames_themselves(angled_clip):
    assert pipeline.decoded_frame_rate(angled_clip["path"]) == pytest.approx(angled_clip["fps"], rel=0.01)


# --- a steady camera is not a moving one -------------------------------------


def test_frames_that_miss_the_goal_do_not_count_as_the_camera_moving(monkeypatch):
    # The small net in glare was found in 8 of 24 frames, all in one place.
    quad = np.array([[818.0, 999], [1018, 979], [1024, 1095], [820, 1099]])
    calls = iter([quad] * 8 + [None] * 16)

    def fake(frame, cfg, rng=None):
        q = next(calls)
        return None if q is None else (q.copy(), 0.75, "posts")

    monkeypatch.setattr(net_detect, "detect_net_in_frame", fake)
    notes: list[str] = []
    det = net_detect.detect_net([np.zeros((4, 4, 3), np.uint8)] * 24, Config(), notes)
    assert det is not None, notes
    assert det.frames_used == 8
    assert np.abs(det.quad - quad).max() < 1e-6


# --- clutter that stays put ----------------------------------------------------


def _c(frame: int, x: float, y: float, score: float = 0.5) -> Candidate:
    return Candidate(frame=frame, x=x, y=y, area_px=20, score=score, aspect=1.5, angle_deg=0.0,
                     length_px=6.0, darkness=0.5)


def _scene(frames: int = 60, leaves: int = 30, seed: int = 0) -> tuple[dict, dict]:
    """Leaves flickering in fixed spots, and a puck crossing once."""
    rng = np.random.default_rng(seed)
    spots = rng.uniform([0, 0], [400, 300], size=(leaves, 2))
    by_frame, puck = {}, {}
    for f in range(frames):
        cs = [_c(f, *(p + rng.normal(0, 1.0, 2)), score=0.8) for p in spots if rng.random() < 0.7]
        if 20 <= f < 40:
            puck[f] = (500.0 + 15 * (f - 20), 400.0 - 5 * (f - 20))
            cs.append(_c(f, *puck[f], score=0.6))
        by_frame[f] = sorted(cs, key=lambda c: -c.score)
    return by_frame, puck


def test_flicker_in_one_spot_is_recurring_and_a_passing_puck_is_not():
    by_frame, puck = _scene()
    flagged = recurring(by_frame, radius_px=6.0, window=8, gap=2, min_hits=3)
    puck_ids = {(f, next(i for i, c in enumerate(by_frame[f]) if (c.x, c.y) == puck[f])) for f in puck}
    assert not puck_ids & flagged
    leaves = sum(len(v) for v in by_frame.values()) - len(puck)
    assert len(flagged) > 0.7 * leaves


def test_the_cap_keeps_the_puck_ahead_of_brighter_clutter():
    by_frame, puck = _scene()
    cfg = Config()
    capped, demoted, busy = demote_recurring(by_frame, 6.0, 240.0, cfg)
    assert demoted > 0
    for f, xy in puck.items():
        assert any((c.x, c.y) == xy for c in capped[f])
        assert len(capped[f]) <= cfg.puck.max_candidates_per_frame


# --- a track that bends where the flight begins and ends -----------------------


def test_the_flight_is_kept_when_the_track_bends_at_both_ends():
    cfg = Config()
    gw = 200.0
    cands = {}
    # Dragged along the ice, then flying up and across, then dropping in the net.
    for f in range(0, 6):
        cands[f] = [_c(f, 100.0 + 8 * f, 500.0)]
    for f in range(6, 26):
        cands[f] = [_c(f, 148.0 + 30 * (f - 6), 500.0 - 12 * (f - 6))]
    for f in range(26, 30):
        cands[f] = [_c(f, 748.0 + 2 * (f - 26), 260.0 + 25 * (f - 26))]
    tracks = build_tracks(cands, gw, cfg)
    # The drag, once cut off, may stand as a track of its own; the speed and
    # approach rules deal with that.  The flight must be one clean track.
    flight = max(tracks, key=len)
    assert 5 <= flight.start_frame <= 7 and 24 <= flight.end_frame <= 26


# --- lines of flicker that happen to line up -----------------------------------


def test_a_track_seen_only_in_clutter_is_not_a_shot():
    cfg = Config()
    gw = 200.0
    busy = {f: np.array([[300 + 3 * k, 300 + (k % 5)] for k in range(30)], float) for f in range(20)}
    in_clutter = Track([_c(f, 300 + 10 * f, 302) for f in range(5)])
    in_clear = Track([_c(f, 600 + 30 * f, 500 - 10 * f) for f in range(5, 15)])
    assert clear_sightings(in_clutter, busy, 0.5 * gw, cfg.track.clutter_max_neighbours) == 0
    assert clear_sightings(in_clear, busy, 0.5 * gw, cfg.track.clutter_max_neighbours) == 10
    kept, dropped = filter_by_clutter([in_clutter, in_clear], busy, gw, 60.0, cfg)
    assert kept == [in_clear] and dropped == 1


# --- a stray point after the impact ------------------------------------------


def test_a_lone_detection_after_a_gap_does_not_end_the_track():
    # A clean flight, then -- three frames after the last sighting -- the
    # netting springing back near where the puck would have gone.
    cfg = Config()
    cands = {f: [_c(f, 100.0 + 20 * f, 400.0 - 4 * f)] for f in range(0, 15)}
    cands[17] = [_c(17, 100.0 + 20 * 17 - 6, 400.0 - 4 * 17 - 14)]
    at_60 = max(build_tracks(cands, 200.0, cfg, fps=60.0), key=len)
    assert at_60.end_frame == 14
    # At 30 fps a flight is a handful of frames and gaps are normal: keep it.
    at_30 = max(build_tracks(cands, 200.0, cfg, fps=30.0), key=len)
    assert at_30.end_frame == 17


def test_a_puck_in_the_pile_before_the_release_does_not_start_the_track():
    cfg = Config()
    cands = {0: [_c(0, 90.0, 405.0)]}
    cands.update({f: [_c(f, 100.0 + 20 * (f - 3), 400.0 - 4 * (f - 3))] for f in range(3, 18)})
    t = max(build_tracks(cands, 200.0, cfg, fps=60.0), key=len)
    assert t.start_frame == 3


def test_a_puck_reappearing_on_its_line_after_a_post_still_ends_the_track():
    cfg = Config()
    cands = {f: [_c(f, 100.0 + 20 * f, 400.0 - 4 * f)] for f in range(0, 15)}
    cands[17] = [_c(17, 100.0 + 20 * 17, 400.0 - 4 * 17)]  # hidden for two frames, then at the net
    t = max(build_tracks(cands, 200.0, cfg, fps=60.0), key=len)
    assert t.end_frame == 17
