"""Turning one slow-motion shot into the normal-speed clips a phone would have made."""

import cv2

from shottracker.fullspeed import Reading, decimate, format_table
from shottracker.synth import SynthShot, render_session


def test_every_starting_frame_becomes_its_own_normal_speed_clip(tmp_path):
    src = str(tmp_path / "slowmo.mp4")
    render_session(src, [SynthShot(target_x_in=10, target_y_in=20, start_time_s=0.1, flight_time_s=0.25)],
                   fps=240, duration_s=0.5)
    capture, versions = decimate(src, str(tmp_path), rates=(30, 60))
    assert capture == 240
    assert sorted({(r, p) for r, p, _ in versions}) == [(30, p) for p in range(8)] + [(60, p) for p in range(4)]
    for rate, phase, path in versions:
        cap = cv2.VideoCapture(path)
        assert cap.get(cv2.CAP_PROP_FPS) == rate
        # Every (240 / rate)-th frame from the starting one.
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(range(phase, 120, 240 // rate))
        cap.release()


def test_a_rate_that_is_not_a_whole_step_is_skipped(tmp_path):
    src = str(tmp_path / "slowmo.mp4")
    render_session(src, [SynthShot(target_x_in=0, target_y_in=20, start_time_s=0.1, flight_time_s=0.25)],
                   fps=240, duration_s=0.4)
    _, versions = decimate(src, str(tmp_path), rates=(50,))
    assert versions == []


def test_the_summary_says_how_much_normal_speed_loses():
    result = {"capture_fps": 240.0, "reference_mph": 40.0, "reference_xy": [10.0, 20.0], "readings": [
        Reading(30, 0, 1, 36.0, -10.0, 4.0, "on_net"),
        Reading(30, 1, 0, None, None, None, None),
        Reading(60, 0, 1, 40.4, 1.0, 1.0, "on_net"),
    ]}
    text = format_table(result)
    assert "40.0 mph" in text and "none" in text
    assert "at 30 fps: speed within 10.0%" in text and "(1 of 2 versions read)" in text
