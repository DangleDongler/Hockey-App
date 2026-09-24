"""A backyard like the one the first real slow-motion clip came from.

Phone propped on the ground 27 ft out and 12 ft to the side, filming in
portrait at 240 fps; the shooter 18 ft 9 in out; a bush beside the goal whose
leaves never stop moving; netting that shakes after every impact.  On real
footage the last two made fake shots -- one "at 223 mph" -- so this scene
pins down that they no longer do.
"""

import os

import numpy as np
import pytest

from shottracker.config import CameraConfig, Config
from shottracker.pipeline import analyze
from shottracker.synth import SynthShot, render_session

SHOTS = [
    # Released a little to the shooter's left of centre, 225 in out.
    dict(target_x_in=-28, target_y_in=40, flight_time_s=0.30, start_time_s=0.25),   # top shelf left
    dict(target_x_in=26, target_y_in=8, flight_time_s=0.36, start_time_s=1.05),     # low right
    dict(target_x_in=43, target_y_in=24, flight_time_s=0.33, start_time_s=1.85),    # just wide right
]


@pytest.fixture(scope="session")
def backyard(workdir):
    path = os.path.join(workdir, "backyard.mp4")
    shots = [SynthShot(release_z_in=225, release_x_in=-20, **s) for s in SHOTS]
    truth = render_session(path, shots, camera="backyard", fps=240, net_sway=True, foliage=True,
                           ice_markings=False)
    return truth


def _analyze(truth, hfov=None):
    cfg = Config()
    cfg.speed.shot_distance_ft = 18.75
    cfg.speed.shooter_offset_ft = -20 / 12
    if hfov:
        cfg.camera.assumed_focal_frac = CameraConfig.frac_from_hfov(hfov)
    return analyze(truth["path"], cfg)


@pytest.fixture(scope="session")
def backyard_result(backyard):
    return _analyze(backyard)


def test_the_goal_is_found_past_the_bush(backyard, backyard_result):
    assert backyard_result.net is not None
    assert np.abs(backyard_result.net.quad - np.asarray(backyard["net_quad"])).max() < 4


def test_every_shot_and_nothing_else(backyard, backyard_result):
    frames = [s.impact_frame for s in backyard_result.shots]
    assert len(frames) == 3, f"expected the 3 shots, got impacts at {frames}"
    for got, t in zip(frames, backyard["shots"]):
        assert abs(got - t["impact_frame"]) <= 2


def test_the_scene_really_does_tempt_the_tracker(backyard):
    """Without the rules that came from real footage, the bush and the netting
    make shots of their own -- so the test above is guarding something."""
    cfg = Config()
    cfg.speed.shot_distance_ft = 18.75
    cfg.track.min_track_s = 0.0
    cfg.shot.min_start_outside_goal = -1e9
    cfg.shot.min_approach_goal = -1e9
    cfg.shot.min_shot_separation_s = 0.0
    assert len(analyze(backyard["path"], cfg).shots) > len(SHOTS)


def test_shots_are_placed_within_a_few_inches(backyard, backyard_result):
    for shot, t in zip(backyard_result.shots, backyard["shots"]):
        err = np.hypot(shot.impact_goal_in[0] - t["impact_x_in"], shot.impact_goal_in[1] - t["impact_y_in"])
        assert err < 4.0
    assert [s.outcome for s in backyard_result.shots] == ["on_net", "on_net", "miss"]


def test_speed_error_bars_are_honest(backyard, backyard_result):
    # From this low, square-on spot the lens cannot be measured, and the
    # time-of-flight estimate reads about 14% low.  What must hold is that
    # the error bar says so: the truth sits inside two of them.
    for shot, t in zip(backyard_result.shots, backyard["shots"]):
        assert abs(shot.speed.mph - t["release_speed_mph"]) < 2 * shot.speed.uncertainty_mph


def test_with_the_lens_known_gravity_pins_the_speed(backyard):
    # The gravity-based fit is exact once the lens is right -- the lever for
    # making backyard speeds tight, pending real footage to check it on.
    result = _analyze(backyard, hfov=70.0)
    for shot, t in zip(result.shots, backyard["shots"]):
        ballistic = shot.speed.mph if shot.speed.method == "ballistic_3d" else shot.speed.alternatives["ballistic_3d"]
        assert ballistic == pytest.approx(t["release_speed_mph"], rel=0.03)
