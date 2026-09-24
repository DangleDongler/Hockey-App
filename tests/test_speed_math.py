"""Unit tests for the geometry underneath the speed estimators."""

import numpy as np
import pytest

from shottracker.camera import calibrate_from_homography
from shottracker.config import GoalSpec
from shottracker.geometry import GoalPlane, outer_rect
from shottracker.speed import _closest_param_on_segment, _rays
from shottracker.synth import camera_preset


@pytest.mark.parametrize("s_true", [0.0, 0.25, 0.5, 0.9, 1.0, 1.4])
def test_ray_through_a_known_point_recovers_its_parameter(s_true):
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([10.0, 0.0, 0.0])
    origin = np.array([0.0, 5.0, 10.0])
    target = a + s_true * (b - a)
    d = target - origin
    d /= np.linalg.norm(d)
    got = _closest_param_on_segment(origin, d.reshape(1, 3), a, b)[0]
    assert got == pytest.approx(s_true, abs=1e-9)


def test_parameter_advances_along_the_segment():
    a = np.array([0.0, 0.5, 240.0])
    b = np.array([-30.0, 42.0, 0.0])
    origin = np.array([170.0, 58.0, 300.0])
    dirs = []
    for s in np.linspace(0.1, 1.0, 10):
        d = (a + s * (b - a)) - origin
        dirs.append(d / np.linalg.norm(d))
    got = _closest_param_on_segment(origin, np.array(dirs), a, b)
    assert np.all(np.diff(got) > 0)
    np.testing.assert_allclose(got, np.linspace(0.1, 1.0, 10), atol=1e-9)


def test_rays_point_from_the_camera_toward_the_scene():
    spec = GoalSpec()
    cam = camera_preset("angled")
    corners3d = np.hstack([outer_rect(spec), np.zeros((4, 1))])
    plane = GoalPlane(cam.project(corners3d), spec)
    model = calibrate_from_homography(plane.H, (cam.width, cam.height), outer_rect(spec), plane.image_quad)

    target = np.array([[0.0, 24.0, 0.0]])
    uv = model.project(target)
    origin, dirs = _rays(model, uv)
    np.testing.assert_allclose(origin, model.position, atol=1e-6)
    # Walking along the ray from the camera must arrive at the target.
    to_target = target[0] - origin
    assert np.allclose(dirs[0], to_target / np.linalg.norm(to_target), atol=1e-6)


def test_an_absurd_estimate_never_wins(monkeypatch):
    """A degenerate solve returns a number, not a speed. It must not be picked
    over a workable estimate just because its residuals looked tidy."""
    from shottracker.config import Config
    from shottracker.speed import SpeedEstimate, estimate_speed

    cfg = Config()
    results = [
        SpeedEstimate(mph=87474.0, method="time_of_flight", confidence=0.97),
        SpeedEstimate(mph=62.0, method="goal_plane", confidence=0.30),
    ]

    import shottracker.speed as speed_mod

    # Drive estimate_speed's selection directly with the two candidates.
    monkeypatch.setattr(speed_mod, "estimate_time_of_flight", lambda *a, **k: results[0])
    monkeypatch.setattr(speed_mod, "estimate_ballistic_3d", lambda *a, **k: None)
    monkeypatch.setattr(speed_mod, "estimate_goal_plane", lambda *a, **k: results[1])
    chosen = estimate_speed(None, None, object(), np.array([0.0, 24.0]), 120.0, cfg)

    assert chosen is not None
    assert chosen.mph == pytest.approx(62.0)
    # Recorded for diagnosis, but not told to the player: it changes nothing
    # about the number they see.
    assert chosen.rejected == {"time_of_flight": pytest.approx(87474.0)}
    assert not any("87474" in n for n in chosen.notes)
