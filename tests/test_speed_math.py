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
