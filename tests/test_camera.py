"""The calibration must recover a camera we placed ourselves."""

import numpy as np
import pytest

from shottracker.camera import calibrate_from_homography
from shottracker.config import GoalSpec
from shottracker.geometry import GoalPlane, outer_rect
from shottracker.synth import camera_preset

PRESETS = ["side", "angled", "head_on", "extreme_side"]


def _calibrate(preset: str):
    spec = GoalSpec()
    cam = camera_preset(preset)
    corners3d = np.hstack([outer_rect(spec), np.zeros((4, 1))])
    quad = cam.project(corners3d)
    plane = GoalPlane(quad, spec)
    model = calibrate_from_homography(
        plane.H, (cam.width, cam.height), outer_rect(spec), plane.image_quad
    )
    return cam, model


@pytest.mark.parametrize("preset", PRESETS)
def test_recovers_focal_length(preset):
    cam, model = _calibrate(preset)
    assert model is not None
    assert model.focal_px == pytest.approx(cam.fx, rel=0.02)


@pytest.mark.parametrize("preset", PRESETS)
def test_recovers_camera_position(preset):
    cam, model = _calibrate(preset)
    # Within an inch, from four corners and the knowledge that a goal is 6x4.
    assert np.linalg.norm(model.position - cam.position) < 1.0


@pytest.mark.parametrize("preset", PRESETS)
def test_reprojects_the_corners_it_was_given(preset):
    _, model = _calibrate(preset)
    assert model.residual_px < 0.5


@pytest.mark.parametrize("preset", PRESETS)
def test_projection_agrees_with_the_true_camera(preset):
    cam, model = _calibrate(preset)
    probes = np.array([[0.0, 24.0, 0.0], [-20.0, 40.0, 60.0], [10.0, 2.0, 240.0]])
    assert np.abs(model.project(probes) - cam.project(probes)).max() < 2.0


def test_declines_when_the_geometry_is_degenerate():
    # A perfectly fronto-parallel view carries no focal-length information.
    quad = np.array([[100, 100], [867.5, 100], [867.5, 603.75], [100, 603.75]], dtype=float)
    plane = GoalPlane(quad)
    assert calibrate_from_homography(plane.H, (1280, 720)) is None


def test_shot_line_angle_is_larger_from_the_side():
    from_point = np.array([0.0, 0.5, 240.0])
    _, side = _calibrate("side")
    _, head_on = _calibrate("head_on")
    assert side.shot_line_angle_deg(from_point) > head_on.shot_line_angle_deg(from_point)
