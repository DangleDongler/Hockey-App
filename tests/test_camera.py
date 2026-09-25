"""The calibration must recover a camera we placed ourselves."""

import numpy as np
import pytest

from shottracker.camera import calibrate_from_homography
from shottracker.config import GoalSpec
from shottracker.geometry import GoalPlane, outer_rect
from shottracker.synth import camera_preset

PRESETS = ["side", "angled", "head_on", "extreme_side"]


def _calibrate(preset: str, *, jitter: float = 0.0, assumed: float | None = None):
    """Calibrate from a perfectly-projected goal.

    ``jitter`` is how imprecisely the corners are taken to be known. 0 asks
    what the geometry says in the ideal case; the pipeline's default of a
    pixel asks what survives real detection.
    """
    spec = GoalSpec()
    cam = camera_preset(preset)
    corners3d = np.hstack([outer_rect(spec), np.zeros((4, 1))])
    quad = cam.project(corners3d)
    plane = GoalPlane(quad, spec)
    model = calibrate_from_homography(
        plane.H, (cam.width, cam.height), outer_rect(spec), plane.image_quad,
        assumed_focal_px=assumed, corner_jitter_px=jitter,
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



# --- what survives corners known only to a pixel ----------------------------

@pytest.mark.parametrize("preset", ["side", "angled", "extreme_side"])
def test_an_off_axis_view_still_measures_the_lens_under_realistic_noise(preset):
    cam, model = _calibrate(preset, jitter=1.0, assumed=0.8 * 1280)
    assert model is not None
    assert not model.focal_assumed, "an off-axis goal should pin the lens down"
    assert model.focal_px == pytest.approx(cam.fx, rel=0.15)


def test_a_square_on_view_admits_it_cannot_measure_the_lens():
    """Square-on, the focal length is not observable from the goal. Saying so
    and falling back to a stated lens beats reporting a confident wrong one --
    an earlier version returned 663 px against a true 1065."""
    _, model = _calibrate("head_on", jitter=1.0, assumed=0.8 * 1280)
    assert model is not None
    assert model.focal_assumed
    assert model.focal_px == pytest.approx(0.8 * 1280)


def test_with_nothing_to_fall_back_on_it_declines():
    _, model = _calibrate("head_on", jitter=1.0, assumed=None)
    assert model is None


def test_a_measured_focal_reports_how_much_it_moved():
    _, model = _calibrate("side", jitter=1.0, assumed=0.8 * 1280)
    assert 0.0 < model.focal_spread <= 0.15


# --- the pose, fitted to the corners ----------------------------------------

def _known_lens(preset: str, spec: GoalSpec, seen_as: GoalSpec, noise: float, seed: int):
    """Calibrate with the lens known from a goal of size ``spec`` taken to be ``seen_as``."""
    cam = camera_preset(preset)
    quad = cam.project(np.hstack([outer_rect(spec), np.zeros((4, 1))]))
    quad = quad + np.random.default_rng(seed).normal(0.0, noise, quad.shape)
    plane = GoalPlane(quad, seen_as)
    model = calibrate_from_homography(plane.H, (cam.width, cam.height), outer_rect(seen_as), plane.image_quad,
                                      known_focal_px=cam.fx)
    return cam, quad, model


@pytest.mark.parametrize("preset", ["angled", "side"])
def test_with_the_lens_known_the_pose_is_fitted_to_the_corners(preset):
    """A pixel of error in the outline used to move the camera about 20 in when
    the pose was read off the homography; fitted to the corners, under 10."""
    errs = [np.linalg.norm(model.position - cam.position)
            for cam, _, model in (_known_lens(preset, GoalSpec(), GoalSpec(), 1.0, s) for s in range(40))]
    assert np.median(errs) < 10.0


def test_an_outline_that_does_not_fit_the_goal_keeps_the_camera_above_ground():
    """A net smaller than the size entered: the best fit to the corners tilts
    the view to explain it and puts the camera underground, so it is not used."""
    cam, _, model = _known_lens("angled", GoalSpec(mouth_height_in=41.0), GoalSpec(), 0.0, 0)
    assert model is not None
    assert model.position[1] > 0.0


def test_the_outline_says_how_tall_the_goal_really_is():
    from shottracker.camera import outline_height_fit

    cam, quad, _ = _known_lens("angled", GoalSpec(mouth_height_in=41.0), GoalSpec(), 0.3, 0)
    K = np.array([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]])
    entered, height, best = outline_height_fit(quad, K, GoalSpec())
    assert height == pytest.approx(41.0, abs=1.0)
    assert best < entered
