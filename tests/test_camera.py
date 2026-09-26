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


def test_from_far_and_low_the_pose_keeps_to_the_goals_width():
    """From the ground 27 ft out a pixel's error in the crossbar swings a pose
    fitted to the corners by 8 in, and real outlines err most in just that
    way; the homography's pose, scaled by the goal's width, moves under 2."""
    from shottracker.camera import _planar_pose

    cam = camera_preset("backyard", 1080, 1920)
    quad = cam.project(np.hstack([outer_rect(GoalSpec()), np.zeros((4, 1))]))
    K = np.array([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]])
    assert _planar_pose(outer_rect(GoalSpec()), quad, K) is None
    near = camera_preset("angled")
    quad = near.project(np.hstack([outer_rect(GoalSpec()), np.zeros((4, 1))]))
    K = np.array([[near.fx, 0.0, near.cx], [0.0, near.fy, near.cy], [0.0, 0.0, 1.0]])
    assert _planar_pose(outer_rect(GoalSpec()), quad, K) is not None


def test_feet_hidden_in_grass_are_put_back():
    """From the ground the lawn hides the bottom of the posts; with the lens
    known, the outline's proportions say how much, and where the feet are."""
    from shottracker.camera import hidden_feet

    cam = camera_preset("backyard", 1080, 1920)
    K = np.array([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]])
    goal = GoalSpec()
    true = cam.project(np.hstack([outer_rect(goal), np.zeros((4, 1))]))
    seen = true.copy()
    hw = goal.outer_width_in / 2.0
    seen[2:] = cam.project(np.array([[hw, 7.0, 0.0], [-hw, 7.0, 0.0]]))   # the red stops 7 in up
    quad, hidden = hidden_feet(seen, K, goal)
    assert hidden == pytest.approx(7.0, abs=0.5)
    assert np.abs(quad - true).max() < 1.0
    assert hidden_feet(true, K, goal) is None


def _on_the_gravel():
    """A 0.5x phone in portrait on the ground 25 ft out and 9 ft to the side,
    the net 180 px wide with 8 in of its posts in the grass: the real
    slow-motion clips' view."""
    from shottracker.camera import _look_at

    spec = GoalSpec()
    W, H, f = 1080, 1920, 712.75
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    centre = np.array([110.0, 4.0, 307.0])
    R = _look_at(centre, np.array([-150.0, 70.0, 0.0]))
    t = -R @ centre
    hw, top = spec.outer_width_in / 2, spec.outer_height_in
    pts = (R @ np.array([[-hw, top, 0], [hw, top, 0], [hw, 8.0, 0], [-hw, 8.0, 0]]).T).T + t
    seen = (K @ (pts / pts[:, 2:3]).T).T[:, :2]
    # The near post leans in the picture and the outline follows it a
    # couple of pixels short, as on the real clips.
    misread = seen + np.array([[0, 0], [2.5, 0], [-2.5, 0], [0, 0]])
    return spec, K, centre, misread


def test_a_phone_on_the_ground_is_held_there_when_the_net_is_small():
    from shottracker.camera import ground_pose, hidden_feet

    spec, K, centre, quad = _on_the_gravel()
    restored = hidden_feet(quad, K, spec)
    q = restored[0] if restored is not None else quad
    plane = GoalPlane(q, spec)
    free = calibrate_from_homography(plane.H, (1080, 1920), outer_rect(spec), plane.image_quad,
                                     known_focal_px=K[0, 0], known_focal_spread=0.06)
    # The outline alone cannot place it: the phone ends up four feet in the air.
    assert not free.pose_from_corners
    assert free.position[1] > 30.0

    R, t, hidden, rms = ground_pose(quad, K, spec, 4.0)
    assert np.linalg.norm(-R.T @ t - centre) < 36.0
    assert hidden == pytest.approx(8.0, abs=2.0)
    # How high exactly hardly matters.
    for h in (-2.0, 10.0):
        R, t, _, _ = ground_pose(quad, K, spec, h)
        assert np.linalg.norm((-R.T @ t - centre)[[0, 2]]) < 36.0


def test_held_at_the_true_height_the_outline_gives_the_pose_back():
    from shottracker.camera import ground_pose

    spec, K, centre, quad = _on_the_gravel()
    exact = quad - np.array([[0, 0], [2.5, 0], [-2.5, 0], [0, 0]])
    R, t, hidden, rms = ground_pose(exact, K, spec, 4.0)
    assert np.allclose(-R.T @ t, centre, atol=0.5)
    assert hidden == pytest.approx(8.0, abs=0.1) and rms < 0.05
