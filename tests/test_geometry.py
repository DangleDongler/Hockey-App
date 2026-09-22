import numpy as np
import pytest

from shottracker.config import GoalSpec
from shottracker.geometry import (
    GoalPlane,
    build_zones,
    mouth_rect,
    order_quad,
    outer_rect,
    zone_for,
)

# A head-on goal rendered at exactly 10 px per inch.
HEAD_ON = np.array([[100, 100], [867.5, 100], [867.5, 603.75], [100, 603.75]], dtype=float)


def test_order_quad_is_insensitive_to_input_order():
    expected = order_quad(HEAD_ON)
    for roll in range(4):
        scrambled = np.roll(HEAD_ON, roll, axis=0)
        np.testing.assert_allclose(order_quad(scrambled), expected, atol=1e-9)
    reversed_winding = HEAD_ON[::-1]
    np.testing.assert_allclose(order_quad(reversed_winding), expected, atol=1e-9)


def test_goal_plane_roundtrips_points():
    plane = GoalPlane(HEAD_ON)
    pts = np.array([[0.0, 0.0], [12.0, 30.0], [-35.0, 47.0], [40.0, 55.0]])
    np.testing.assert_allclose(plane.to_goal(plane.to_image(pts)), pts, atol=1e-6)


def test_scale_matches_the_rendered_geometry():
    plane = GoalPlane(HEAD_ON)
    assert plane.px_per_inch_at(0.0, 24.0) == pytest.approx(10.0, rel=1e-6)


def test_mouth_is_inset_from_the_pipe_by_one_post():
    spec = GoalSpec()
    plane = GoalPlane(HEAD_ON, spec)
    mouth = plane.mouth_image_quad
    # The mouth's top-left sits one post diameter inside the outer corner.
    assert mouth[0][0] - HEAD_ON[0][0] == pytest.approx(spec.post_diameter_in * 10.0, rel=1e-6)
    assert mouth[0][1] - HEAD_ON[0][1] == pytest.approx(spec.post_diameter_in * 10.0, rel=1e-6)


def test_outer_rect_surrounds_the_mouth():
    spec = GoalSpec()
    outer, mouth = outer_rect(spec), mouth_rect(spec)
    assert outer[:, 0].max() > mouth[:, 0].max()
    assert outer[:, 1].max() > mouth[:, 1].max()
    # Posts stand on the ice, so both share y = 0 at the bottom.
    assert outer[:, 1].min() == mouth[:, 1].min() == 0.0


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (0, 8, "five_hole"),
        (-30, 44, "top_left"),
        (30, 44, "top_right"),
        (0, 24, "mid_mid"),
        (-30, 8, "low_left"),
        (30, 8, "low_right"),
    ],
)
def test_zones_match_hockey_naming(x, y, expected):
    zones = build_zones(GoalSpec())
    assert zone_for(zones, x, y).key == expected


def test_zones_tile_the_mouth_without_gaps():
    spec = GoalSpec()
    zones = build_zones(spec)
    rng = np.random.default_rng(0)
    xs = rng.uniform(-spec.mouth_width_in / 2, spec.mouth_width_in / 2, 500)
    ys = rng.uniform(0, spec.mouth_height_in, 500)
    assert all(zone_for(zones, x, y) is not None for x, y in zip(xs, ys))


def test_obliquity_is_zero_head_on_and_grows_off_axis():
    assert GoalPlane(HEAD_ON).obliquity == pytest.approx(0.0, abs=1e-9)
    skewed = HEAD_ON.copy()
    skewed[1][1] += 60      # foreshorten the right post
    skewed[2][1] -= 60
    assert GoalPlane(skewed).obliquity > 0.2


def test_contains_image_point():
    plane = GoalPlane(HEAD_ON)
    inside = plane.to_image([[0.0, 24.0]])[0]
    outside = plane.to_image([[60.0, 24.0]])[0]
    assert plane.contains_image_point(inside)
    assert not plane.contains_image_point(outside)
