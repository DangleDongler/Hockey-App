"""The goal outline must be found, and found in the right place."""

import cv2
import numpy as np
import pytest

from shottracker.config import Config
from shottracker.net_detect import consensus_quad, detect_net, detect_net_in_frame
from shottracker.pipeline import sample_frames


def _frames(truth, n=12):
    return sample_frames(truth["path"], n, truth["frames"])


def _max_error_fraction(quad, truth_quad):
    """Corner error as a fraction of the goal's own width, so it is scale-free."""
    err = np.linalg.norm(np.asarray(quad) - np.asarray(truth_quad), axis=1).max()
    width = np.linalg.norm(np.asarray(truth_quad)[1] - np.asarray(truth_quad)[0])
    return err / width


@pytest.mark.parametrize("clip", ["angled_clip", "side_clip", "head_on_clip"])
def test_finds_the_net_from_every_camera_angle(clip, request):
    truth = request.getfixturevalue(clip)
    det = detect_net(_frames(truth), Config())
    assert det is not None, f"no net found in the {clip} clip"
    assert _max_error_fraction(det.quad, truth["net_quad"]) < 0.02
    assert det.confidence > 0.8


def test_ignores_the_red_goal_line_painted_on_the_ice(angled_clip):
    """The rink's red lines are the obvious false positive for a red-pipe detector."""
    frames = _frames(angled_clip, 4)
    quad, _, _ = detect_net_in_frame(frames[0], Config())
    truth = np.asarray(angled_clip["net_quad"])
    # A quad that had swallowed the goal line would be far wider than the goal.
    assert np.linalg.norm(quad[1] - quad[0]) < 1.6 * np.linalg.norm(truth[1] - truth[0])
    assert _max_error_fraction(quad, truth) < 0.03


def test_single_frame_detection_matches_the_consensus(angled_clip):
    cfg = Config()
    frames = _frames(angled_clip, 8)
    singles = [detect_net_in_frame(f, cfg) for f in frames]
    assert all(s is not None for s in singles)
    agreed = detect_net(frames, cfg)
    for quad, _, _ in singles:
        assert _max_error_fraction(quad, agreed.quad) < 0.03


def test_consensus_rejects_a_frame_that_disagrees():
    good = np.array([[100.0, 100], [300, 100], [300, 240], [100, 240]])
    quads = [good + np.random.default_rng(i).normal(0, 0.5, (4, 2)) for i in range(6)]
    quads.append(good + 400.0)  # one wildly wrong frame
    quad, keep = consensus_quad(quads, tol_frac=0.06)
    assert len(keep) == 6
    assert np.linalg.norm(quad - good, axis=1).max() < 3.0


def test_reports_when_the_camera_is_square_to_the_net(head_on_clip):
    det = detect_net(_frames(head_on_clip), Config())
    assert any("square to the net" in n for n in det.notes)


def test_gives_up_rather_than_guessing_on_a_frame_with_no_net():
    blank = np.full((720, 1280, 3), 220, dtype=np.uint8)
    cv2.circle(blank, (640, 360), 80, (40, 40, 200), -1)  # a red blob, not a goal
    assert detect_net_in_frame(blank, Config()) is None
    assert detect_net([blank] * 5, Config()) is None


# --- finding a goal by its shape --------------------------------------------
#
# The colour method needs bright red pipe.  Real backyard nets are often a dark
# maroon in shade, washed out in sun, and the same hue as the shooter's legs.
# These tests pin down the shape method that handles them, and each of the
# ways it went wrong on real footage.

MAROON = (60, 30, 120)      # BGR; shaded goal pipe, H ~170
LIGHT_BAR = (205, 205, 205)  # a crossbar washed out by sun, seen through mesh


def _goal_like(img, x0, x1, top, foot, post_w=6, bar=LIGHT_BAR):
    cv2.rectangle(img, (x0, top), (x0 + post_w, foot), MAROON, -1)
    cv2.rectangle(img, (x1 - post_w, top), (x1, foot), MAROON, -1)
    cv2.rectangle(img, (x0, top), (x1, top + post_w), bar, -1)


def _backdrop(h=720, w=1280, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 95, dtype=np.uint8)
    return np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("clip", ["side_clip", "head_on_clip"])
def test_the_shape_method_finds_a_bent_goal_from_any_angle(clip, request):
    truth = request.getfixturevalue(clip)
    cfg = Config()
    cfg.net.method = "posts"
    det = detect_net(_frames(truth), cfg)
    assert det is not None and det.method == "posts"
    assert _max_error_fraction(det.quad, truth["net_quad"]) < 0.05


def test_a_dark_goal_with_a_washed_out_crossbar_is_found():
    img = _backdrop()
    _goal_like(img, 500, 800, 300, 500)
    got = detect_net_in_frame(img, Config())
    assert got is not None
    quad, _, method = got
    assert method == "posts"
    expected = np.array([[500, 300], [800, 300], [800, 500], [500, 500]], dtype=float)
    assert np.abs(quad - expected).max() < 8


def test_upright_bars_with_nothing_across_them_are_not_a_goal():
    """A fence is a row of thin upright boards. Without a bar joining two posts'
    tops there is no goal height to read, so nothing is reported."""
    img = _backdrop()
    for x in range(420, 900, 60):
        cv2.rectangle(img, (x, 280), (x + 6, 520), MAROON, -1)
    assert detect_net_in_frame(img, Config()) is None


def test_a_goal_inside_a_backstop_frame_is_taken_over_the_backstop():
    """Taking the outer frame would scale every result down by the difference."""
    img = _backdrop()
    _goal_like(img, 380, 900, 170, 560)          # backstop frame
    _goal_like(img, 520, 760, 380, 540)          # the goal, inside it
    quad, _, _ = detect_net_in_frame(img, Config())
    width = np.linalg.norm(quad[1] - quad[0])
    assert width == pytest.approx(240, abs=12)


def test_flowers_resting_on_a_post_do_not_hide_it():
    """Red flowers touching a post's top fuse with it into one blob; the post is
    still the long straight thin run inside it."""
    from shottracker.net_detect import find_posts

    img = _backdrop()
    cv2.rectangle(img, (600, 300), (606, 520), MAROON, -1)
    cv2.circle(img, (625, 285), 30, MAROON, -1)      # shrub resting on its top
    posts = find_posts(img, Config().net)
    tall = [p for p in posts if p.height > 150]
    assert len(tall) == 1
    assert tall[0].x0 >= 598 and tall[0].x1 <= 608


def test_a_goal_that_moves_around_the_frame_is_not_given_one_outline():
    """Hand-held footage: the goal is found in each frame but in different places.
    A single outline would be wrong for most of the clip, so none is given."""
    frames = []
    for shift in (0, 60, 120, 180, 240, 300, 360, 420):
        img = _backdrop(seed=shift)
        _goal_like(img, 300 + shift, 600 + shift, 300, 500)
        frames.append(img)
    notes = []
    assert detect_net(frames, Config(), notes) is None
    assert any("camera" in n and "moved" in n for n in notes)
