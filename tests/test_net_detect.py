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
