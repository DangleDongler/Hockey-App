"""Scoring shots against what the player was aiming at."""

import numpy as np
import pytest

from shottracker.config import GoalSpec, TargetConfig
from shottracker.targets import (
    resolve_targets,
    score_shot,
    summarize_scores,
    targeting_block,
)

GOAL = GoalSpec()


def test_named_targets_sit_in_from_the_pipe():
    (tl,) = resolve_targets(TargetConfig(kind="top_left"), GOAL)
    assert (tl.x_in, tl.y_in) == (-30.0, 42.0)
    (fh,) = resolve_targets(TargetConfig(kind="five_hole"), GOAL)
    assert (fh.x_in, fh.y_in) == (0.0, 6.0)


def test_targets_scale_with_a_smaller_net():
    small = GoalSpec(mouth_width_in=54.0, mouth_height_in=44.0)
    (tr,) = resolve_targets(TargetConfig(kind="top_right"), small)
    assert (tr.x_in, tr.y_in) == (21.0, 38.0)


def test_a_group_expands_to_its_members():
    keys = {t.key for t in resolve_targets(TargetConfig(kind="any_corner"), GOAL)}
    assert keys == {"top_left", "top_right", "bottom_left", "bottom_right"}


def test_no_target_means_no_scoring():
    assert resolve_targets(TargetConfig(), GOAL) == []
    assert targeting_block([(0.0, 20.0, "on_net")], GOAL, TargetConfig()) is None


def test_custom_needs_an_aim_point():
    with pytest.raises(ValueError):
        resolve_targets(TargetConfig(kind="custom"), GOAL)
    (c,) = resolve_targets(TargetConfig(kind="custom", x_in=10, y_in=20, radius_in=4), GOAL)
    assert (c.x_in, c.y_in, c.radius_in) == (10.0, 20.0, 4.0)


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError):
        resolve_targets(TargetConfig(kind="glove_side"), GOAL)


def test_each_shot_is_scored_against_the_corner_it_was_going_for():
    targets = resolve_targets(TargetConfig(kind="any_corner"), GOAL)
    s = score_shot(28.0, 7.0, "on_net", targets, GOAL)
    assert s.target_key == "bottom_right"
    assert s.hit
    assert s.dx_in == pytest.approx(-2.0) and s.dy_in == pytest.approx(1.0)


def test_the_radius_is_the_line_between_hit_and_miss():
    targets = resolve_targets(TargetConfig(kind="top_left", radius_in=6.0), GOAL)
    assert score_shot(-30.0, 42.0 - 5.9, "on_net", targets, GOAL).hit
    assert not score_shot(-30.0, 42.0 - 6.1, "on_net", targets, GOAL).hit


def test_ringing_it_off_the_pipe_inside_the_circle_is_not_a_hit():
    """A post is a near miss, not a goal, however close to the target."""
    targets = resolve_targets(TargetConfig(kind="top_left", radius_in=8.0), GOAL)
    s = score_shot(-35.0, 44.0, "post", targets, GOAL)
    assert s.distance_in < 8.0
    assert not s.hit


def test_a_consistent_lean_is_reported_with_its_direction():
    targets = resolve_targets(TargetConfig(kind="top_right"), GOAL)
    rng = np.random.default_rng(1)
    # Every shot lands about 5" low and 3" left of the top-right target.
    shots = [(30.0 - 3 + rng.normal(0, 1), 42.0 - 5 + rng.normal(0, 1)) for _ in range(8)]
    summary = summarize_scores([score_shot(x, y, "on_net", targets, GOAL) for x, y in shots])
    bias = summary["bias"]
    assert bias["significant"]
    assert "low" in bias["description"] and "left" in bias["description"]


def test_scatter_around_the_target_is_not_called_a_lean():
    targets = resolve_targets(TargetConfig(kind="top_right"), GOAL)
    ring = [(30 + 5 * np.cos(a), 42 + 5 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    summary = summarize_scores([score_shot(x, y, "on_net", targets, GOAL) for x, y in ring])
    assert not summary["bias"]["significant"]


def test_two_shots_are_too_few_to_call_a_lean():
    targets = resolve_targets(TargetConfig(kind="top_right"), GOAL)
    scores = [score_shot(25.0, 36.0, "on_net", targets, GOAL) for _ in range(2)]
    summary = summarize_scores(scores)
    assert not summary["bias"]["significant"]
    assert "too few" in summary["bias"]["description"]


def test_the_summary_counts_add_up():
    block = targeting_block(
        [(-30, 42, "on_net"), (30, 6, "on_net"), (0, 8, "on_net"), (45, 29, "miss")],
        GOAL,
        TargetConfig(kind="any_corner"),
    )
    s = block["summary"]
    assert (s["hits"], s["shots"]) == (2, 4)
    assert s["hit_rate_pct"] == pytest.approx(50.0)
    assert len(block["per_shot"]) == 4


# --- through the pipeline and the server ------------------------------------

def test_a_finished_session_can_be_rescored_without_the_video(angled_result):
    _, result = angled_result
    try:
        result.config.target.kind = "any_corner"
        d = result.to_dict()
        assert d["targeting"]["label"] == "Any Corner"
        assert all("vs_target" in s for s in d["shots"])
        # The top-left and bottom-right shots are the ones that were aimed well.
        hits = [s["vs_target"]["hit"] for s in d["shots"]]
        assert hits.count(True) == 2
        assert all(len(t["image_outline"]) > 10 for t in d["targeting"]["targets"])
    finally:
        result.config.target.kind = None


def test_the_server_rescores_a_session_instantly(angled_clip):
    import sys
    from pathlib import Path

    from fastapi.testclient import TestClient

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
    import app as server

    client = TestClient(server.app)
    with open(angled_clip["path"], "rb") as fh:
        res = client.post(
            "/api/analyze",
            files={"video": ("clip.mp4", fh, "video/mp4")},
            data={"shot_distance_ft": "20", "target": "top_shelf"},
        )
    assert res.status_code == 202
    job_id = res.json()["id"]

    # TestClient runs background tasks before returning, so the job is done.
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done", job.get("error")
    assert job["result"]["targeting"]["kind"] == "top_shelf"

    rescored = client.post(f"/api/jobs/{job_id}/target", data={"target": "five_hole"}).json()
    assert rescored["result"]["targeting"]["kind"] == "five_hole"
    cleared = client.post(f"/api/jobs/{job_id}/target", data={"target": ""}).json()
    assert cleared["result"]["targeting"] is None

    bad = client.post(f"/api/jobs/{job_id}/target", data={"target": "glove_side"})
    assert bad.status_code == 400
    client.delete(f"/api/jobs/{job_id}")
