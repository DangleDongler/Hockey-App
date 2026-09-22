"""End-to-end grading against clips whose answers we know exactly."""

import json

import numpy as np
import pytest

from shottracker.config import Config
from shottracker.geometry import build_zones, zone_for
from shottracker.pipeline import analyze
from shottracker.report import format_text_report, shot_chart_svg, summarize


def _pairs(result, truth):
    """Match measured shots to true shots by impact frame."""
    assert len(result.shots) == len(truth["shots"]), (
        f"expected {len(truth['shots'])} shots, got {len(result.shots)}"
    )
    return list(zip(sorted(result.shots, key=lambda s: s.impact_frame), truth["shots"]))


def test_finds_every_shot(angled_result):
    truth, result = angled_result
    assert len(result.shots) == len(truth["shots"])
    assert all(s.speed is not None for s in result.shots)


def _impact_error_in(shot, t):
    return float(np.hypot(
        shot.impact_goal_in[0] - t["impact_x_in"],
        shot.impact_goal_in[1] - t["impact_y_in"],
    ))


def test_shots_on_the_net_land_within_three_inches(angled_result):
    truth, result = angled_result
    on_net = [(s, t) for s, t in _pairs(result, truth) if s.outcome == "on_net"]
    assert on_net
    for shot, t in on_net:
        err = _impact_error_in(shot, t)
        assert err < 3.0, f"shot at {t['impact_x_in']},{t['impact_y_in']} was off by {err:.1f} in"


def test_shots_that_miss_are_placed_less_precisely_but_still_close(angled_result):
    """Off the net the goal plane is being extrapolated, so accuracy degrades."""
    truth, result = angled_result
    missed = [(s, t) for s, t in _pairs(result, truth) if s.outcome != "on_net"]
    assert missed
    for shot, t in missed:
        assert _impact_error_in(shot, t) < 10.0


def test_speed_is_within_five_percent_given_the_shooting_distance(angled_result):
    """The headline claim: tell it where you shot from and it gets the speed."""
    truth, result = angled_result
    for shot, t in _pairs(result, truth):
        rel = abs(shot.speed.mph - t["release_speed_mph"]) / t["release_speed_mph"]
        assert rel < 0.05, (
            f"{shot.speed.mph:.1f} mph vs a true {t['release_speed_mph']:.1f} mph "
            f"({rel:.1%} out, method {shot.speed.method})"
        )


def test_zones_match_where_the_puck_actually_went(angled_result):
    truth, result = angled_result
    zones = build_zones(Config().goal)
    for shot, t in _pairs(result, truth):
        expected = zone_for(zones, t["impact_x_in"], t["impact_y_in"])
        if expected is None:          # the deliberate miss
            assert shot.outcome == "miss"
        else:
            assert shot.outcome == "on_net"
            assert shot.zone_key == expected.key


def test_the_wide_shot_is_called_a_miss(angled_result):
    truth, result = angled_result
    misses = [s for s in result.shots if s.outcome == "miss"]
    assert len(misses) == 1
    assert "wide right" in misses[0].miss_detail


def test_accuracy_summary_adds_up(angled_result):
    _, result = angled_result
    s = summarize(result)
    assert s["on_net"] + s["posts"] + s["misses"] == s["shots"]
    assert s["accuracy_pct"] == pytest.approx(100.0 * s["on_net"] / s["shots"])
    assert sum(s["zone_counts"].values()) == s["on_net"]
    assert s["speed_mph"]["min"] <= s["speed_mph"]["mean"] <= s["speed_mph"]["max"]


@pytest.mark.parametrize("clip", ["side_clip", "head_on_clip"])
def test_other_camera_angles_find_every_shot_and_place_it(clip, request, analyze_truth):
    truth = request.getfixturevalue(clip)
    result = analyze_truth(truth)
    for shot, t in _pairs(result, truth):
        limit = 3.0 if shot.outcome == "on_net" else 10.0
        assert _impact_error_in(shot, t) < limit


@pytest.mark.parametrize("clip", ["angled_clip", "side_clip", "head_on_clip"])
def test_reported_error_bars_actually_cover_the_truth(clip, request, analyze_truth):
    """An error bar that does not contain the answer is worse than no error bar."""
    truth = request.getfixturevalue(clip)
    result = analyze_truth(truth)
    for shot, t in _pairs(result, truth):
        assert shot.speed.uncertainty_mph is not None
        off_by = abs(shot.speed.mph - t["release_speed_mph"])
        assert off_by <= shot.speed.uncertainty_mph, (
            f"{shot.speed.mph:.1f} +/- {shot.speed.uncertainty_mph:.1f} mph does not cover "
            f"the true {t['release_speed_mph']:.1f} mph"
        )


def test_a_square_on_camera_says_it_had_to_guess_the_lens(head_on_clip, analyze_truth):
    result = analyze_truth(head_on_clip)
    assert result.camera is not None and result.camera.focal_assumed
    assert any("field of view" in w for w in result.warnings)
    # It still gets within ten percent, and admits less confidence than a side view.
    for shot in result.shots:
        assert shot.speed.confidence < 0.9


def test_a_side_camera_measures_the_lens_instead_of_guessing(side_clip, analyze_truth):
    result = analyze_truth(side_clip)
    assert result.camera is not None and not result.camera.focal_assumed


def test_speed_scales_with_a_misstated_frame_rate(angled_clip, analyze_truth):
    """Speed is linear in fps, which is why a mislabelled slow-motion clip lies."""
    base = analyze_truth(angled_clip)
    cfg = Config()
    cfg.speed.shot_distance_ft = 20.0
    cfg.fps_override = 60.0                      # half the true rate
    halved = analyze(angled_clip["path"], cfg)
    ratio = halved.shots[0].speed.mph / base.shots[0].speed.mph
    assert ratio == pytest.approx(0.5, rel=0.05)


def test_without_a_distance_it_still_reports_a_speed_and_says_it_is_weaker(angled_clip, analyze_truth):
    result = analyze_truth(angled_clip, distance_ft=None)
    assert all(s.speed is not None for s in result.shots)
    assert all(s.speed.method != "time_of_flight" for s in result.shots)
    assert any("shooting distance" in w for w in result.warnings)


def test_a_manual_net_outline_overrides_detection(angled_clip):
    cfg = Config()
    cfg.speed.shot_distance_ft = 20.0
    quad = np.asarray(angled_clip["net_quad"])
    result = analyze(angled_clip["path"], cfg, net_quad=quad)
    assert result.net.method == "manual"
    np.testing.assert_allclose(result.net.quad, quad, atol=1e-6)
    assert len(result.shots) == len(angled_clip["shots"])


def test_result_serializes_to_json(angled_result):
    _, result = angled_result
    blob = json.dumps(result.to_dict())
    back = json.loads(blob)
    assert back["net"]["method"]
    assert len(back["shots"]) == len(result.shots)
    assert back["summary"]["accuracy_pct"] is not None


def test_shot_chart_is_wellformed_svg_with_one_mark_per_shot(angled_result):
    import xml.dom.minidom

    _, result = angled_result
    svg = shot_chart_svg(result)
    doc = xml.dom.minidom.parseString(svg)
    assert doc.documentElement.tagName == "svg"
    assert len(doc.getElementsByTagName("circle")) == len(result.shots)


def test_text_report_mentions_the_essentials(angled_result):
    _, result = angled_result
    text = format_text_report(result)
    for expected in ("Net", "Shots", "Accuracy", "Speed", "mph"):
        assert expected in text


def test_a_clip_with_no_net_reports_why(workdir, tmp_path):
    import cv2

    path = str(tmp_path / "blank.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
    for _ in range(30):
        writer.write(np.full((360, 640, 3), 210, dtype=np.uint8))
    writer.release()

    result = analyze(path, Config())
    assert result.net is None
    assert result.shots == []
    assert any("no hockey net was found" in w for w in result.warnings)


def test_a_smaller_training_net_is_measured_on_its_own_terms():
    """Zones and the chart follow the goal that was configured, not a regulation one."""
    from shottracker.pipeline import SessionResult, VideoInfo

    cfg = Config()
    cfg.goal.mouth_width_in = 48.0
    cfg.goal.mouth_height_in = 36.0
    result = SessionResult(video=VideoInfo("none", 1280, 720, 30.0, 10), net=None, config=cfg)

    zones = build_zones(cfg.goal)
    five_hole = next(z for z in zones if z.key == "five_hole")
    assert five_hole.x0 == pytest.approx(-8.0)
    assert five_hole.x1 == pytest.approx(8.0)
    assert summarize(result)["zone_counts"].keys() == {z.key for z in zones}
