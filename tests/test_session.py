"""Several clips of one goal, read as one session."""

import json

import pytest

from shottracker.config import Config
from shottracker.report import format_session_report, shot_chart_svg, summarize
from shottracker.session import ShotSession


@pytest.fixture(scope="session")
def side_result(side_clip, analyze_truth):
    return side_clip, analyze_truth(side_clip)


@pytest.fixture
def two_clips(angled_result, side_result):
    (_, angled), (_, side) = angled_result, side_result
    return ShotSession.from_results([("angled.mp4", angled), ("side.mp4", side)], Config()), angled, side


def test_shots_pool_across_clips_and_number_through_the_session(two_clips):
    session, angled, side = two_clips
    assert len(session.shots) == len(angled.shots) + len(side.shots)
    assert [s.index for s in session.shots] == list(range(len(session.shots)))
    # Clip by clip, in the order they were given, each keeping its own shot order.
    clips = [ci for ci, _ in session.placed]
    assert clips == sorted(clips)


def test_pooling_leaves_each_clips_own_shots_untouched(two_clips):
    session, _, side = two_clips
    before = [s.index for s in side.shots]
    session.shots  # noqa: B018 -- renumbering must work on copies
    assert [s.index for s in side.shots] == before


def test_statistics_cover_every_clip(two_clips):
    session, angled, side = two_clips
    s = summarize(session)
    assert s["shots"] == len(angled.shots) + len(side.shots)
    assert s["on_net"] == summarize(angled)["on_net"] + summarize(side)["on_net"]
    assert s["speed_mph"]["max"] == max(summarize(angled)["speed_mph"]["max"], summarize(side)["speed_mph"]["max"])


def test_the_target_is_scored_over_the_whole_session(two_clips):
    session, _, _ = two_clips
    session.config.target.kind = "any_corner"
    t = session.targeting()
    assert t["summary"]["shots"] == len(session.shots)
    assert len(t["per_shot"]) == len(session.shots)


def test_the_result_says_which_clip_each_shot_came_from(two_clips):
    session, angled, _ = two_clips
    d = json.loads(json.dumps(session.to_dict()))
    assert [c["name"] for c in d["clips"]] == ["angled.mp4", "side.mp4"]
    assert [s["clip"] for s in d["shots"]] == [ci for ci, _ in session.placed]
    first_side = next(s for s in d["shots"] if s["clip"] == 1)
    assert first_side["clip_shot"] == 0 and first_side["index"] == len(angled.shots)
    # Each clip keeps what is needed to draw it over its own footage.
    assert all(c["goal"]["mouth_outline"] for c in d["clips"])


def test_one_chart_and_one_report_for_the_session(two_clips):
    session, _, _ = two_clips
    svg = shot_chart_svg(session)
    assert svg.count("<circle") >= len(session.shots)
    text = format_session_report(session)
    assert "[1] angled.mp4" in text and "[2] side.mp4" in text
    assert f"Shots     {len(session.shots)}" in text


def test_a_note_every_clip_shares_is_said_once(two_clips):
    session, angled, side = two_clips
    angled.warnings.append("same note")
    side.warnings.append("same note")
    try:
        assert session.warnings.count("every clip: same note") == 1
    finally:
        angled.warnings.remove("same note")
        side.warnings.remove("same note")


def test_a_single_clip_session_reads_like_the_clip(angled_result):
    _, angled = angled_result
    session = ShotSession.from_results([("angled.mp4", angled)])
    assert format_session_report(session).startswith("Clip ")
    assert session.warnings == angled.warnings
