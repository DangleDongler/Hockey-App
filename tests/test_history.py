"""Sessions over time, and sessions that outlive the server that made them."""

import importlib
import json
import sys
from pathlib import Path

import pytest

from shottracker.config import GoalSpec, TargetConfig
from shottracker.history import progress, rescore, session_record

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))


def _result(shots, speeds, on_net, target=None):
    """Just enough of a saved session result for the history to read."""
    res = {
        "clips": [{}],
        "summary": {
            "shots": shots,
            "on_net": on_net,
            "accuracy_pct": 100.0 * on_net / shots if shots else None,
            "speed_mph": {"mean": sum(speeds) / len(speeds) if speeds else None,
                          "max": max(speeds) if speeds else None},
        },
        "targeting": None,
    }
    if target:
        hits, n = target
        res["targeting"] = {"label": "Any Corner", "summary": {"hits": hits, "shots": n}}
    return res


def test_a_session_becomes_one_line_of_history():
    r = session_record("abc", "2026-09-20T10:00:00+00:00", _result(4, [40, 50, 60, 50], 3, target=(1, 4)))
    assert r["shots"] == 4 and r["speed_mean"] == 50 and r["accuracy_pct"] == 75.0
    assert r["hit_pct"] == 25.0 and r["target_label"] == "Any Corner"


def test_the_latest_session_is_compared_with_the_average_of_the_ones_before():
    records = [
        session_record(f"s{i}", f"2026-09-{10 + i:02d}T10:00:00+00:00", _result(5, [v] * 5, 3))
        for i, v in enumerate([30, 34, 38, 45])
    ]
    p = progress(records)
    speed = p["speed_mean"]
    assert speed["latest"] == 45
    assert speed["baseline"] == pytest.approx(34.0)       # mean of 30, 34, 38
    assert speed["delta"] == pytest.approx(11.0)
    assert speed["baseline_sessions"] == 3
    assert [x["value"] for x in speed["series"]] == [30, 34, 38, 45]


def test_only_the_last_few_sessions_form_the_baseline():
    records = [
        session_record(f"s{i}", f"2026-09-{10 + i:02d}T10:00:00+00:00", _result(5, [v] * 5, 3))
        for i, v in enumerate([10, 10, 40, 40, 40, 40, 40, 50])
    ]
    speed = progress(records, baseline=5)["speed_mean"]
    assert speed["baseline"] == pytest.approx(40.0) and speed["baseline_sessions"] == 5


def test_a_first_session_has_no_trend_yet():
    p = progress([session_record("s0", "2026-09-10T10:00:00+00:00", _result(3, [40, 41, 42], 2))])
    assert p["speed_mean"]["delta"] is None and p["speed_mean"]["baseline"] is None


def test_sessions_without_shots_or_targets_stay_out_of_those_trends():
    records = [
        session_record("a", "2026-09-10T10:00:00+00:00", _result(4, [40] * 4, 2, target=(2, 4))),
        session_record("b", "2026-09-11T10:00:00+00:00", _result(0, [], 0)),
        session_record("c", "2026-09-12T10:00:00+00:00", _result(4, [44] * 4, 3)),
    ]
    p = progress(records)
    assert [x["id"] for x in p["speed_mean"]["series"]] == ["a", "c"]
    assert [x["id"] for x in p["hit_pct"]["series"]] == ["a"]


def test_a_saved_session_is_re_scored_from_its_impact_points():
    shot = {"impact_goal_in": [-30.0, 42.0], "outcome": "on_net"}
    result = {"shots": [dict(shot)], "clips": [{"net": None, "shots": [dict(shot)]}], "targeting": None}
    rescore(result, GoalSpec(), TargetConfig(kind="top_left"))
    assert result["targeting"]["summary"]["hits"] == 1
    assert result["shots"][0]["vs_target"]["hit"] is True
    assert result["clips"][0]["targeting"]["summary"]["hits"] == 1
    rescore(result, GoalSpec(), TargetConfig(kind=None))
    assert result["targeting"] is None and "vs_target" not in result["shots"][0]


# --- the server keeps sessions ----------------------------------------------


@pytest.fixture
def fresh_server(tmp_path, monkeypatch):
    """The server module loaded against an empty data directory."""
    monkeypatch.setenv("SHOTTRACKER_DATA", str(tmp_path))
    import app as server

    server = importlib.reload(server)
    yield server
    server.JOBS.clear()


def test_a_finished_session_is_saved_listed_and_restored(fresh_server, angled_clip, tmp_path):
    from fastapi.testclient import TestClient

    server = fresh_server
    client = TestClient(server.app)
    with open(angled_clip["path"], "rb") as fh:
        job_id = client.post("/api/analyze", files={"videos": ("a.mp4", fh, "video/mp4")},
                             data={"shot_distance_ft": "20", "target": "any_corner"}).json()["id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done", job.get("error")

    saved = json.loads((tmp_path / job_id / "session.json").read_text())
    assert saved["settings"]["target"]["kind"] == "any_corner"

    listed = client.get("/api/sessions").json()
    assert [s["id"] for s in listed["sessions"]] == [job_id]
    assert listed["sessions"][0]["shots"] == len(job["result"]["shots"])
    assert "speed_mean" in listed["progress"]

    # A restart forgets every job in memory; the saved session comes back.
    server.JOBS.clear()
    server._restore_saved_sessions()
    back = client.get(f"/api/jobs/{job_id}").json()
    assert back["status"] == "done"
    assert back["result"]["shots"] == job["result"]["shots"]
    assert client.get(f"/api/jobs/{job_id}/video", params={"clip": 0}).status_code == 200

    # ...and can still be re-scored, from the saved impact points alone.
    rescored = client.post(f"/api/jobs/{job_id}/target", data={"target": "five_hole"}).json()
    assert rescored["result"]["targeting"]["kind"] == "five_hole"
    again = json.loads((tmp_path / job_id / "session.json").read_text())
    assert again["result"]["targeting"]["kind"] == "five_hole"

    client.delete(f"/api/jobs/{job_id}")
    assert client.get("/api/sessions").json()["sessions"] == []


def test_a_damaged_save_file_loses_only_that_session(fresh_server, tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "session.json").write_text("{not json")
    fresh_server._restore_saved_sessions()
    assert "broken" not in fresh_server.JOBS
