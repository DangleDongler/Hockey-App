"""The web server with several clips in one upload."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))


@pytest.fixture(scope="module")
def client():
    import app as server
    from fastapi.testclient import TestClient

    return TestClient(server.app)


def _upload(client, *paths, **data):
    handles = [open(p, "rb") for p in paths]
    try:
        files = [("videos", (Path(p).name, fh, "video/mp4")) for p, fh in zip(paths, handles)]
        res = client.post("/api/analyze", files=files, data={"shot_distance_ft": "20", **data})
    finally:
        for fh in handles:
            fh.close()
    assert res.status_code == 202, res.text
    # TestClient runs background tasks before returning, so the job is done.
    return res.json()["id"], client.get(f"/api/jobs/{res.json()['id']}").json()


def test_two_clips_make_one_session(client, angled_clip, side_clip):
    job_id, job = _upload(client, angled_clip["path"], side_clip["path"], target="any_corner")
    try:
        assert job["status"] == "done", job.get("error")
        r = job["result"]
        assert [c["upload_index"] for c in r["clips"]] == [0, 1]
        assert len(r["shots"]) == sum(len(c["shots"]) for c in r["clips"])
        assert {s["clip"] for s in r["shots"]} == {0, 1}
        assert r["targeting"]["summary"]["shots"] == len(r["shots"])
        assert all(c["has_net"] for c in job["clips"])

        # Each clip's footage and frames are reachable on their own.
        for clip in (0, 1):
            assert client.get(f"/api/jobs/{job_id}/video", params={"clip": clip}).status_code == 200
            info = client.get(f"/api/jobs/{job_id}/info", params={"clip": clip}).json()
            assert info["frame_count"] > 0
        assert client.get(f"/api/jobs/{job_id}/video", params={"clip": 2}).status_code == 404

        # Re-scoring covers the whole session.
        rescored = client.post(f"/api/jobs/{job_id}/target", data={"target": "five_hole"}).json()
        assert rescored["result"]["targeting"]["summary"]["shots"] == len(r["shots"])
        assert client.get(f"/api/jobs/{job_id}/chart.svg").status_code == 200
    finally:
        client.delete(f"/api/jobs/{job_id}")


def test_a_clip_left_out_drops_out_of_the_session(client, angled_clip, side_clip):
    job_id, job = _upload(client, angled_clip["path"], side_clip["path"])
    try:
        n_side = len(job["result"]["clips"][1]["shots"])
        after = client.post(f"/api/jobs/{job_id}/skip", data={"clip": 0}).json()
        assert after["clips"][0]["skipped"]
        r = after["result"]
        assert len(r["clips"]) == 1 and r["clips"][0]["upload_index"] == 1
        assert len(r["shots"]) == n_side
    finally:
        client.delete(f"/api/jobs/{job_id}")


def test_marking_one_clip_re_reads_only_that_clip(client, angled_clip, side_clip):
    job_id, job = _upload(client, angled_clip["path"], side_clip["path"])
    try:
        before = job["result"]["clips"][0]["shots"]
        quad = np.asarray(side_clip["net_quad"]).round(1)
        res = client.post(f"/api/jobs/{job_id}/reanalyze",
                          data={"net_quad": ",".join(str(v) for v in quad.ravel()), "clip": 1})
        assert res.status_code == 202
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "done", job.get("error")
        assert job["clips"][1]["net_quad"] is not None and job["clips"][0]["net_quad"] is None
        assert job["result"]["clips"][1]["net"]["method"] == "manual"
        assert job["result"]["clips"][0]["shots"] == before
    finally:
        client.delete(f"/api/jobs/{job_id}")


def test_a_single_clip_still_uploads_as_video(client, angled_clip):
    with open(angled_clip["path"], "rb") as fh:
        res = client.post("/api/analyze", files={"video": ("clip.mp4", fh, "video/mp4")},
                          data={"shot_distance_ft": "20"})
    job_id = res.json()["id"]
    try:
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "done" and len(job["result"]["clips"]) == 1
    finally:
        client.delete(f"/api/jobs/{job_id}")


def test_an_upload_with_no_video_is_refused(client):
    assert client.post("/api/analyze", data={"shot_distance_ft": "20"}).status_code == 400


def test_an_unknown_lens_is_refused_before_anything_is_saved(client, angled_clip):
    import app as server

    before = set(server.DATA_DIR.iterdir()) if server.DATA_DIR.exists() else set()
    with open(angled_clip["path"], "rb") as fh:
        res = client.post("/api/analyze", files=[("videos", ("a.mp4", fh, "video/mp4"))],
                          data={"shot_distance_ft": "20", "lens": "7x"})
    assert res.status_code == 400 and "lens" in res.json()["detail"]
    after = set(server.DATA_DIR.iterdir()) if server.DATA_DIR.exists() else set()
    assert after == before


def test_the_lens_chosen_is_kept_with_the_session(client, angled_clip):
    import app as server

    job_id, job = _upload(client, angled_clip["path"], lens="0.5x")
    try:
        assert job["status"] == "done", job.get("error")
        cfg = server._config_from(server._settings(server.JOBS[job_id].cfg))
        assert cfg.camera.focal_35mm == 14.0
    finally:
        client.delete(f"/api/jobs/{job_id}")


def test_where_the_phone_was_is_kept_with_the_session(client, angled_clip):
    import app as server

    from shottracker.camera import ON_GROUND_HEIGHT_IN

    job_id, job = _upload(client, angled_clip["path"], lens="0.5x", phone="ground")
    try:
        assert job["status"] == "done", job.get("error")
        cfg = server._config_from(server._settings(server.JOBS[job_id].cfg))
        assert cfg.camera.height_in == ON_GROUND_HEIGHT_IN
    finally:
        client.delete(f"/api/jobs/{job_id}")
    with open(angled_clip["path"], "rb") as fh:
        res = client.post("/api/analyze", files=[("videos", ("a.mp4", fh, "video/mp4"))],
                          data={"shot_distance_ft": "20", "phone": "roof"})
    assert res.status_code == 400 and "phone" in res.json()["detail"]


def test_a_video_with_the_shots_drawn_on_can_be_saved(client, angled_clip):
    job_id, job = _upload(client, angled_clip["path"])
    try:
        assert job["status"] == "done", job.get("error")
        assert client.get(f"/api/jobs/{job_id}/marked").json()["status"] == "none"
        assert client.get(f"/api/jobs/{job_id}/marked.mp4").status_code == 404

        res = client.post(f"/api/jobs/{job_id}/marked", data={"clip": "0"})
        assert res.status_code == 202, res.text
        state = client.get(f"/api/jobs/{job_id}/marked").json()
        assert state["status"] == "done", state
        assert "path" not in state   # where it lives on the server is not the page's business

        video = client.get(f"/api/jobs/{job_id}/marked.mp4")
        assert video.status_code == 200
        assert video.headers["content-type"] == "video/mp4"
        assert "angled-shots.mp4" in video.headers["content-disposition"]
        assert len(video.content) > 10_000

        assert client.post(f"/api/jobs/{job_id}/marked", data={"clip": "1"}).status_code == 404
    finally:
        client.delete(f"/api/jobs/{job_id}")
    assert client.get(f"/api/jobs/{job_id}/marked").status_code == 404
