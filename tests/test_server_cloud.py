"""The web app run on the internet: behind a password, keeping no clips."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
WEB = Path(__file__).resolve().parent.parent / "web"


@pytest.fixture()
def server(monkeypatch):
    import app as server

    monkeypatch.setattr(server, "PASSWORD", "puck-and-net")
    return server


def _client(server):
    from fastapi.testclient import TestClient

    return TestClient(server.app, follow_redirects=False)


def test_with_a_password_nothing_opens_without_logging_in(server):
    c = _client(server)
    assert c.get("/").status_code == 303 and c.get("/").headers["location"] == "/login"
    assert c.get("/app.js").status_code == 303
    assert c.get("/api/sessions").status_code == 401
    # What the phone needs to log in and to put the app on its home screen.
    assert c.get("/login").status_code == 200
    assert c.get("/manifest.webmanifest").status_code == 200
    assert c.get("/icons/apple-touch-icon.png").status_code == 200
    assert c.get("/api/health").json() == {"ok": True}

    t = time.time()
    wrong = c.post("/login", data={"password": "slapshot"})
    assert wrong.headers["location"] == "/login?wrong=1" and time.time() - t >= 1.0
    assert c.get("/api/sessions").status_code == 401

    right = c.post("/login", data={"password": "puck-and-net"})
    assert right.status_code == 303 and right.headers["location"] == "/"
    cookie = right.headers["set-cookie"]
    assert "HttpOnly" in cookie and "samesite=lax" in cookie.lower()
    assert c.get("/api/sessions").status_code == 200
    assert c.get("/").status_code == 200


def test_changing_the_password_signs_everyone_out(server, monkeypatch):
    c = _client(server)
    c.post("/login", data={"password": "puck-and-net"})
    assert c.get("/api/sessions").status_code == 200
    monkeypatch.setattr(server, "PASSWORD", "a-new-one-now")
    assert c.get("/api/sessions").status_code == 401


def test_at_home_there_is_no_password(monkeypatch):
    import app as server

    monkeypatch.setattr(server, "PASSWORD", None)
    assert _client(server).get("/api/sessions").status_code == 200


def _session_folder(root: Path, name: str, age_h: float, clip: bool = True) -> Path:
    d = root / name
    d.mkdir()
    (d / "session.json").write_text("{}")
    files = [d / "session.json"]
    if clip:
        (d / "clip0.mov").write_bytes(b"video")
        (d / "clip0.marked.mp4").write_bytes(b"video")
        files += [d / "clip0.mov", d / "clip0.marked.mp4"]
    when = time.time() - age_h * 3600
    for f in files:
        os.utime(f, (when, when))
    return d


def test_clips_are_deleted_after_the_hours_given_and_results_kept(tmp_path, monkeypatch):
    import app as server

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    old = _session_folder(tmp_path, "old", age_h=3)
    new = _session_folder(tmp_path, "new", age_h=0.5)
    assert server.delete_old_clips(2 * 3600) == 2
    assert sorted(p.name for p in old.iterdir()) == ["session.json"]
    assert sorted(p.name for p in new.iterdir()) == ["clip0.marked.mp4", "clip0.mov", "session.json"]
    # Going to sleep takes the rest.
    assert server.delete_old_clips(0.0) == 2
    assert sorted(p.name for p in new.iterdir()) == ["session.json"]


def test_a_clip_being_analysed_is_never_deleted(tmp_path, monkeypatch):
    import app as server

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    busy = _session_folder(tmp_path, "busy", age_h=5)
    job = server.Job(id="busy", clips=[server.ClipJob(name="c", video_path=str(busy / "clip0.mov"))],
                     status="running")
    monkeypatch.setitem(server.JOBS, "busy", job)
    assert server.delete_old_clips(0.0) == 0
    assert (busy / "clip0.mov").exists()


def test_a_deleted_clip_says_so_and_the_results_stay(angled_clip, monkeypatch):
    import app as server

    monkeypatch.setattr(server, "PASSWORD", None)
    c = _client(server)
    with open(angled_clip["path"], "rb") as fh:
        res = c.post("/api/analyze", files=[("videos", ("a.mp4", fh, "video/mp4"))],
                     data={"shot_distance_ft": "20"})
    job_id = res.json()["id"]
    try:
        job = c.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "done"
        Path(server.JOBS[job_id].clips[0].video_path).unlink()
        for url in (f"/api/jobs/{job_id}/video", f"/api/jobs/{job_id}/info", f"/api/jobs/{job_id}/frame.png"):
            r = c.get(url)
            assert r.status_code == 410 and "deleted from the server" in r.json()["detail"], url
        assert c.post(f"/api/jobs/{job_id}/marked", data={"clip": "0"}).status_code == 410
        assert c.get(f"/api/jobs/{job_id}").json()["result"]["shots"]
    finally:
        c.delete(f"/api/jobs/{job_id}")


def test_the_app_can_be_put_on_the_home_screen():
    manifest = json.loads((WEB / "manifest.webmanifest").read_text())
    assert manifest["display"] == "standalone" and manifest["start_url"] == "/"
    for icon in manifest["icons"]:
        assert (WEB / icon["src"]).exists()
    for page in ("index.html", "login.html"):
        html = (WEB / page).read_text()
        assert 'rel="apple-touch-icon" href="icons/apple-touch-icon.png"' in html
        assert 'rel="manifest"' in html
    assert (WEB / "icons" / "apple-touch-icon.png").exists()
