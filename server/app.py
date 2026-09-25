"""HTTP front end: upload a clip, watch it get analyzed, see the shot chart.

Analysis takes tens of seconds, so uploads become jobs that the browser polls
rather than one long request that a phone network would drop.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from shottracker.config import Config
from shottracker.history import progress, rescore, session_record
from shottracker.net_detect import detect_net_in_frame
from shottracker.pipeline import SessionResult, analyze
from shottracker.report import shot_chart_svg
from shottracker.session import ShotSession
from shottracker.targets import TARGET_CHOICES

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DATA_DIR = Path(os.environ.get("SHOTTRACKER_DATA", Path(tempfile.gettempdir()) / "shottracker-data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.environ.get("SHOTTRACKER_MAX_UPLOAD_MB", "400")) * 1024 * 1024
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass
class ClipJob:
    """One uploaded clip within a session."""

    name: str
    video_path: str
    net_quad: list[list[float]] | None = None   # set when the player marked the net
    result: SessionResult | None = None
    skipped: bool = False                        # the player chose to leave it out


@dataclass
class Job:
    id: str
    clips: list[ClipJob]
    cfg: Config | None = None       # shared by every clip: the goal and the target
    session: ShotSession | None = None   # kept so targets can be re-scored without the video
    status: str = "queued"        # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    result: dict[str, Any] | None = None
    chart_svg: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def video_path(self) -> str:
        return self.clips[0].video_path

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "error": self.error,
            "created_at": self.created_at,
            "result": self.result,
            "clips": [
                {"name": c.name, "has_net": bool(c.result and c.result.net is not None),
                 "net_quad": c.net_quad, "skipped": c.skipped}
                for c in self.clips
            ],
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

app = FastAPI(title="Shot Tracker", version="0.1.0")


# --- saved sessions ----------------------------------------------------------
# Each finished session is written next to its clips as session.json, so the
# history survives a restart and an old session can be reopened (and re-scored
# against a new target) without reading the video again.

SESSION_FILE = "session.json"


def _settings(cfg: Config) -> dict[str, Any]:
    return {
        "shot_distance_ft": cfg.speed.shot_distance_ft,
        "shooter_offset_ft": cfg.speed.shooter_offset_ft,
        "assumed_focal_frac": cfg.camera.assumed_focal_frac,
        "hfov_deg": cfg.camera.hfov_deg,
        "fps_override": cfg.fps_override,
        "goal": {"mouth_width_in": cfg.goal.mouth_width_in, "mouth_height_in": cfg.goal.mouth_height_in},
        "target": {"kind": cfg.target.kind, "radius_in": cfg.target.radius_in,
                   "x_in": cfg.target.x_in, "y_in": cfg.target.y_in},
    }


def _config_from(settings: dict[str, Any]) -> Config:
    cfg = Config()
    cfg.speed.shot_distance_ft = settings.get("shot_distance_ft")
    cfg.speed.shooter_offset_ft = settings.get("shooter_offset_ft") or 0.0
    cfg.camera.assumed_focal_frac = settings.get("assumed_focal_frac") or cfg.camera.assumed_focal_frac
    cfg.camera.hfov_deg = settings.get("hfov_deg")
    cfg.fps_override = settings.get("fps_override")
    goal = settings.get("goal") or {}
    cfg.goal.mouth_width_in = goal.get("mouth_width_in") or cfg.goal.mouth_width_in
    cfg.goal.mouth_height_in = goal.get("mouth_height_in") or cfg.goal.mouth_height_in
    target = settings.get("target") or {}
    cfg.target.kind = target.get("kind")
    cfg.target.radius_in = target.get("radius_in") or cfg.target.radius_in
    cfg.target.x_in, cfg.target.y_in = target.get("x_in"), target.get("y_in")
    return cfg


def _session_path(job: Job) -> Path:
    return Path(job.video_path).parent / SESSION_FILE


def _persist(job: Job) -> None:
    path = _session_path(job)
    if job.result is None:
        path.unlink(missing_ok=True)
        return
    doc = {
        "id": job.id,
        "created_at": job.created_at,
        "settings": _settings(job.cfg or Config()),
        "clips": [{"name": c.name, "file": Path(c.video_path).name, "net_quad": c.net_quad, "skipped": c.skipped}
                  for c in job.clips],
        "result": job.result,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)


def _restore_saved_sessions() -> None:
    """Bring back every saved session, as a finished job, when the server starts."""
    for path in sorted(DATA_DIR.glob(f"*/{SESSION_FILE}")):
        try:
            doc = json.loads(path.read_text())
            clips = [ClipJob(name=c["name"], video_path=str(path.parent / c["file"]),
                             net_quad=c.get("net_quad"), skipped=bool(c.get("skipped")))
                     for c in doc["clips"]]
            job = Job(id=doc["id"], clips=clips, cfg=_config_from(doc.get("settings") or {}),
                      status="done", stage="done", progress=1.0, result=doc["result"],
                      created_at=doc["created_at"])
        except (OSError, ValueError, KeyError, TypeError):
            continue  # a damaged file loses that one session, not the history
        with JOBS_LOCK:
            JOBS.setdefault(job.id, job)


def _publish(job: Job) -> None:
    """Rebuild the session from the clips that are in it, and its result."""
    kept = [i for i, c in enumerate(job.clips) if c.result is not None and not c.skipped]
    if not kept:
        job.session, job.result, job.chart_svg = None, None, None
        _persist(job)
        return
    job.session = ShotSession.from_results([(job.clips[i].name, job.clips[i].result) for i in kept], job.cfg)
    job.result = job.session.to_dict()
    # A skipped clip drops out of the session, so each clip carries its place
    # in the upload: that is the number the clip endpoints take.
    for clip_dict, i in zip(job.result["clips"], kept):
        clip_dict["upload_index"] = i
    job.chart_svg = shot_chart_svg(job.session)
    _persist(job)


def _run_job(job_id: str, cfg: Config, only: int | None = None) -> None:
    """Analyze a job's clips -- all of them, or just the one that was re-marked."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return
    job.cfg = cfg
    job.status, job.stage, job.error = "running", "starting", None
    if only is None:
        todo = list(range(len(job.clips)))
    else:
        # A session restored from disk has no clip results in memory, so the
        # rest of it is read again along with the clip that was re-marked.
        todo = sorted({only} | {i for i, c in enumerate(job.clips) if c.result is None and not c.skipped})

    try:
        for n, ci in enumerate(todo):
            clip = job.clips[ci]
            where = f"clip {ci + 1} of {len(job.clips)}: " if len(job.clips) > 1 else ""

            def progress(stage: str, frac: float, n=n, where=where) -> None:
                job.stage, job.progress = where + stage, (n + float(frac)) / len(todo)

            quad = np.asarray(clip.net_quad, dtype=float) if clip.net_quad else None
            clip.result = analyze(clip.video_path, cfg, net_quad=quad, progress=progress)
        _publish(job)
        job.status, job.stage, job.progress = "done", "done", 1.0
    except Exception as exc:  # surfaced to the client rather than swallowed
        job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
        job.stage = "failed"


MAX_CLIPS = int(os.environ.get("SHOTTRACKER_MAX_CLIPS", "30"))


async def _save_upload(upload: UploadFile, dest: Path, budget: int) -> int:
    """Stream one upload to disk, stopping at ``budget`` bytes."""
    size = 0
    with dest.open("wb") as fh:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > budget:
                raise HTTPException(413, f"the clips add up to more than {MAX_UPLOAD_BYTES // (1024*1024)} MB")
            fh.write(chunk)
    return size


@app.post("/api/analyze")
async def create_job(
    background: BackgroundTasks,
    video: UploadFile | None = File(None),
    videos: list[UploadFile] | None = File(None),
    shot_distance_ft: float | None = Form(None),
    shooter_offset_ft: float = Form(0.0),
    hfov_deg: float | None = Form(None),
    fps_override: float | None = Form(None),
    goal_width_in: float | None = Form(None),
    goal_height_in: float | None = Form(None),
    target: str | None = Form(None),
    target_radius_in: float | None = Form(None),
):
    """Start analyzing one clip, or several clips of the same goal as one session."""
    uploads = [u for u in ([video] if video else []) + list(videos or []) if u is not None]
    if not uploads:
        raise HTTPException(400, "no video was uploaded")
    if len(uploads) > MAX_CLIPS:
        raise HTTPException(400, f"at most {MAX_CLIPS} clips per session")
    for u in uploads:
        suffix = Path(u.filename or "clip.mp4").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(400, f"unsupported file type {suffix!r}; use one of {sorted(ALLOWED_SUFFIXES)}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    clips: list[ClipJob] = []
    budget = MAX_UPLOAD_BYTES
    try:
        for i, u in enumerate(uploads):
            suffix = Path(u.filename or "clip.mp4").suffix.lower()
            dest = job_dir / f"clip{i}{suffix}"
            size = await _save_upload(u, dest, budget)
            if size == 0:
                raise HTTPException(400, f"{u.filename or 'a clip'} was empty")
            budget -= size
            clips.append(ClipJob(name=u.filename or f"clip {i + 1}", video_path=str(dest)))
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    cfg = Config()
    cfg.speed.shot_distance_ft = shot_distance_ft
    cfg.speed.shooter_offset_ft = shooter_offset_ft
    if hfov_deg:
        cfg.camera.hfov_deg = hfov_deg
    if fps_override:
        cfg.fps_override = fps_override
    if goal_width_in:
        cfg.goal.mouth_width_in = goal_width_in
    if goal_height_in:
        cfg.goal.mouth_height_in = goal_height_in
    _apply_target(cfg, target, target_radius_in)

    job = Job(id=job_id, clips=clips)
    with JOBS_LOCK:
        JOBS[job_id] = job
    background.add_task(asyncio.to_thread, _run_job, job_id, cfg)
    return JSONResponse({"id": job_id, "clips": len(clips)}, status_code=202)


def _apply_target(cfg: Config, target: str | None, radius: float | None) -> None:
    if target in (None, "", "none"):
        cfg.target.kind = None
    elif target in TARGET_CHOICES and target != "custom":
        cfg.target.kind = target
    else:
        raise HTTPException(400, f"unknown target {target!r}")
    if radius:
        if not (1.0 <= radius <= 36.0):
            raise HTTPException(400, "target radius should be between 1 and 36 inches")
        cfg.target.radius_in = radius


def _job_or_404(job_id: str) -> Job:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


def _clip_or_404(job: Job, clip: int) -> ClipJob:
    if not 0 <= clip < len(job.clips):
        raise HTTPException(404, "no such clip in this session")
    return job.clips[clip]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job_or_404(job_id).public()


@app.get("/api/jobs/{job_id}/chart.svg")
def get_chart(job_id: str):
    job = _job_or_404(job_id)
    if not job.chart_svg:
        raise HTTPException(409, "the analysis has not finished yet")
    return Response(job.chart_svg, media_type="image/svg+xml")


@app.get("/api/jobs/{job_id}/frame.png")
def get_frame(job_id: str, frame: int = 0, clip: int = 0):
    """A single decoded frame, so the browser can show it even for codecs it
    cannot play. Marking the net by hand depends on this."""
    job = _job_or_404(job_id)
    cap = cv2.VideoCapture(_clip_or_404(job, clip).video_path)
    if not cap.isOpened():
        raise HTTPException(404, "the clip could not be opened")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = max(0, min(frame, max(total - 1, 0)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            raise HTTPException(404, "that frame could not be read")
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise HTTPException(500, "the frame could not be encoded")
    finally:
        cap.release()
    return Response(buf.tobytes(), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/jobs/{job_id}/suggest-net")
def suggest_net(job_id: str, frame: int = 0, clip: int = 0):
    """The goal outline in one frame, to pre-fill the marking screen.

    When the camera moved, no single outline fits the whole clip, but the goal
    can usually still be found in the frame the player is looking at.  This is
    a suggestion: the player confirms it or drags the corners.
    """
    job = _job_or_404(job_id)
    cap = cv2.VideoCapture(_clip_or_404(job, clip).video_path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame, max(total - 1, 0))))
        ok, img = cap.read()
    finally:
        cap.release()
    if not ok:
        raise HTTPException(404, "that frame could not be read")
    cfg = job.cfg or Config()
    found = detect_net_in_frame(img, cfg)
    if found is None:
        return {"quad": None}
    quad, conf, method = found
    return {"quad": np.round(quad, 1).tolist(), "confidence": round(float(conf), 3), "method": method}


@app.get("/api/jobs/{job_id}/info")
def get_info(job_id: str, clip: int = 0):
    """Dimensions and length, needed to scale hand-placed corners correctly."""
    job = _job_or_404(job_id)
    cap = cv2.VideoCapture(_clip_or_404(job, clip).video_path)
    try:
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        cap.release()
    return info


@app.post("/api/jobs/{job_id}/reanalyze")
def reanalyze(job_id: str, background: BackgroundTasks, net_quad: str = Form(...), clip: int = Form(0)):
    """Re-run one clip with the goal outline the player marked.

    Four corners of the outer pipe, in source-video pixels, as
    x1,y1,x2,y2,x3,y3,x4,y4 -- top-left, top-right, bottom-right, bottom-left.
    The session's other clips keep their results.
    """
    job = _job_or_404(job_id)
    target = _clip_or_404(job, clip)
    if job.status == "running":
        raise HTTPException(409, "this session is still being analyzed")
    try:
        vals = [float(v) for v in net_quad.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        raise HTTPException(400, "the corners must be numbers")
    if len(vals) != 8:
        raise HTTPException(400, "four corners are needed: x1,y1,x2,y2,x3,y3,x4,y4")

    target.net_quad = [[vals[i], vals[i + 1]] for i in range(0, 8, 2)]
    target.skipped = False
    job.status, job.stage, job.progress, job.result = "queued", "queued", 0.0, None
    background.add_task(asyncio.to_thread, _run_job, job_id, job.cfg or Config(), clip)
    return JSONResponse({"id": job.id}, status_code=202)


@app.post("/api/jobs/{job_id}/skip")
def skip_clip(job_id: str, clip: int = Form(...)):
    """Leave a clip out of the session -- one whose net the player cannot mark."""
    job = _job_or_404(job_id)
    _clip_or_404(job, clip).skipped = True
    if job.status == "done":
        _publish(job)
    return job.public()


@app.post("/api/jobs/{job_id}/target")
def retarget(job_id: str, target: str | None = Form(None), target_radius_in: float | None = Form(None)):
    """Score a finished session against a different target.

    Scoring only needs the impact points, so this is instant -- the player can
    ask "and how would I have done aiming top shelf?" without a re-run.
    """
    job = _job_or_404(job_id)
    if job.result is None or job.cfg is None:
        raise HTTPException(409, "the analysis has not finished yet")
    _apply_target(job.cfg, target, target_radius_in)
    if job.session is not None:
        _publish(job)
    else:
        # Restored from disk: re-score the saved impact points.
        job.result = rescore(job.result, job.cfg.goal, job.cfg.target)
        _persist(job)
    return job.public()


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str, clip: int = 0):
    job = _job_or_404(job_id)
    path = Path(_clip_or_404(job, clip).video_path)
    if not path.exists():
        raise HTTPException(404, "the clip is no longer on disk")
    return FileResponse(path)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = _job_or_404(job_id)
    shutil.rmtree(Path(job.video_path).parent, ignore_errors=True)
    with JOBS_LOCK:
        JOBS.pop(job_id, None)
    return {"deleted": job.id}


@app.get("/api/sessions")
def list_sessions():
    """Every finished session, newest first, and how the latest compares."""
    with JOBS_LOCK:
        done = [j for j in JOBS.values() if j.status == "done" and j.result is not None]
    records = [session_record(j.id, j.created_at, j.result) for j in done]
    records.sort(key=lambda r: r["created_at"], reverse=True)
    return {"sessions": records, "progress": progress(records)}


@app.get("/api/health")
def health():
    with JOBS_LOCK:
        n = len(JOBS)
    return {"ok": True, "jobs": n}


_restore_saved_sessions()

if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:  # pragma: no cover
    @app.get("/", response_class=HTMLResponse)
    def index():
        return "<h1>Shot Tracker</h1><p>The web UI is missing from this install.</p>"
