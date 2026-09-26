"""HTTP front end: upload a clip, watch it get analyzed, see the shot chart.

Analysis takes tens of seconds, so uploads become jobs that the browser polls
rather than one long request that a phone network would drop.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import mimetypes
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from shottracker.camera import ON_GROUND_HEIGHT_IN
from shottracker.config import CameraConfig, Config
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

# Run on the internet rather than at home, the app is behind a password, and
# the clips themselves are not kept: each is deleted this many hours after it
# was analysed, and all of them whenever the server shuts down (it sleeps when
# nobody is using it).  The results -- speeds, marks, history -- stay.  Unset,
# as at home, nothing asks for a password and nothing is deleted.
PASSWORD = os.environ.get("SHOTTRACKER_PASSWORD") or None
KEEP_CLIPS_HOURS = (float(os.environ["SHOTTRACKER_KEEP_CLIPS_HOURS"])
                    if os.environ.get("SHOTTRACKER_KEEP_CLIPS_HOURS") else None)
CLEAN_EVERY_S = 600


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

def _busy_dirs() -> set[Path]:
    """Folders whose clips are in use: being analysed, or being drawn on."""
    with JOBS_LOCK:
        busy = {Path(j.video_path).parent for j in JOBS.values() if j.status in ("queued", "running")}
        by_id = dict(JOBS)
    with MARKED_LOCK:
        drawing = [k[0] for k, v in MARKED.items() if v.get("status") == "running"]
    busy |= {Path(by_id[j].video_path).parent for j in drawing if j in by_id}
    return busy


def delete_old_clips(older_than_s: float) -> int:
    """Delete clips analysed more than ``older_than_s`` ago, and videos drawn from them.

    A clip counts from when its session was last written (the analysis
    finishing) or, for a session that never finished, from its upload.
    Nothing in use is touched.  Returns how many files went.
    """
    cutoff = time.time() - older_than_s
    busy = _busy_dirs()
    gone = 0
    for folder in DATA_DIR.iterdir() if DATA_DIR.is_dir() else []:
        if not folder.is_dir() or folder in busy:
            continue
        saved = folder / SESSION_FILE
        for f in folder.iterdir():
            if f.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            when = max(f.stat().st_mtime, saved.stat().st_mtime if saved.exists() else 0.0)
            if when <= cutoff:
                f.unlink(missing_ok=True)
                gone += 1
    return gone


def _keep_cleaning(stop: threading.Event) -> None:
    while not stop.wait(CLEAN_EVERY_S):
        delete_old_clips(KEEP_CLIPS_HOURS * 3600)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    stop = threading.Event()
    if KEEP_CLIPS_HOURS is not None:
        delete_old_clips(KEEP_CLIPS_HOURS * 3600)
        threading.Thread(target=_keep_cleaning, args=(stop,), daemon=True).start()
    yield
    stop.set()
    if KEEP_CLIPS_HOURS is not None:
        # Going to sleep: nobody is watching, so no clip is kept past it.
        delete_old_clips(0.0)


app = FastAPI(title="Shot Tracker", version="0.1.0", lifespan=_lifespan)


# --- the password ----------------------------------------------------------

LOGIN_COOKIE = "shottracker"
# Reachable without logging in: the login page itself, and what the phone
# fetches to put the app on its home screen.
OPEN_PATHS = {"/login", "/login.html", "/style.css", "/manifest.webmanifest", "/api/health"}


def _login_token() -> str:
    # Changing the password signs every phone out.
    return hmac.new(PASSWORD.encode(), b"shottracker-login-v1", hashlib.sha256).hexdigest()


@app.middleware("http")
async def _require_login(request: Request, call_next):
    path = request.url.path
    if PASSWORD is None or path in OPEN_PATHS or path.startswith("/icons/"):
        return await call_next(request)
    if hmac.compare_digest(request.cookies.get(LOGIN_COOKIE, ""), _login_token()):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse({"detail": "log in first"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


@app.get("/login")
def login_page():
    return FileResponse(WEB_DIR / "login.html", media_type="text/html")


_WRONG_GUESSES = asyncio.Lock()


@app.post("/login")
async def log_in(request: Request, password: str = Form("")):
    if PASSWORD is not None and hmac.compare_digest(password.encode(), PASSWORD.encode()):
        res = RedirectResponse("/", status_code=303)
        res.set_cookie(LOGIN_COOKIE, _login_token(), max_age=365 * 24 * 3600, httponly=True,
                       samesite="lax", secure=request.url.scheme == "https")
        return res
    # A wrong guess costs a second, one at a time: guessing in parallel
    # gains nothing.
    async with _WRONG_GUESSES:
        await asyncio.sleep(1.0)
    return RedirectResponse("/login?wrong=1", status_code=303)


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
        "focal_35mm": cfg.camera.focal_35mm,
        "camera_height_in": cfg.camera.height_in,
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
    cfg.camera.focal_35mm = settings.get("focal_35mm")
    cfg.camera.height_in = settings.get("camera_height_in")
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
    _forget_marked(job.id)   # drawn from the old results
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
# Where the player says the phone was: its lens's height, when that helps.
PHONE_PLACES = {"ground": ON_GROUND_HEIGHT_IN, "raised": None}


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
    lens: str | None = Form(None),
    phone: str | None = Form(None),
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

    if lens and lens not in CameraConfig.LENS_CHOICES:
        raise HTTPException(400, f"unknown lens {lens!r}; use one of {sorted(CameraConfig.LENS_CHOICES)}")
    if phone and phone not in PHONE_PLACES:
        raise HTTPException(400, f"unknown phone placement {phone!r}; use one of {sorted(PHONE_PLACES)}")

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
    if lens:
        cfg.camera.focal_35mm = CameraConfig.LENS_CHOICES[lens]
    if phone:
        cfg.camera.height_in = PHONE_PLACES[phone]
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


def _clip_file(job: Job, clip: int) -> str:
    """The clip's video on disk, or 410 when it was deleted after analysis."""
    path = _clip_or_404(job, clip).video_path
    if not Path(path).exists():
        raise HTTPException(410, "the video was deleted from the server after analysis, to keep it private; "
                                 "the shots, speeds and marks are kept")
    return path


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
    cap = cv2.VideoCapture(_clip_file(job, clip))
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
    cap = cv2.VideoCapture(_clip_file(job, clip))
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
    cap = cv2.VideoCapture(_clip_file(job, clip))
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
    # Every clip that will be read again must still be there.
    for i, c in enumerate(job.clips):
        if i == clip or (c.result is None and not c.skipped):
            _clip_file(job, i)
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
    return FileResponse(_clip_file(job, clip))


# Videos with the shots drawn on, keyed by (job, clip); rendered on request.
MARKED: dict[tuple[str, int], dict[str, Any]] = {}
MARKED_LOCK = threading.Lock()


def _forget_marked(job_id: str) -> None:
    with MARKED_LOCK:
        for key in [k for k in MARKED if k[0] == job_id]:
            MARKED.pop(key)


def _marked_state(job_id: str, clip: int) -> dict[str, Any]:
    with MARKED_LOCK:
        st = dict(MARKED.get((job_id, clip)) or {"status": "none"})
    st.pop("path", None)
    return st


def _render_marked(key: tuple[str, int], video_path: str, clip_dict: dict, out: str,
                   numbers: dict[int, int]) -> None:
    from shottracker.annotate import render

    def progress(f: float) -> None:
        with MARKED_LOCK:
            if key in MARKED:
                MARKED[key]["progress"] = round(f, 3)

    try:
        render(video_path, clip_dict, out, numbers, progress=progress)
        state = {"status": "done", "progress": 1.0, "path": out}
    except Exception as exc:  # noqa: BLE001 - reported to the page, which offers to try again
        state = {"status": "error", "error": f"could not draw the video: {exc}"}
    with MARKED_LOCK:
        if key in MARKED:   # not forgotten meanwhile (session deleted or re-read)
            MARKED[key] = state


@app.post("/api/jobs/{job_id}/marked")
def make_marked(job_id: str, background: BackgroundTasks, clip: int = Form(0)):
    """Start drawing a clip's shots onto a copy of it (see shottracker.annotate)."""
    job = _job_or_404(job_id)
    clips = (job.result or {}).get("clips") or []
    if not 0 <= clip < len(clips):
        raise HTTPException(404, "no such clip in this session")
    clip_dict = clips[clip]
    upload = _clip_or_404(job, clip_dict.get("upload_index", clip))
    _clip_file(job, clip_dict.get("upload_index", clip))
    key = (job_id, clip)
    with MARKED_LOCK:
        if MARKED.get(key, {}).get("status") == "running":
            return JSONResponse(_marked_state(job_id, clip), status_code=202)
        MARKED[key] = {"status": "running", "progress": 0.0}
    # Numbered as the session numbers them, across all its clips.
    numbers = {s["clip_shot"]: s["index"] + 1 for s in job.result.get("shots", []) if s.get("clip", 0) == clip}
    out = str(Path(upload.video_path).with_name(f"{Path(upload.video_path).stem}.marked.mp4"))
    background.add_task(_render_marked, key, upload.video_path, clip_dict, out, numbers)
    return JSONResponse(_marked_state(job_id, clip), status_code=202)


@app.get("/api/jobs/{job_id}/marked")
def marked_status(job_id: str, clip: int = 0):
    _job_or_404(job_id)
    return _marked_state(job_id, clip)


@app.get("/api/jobs/{job_id}/marked.mp4")
def marked_video(job_id: str, clip: int = 0):
    job = _job_or_404(job_id)
    with MARKED_LOCK:
        state = MARKED.get((job_id, clip)) or {}
    path = state.get("path")
    if state.get("status") != "done" or not path or not Path(path).exists():
        raise HTTPException(404, "the video with the shots drawn on has not been made yet")
    clips = (job.result or {}).get("clips") or []
    name = Path(clips[clip].get("name", "clip") if clip < len(clips) else "clip").stem
    return FileResponse(path, media_type="video/mp4", filename=f"{name}-shots.mp4")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = _job_or_404(job_id)
    shutil.rmtree(Path(job.video_path).parent, ignore_errors=True)
    with JOBS_LOCK:
        JOBS.pop(job_id, None)
    _forget_marked(job_id)
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
    if PASSWORD is not None:   # open to all, so it says nothing more
        return {"ok": True}
    with JOBS_LOCK:
        n = len(JOBS)
    return {"ok": True, "jobs": n}


_restore_saved_sessions()

# What a phone reads to put the app on its home screen.
mimetypes.add_type("application/manifest+json", ".webmanifest")

if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:  # pragma: no cover
    @app.get("/", response_class=HTMLResponse)
    def index():
        return "<h1>Shot Tracker</h1><p>The web UI is missing from this install.</p>"
