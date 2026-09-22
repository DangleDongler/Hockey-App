"""HTTP front end: upload a clip, watch it get analyzed, see the shot chart.

Analysis takes tens of seconds, so uploads become jobs that the browser polls
rather than one long request that a phone network would drop.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from shottracker.config import CameraConfig, Config
from shottracker.pipeline import analyze
from shottracker.report import shot_chart_svg

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DATA_DIR = Path(os.environ.get("SHOTTRACKER_DATA", Path(tempfile.gettempdir()) / "shottracker-data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = int(os.environ.get("SHOTTRACKER_MAX_UPLOAD_MB", "400")) * 1024 * 1024
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


@dataclass
class Job:
    id: str
    video_path: str
    status: str = "queued"        # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    result: dict[str, Any] | None = None
    chart_svg: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "error": self.error,
            "created_at": self.created_at,
            "result": self.result,
        }


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

app = FastAPI(title="Shot Tracker", version="0.1.0")


def _run_job(job_id: str, cfg: Config) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return
    job.status, job.stage = "running", "starting"

    def progress(stage: str, frac: float) -> None:
        job.stage, job.progress = stage, float(frac)

    try:
        result = analyze(job.video_path, cfg, progress=progress)
        job.result = result.to_dict()
        job.chart_svg = shot_chart_svg(result)
        job.status, job.stage, job.progress = "done", "done", 1.0
    except Exception as exc:  # surfaced to the client rather than swallowed
        job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
        job.stage = "failed"


@app.post("/api/analyze")
async def create_job(
    background: BackgroundTasks,
    video: UploadFile = File(...),
    shot_distance_ft: float | None = Form(None),
    shooter_offset_ft: float = Form(0.0),
    hfov_deg: float | None = Form(None),
    fps_override: float | None = Form(None),
):
    suffix = Path(video.filename or "clip.mp4").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported file type {suffix!r}; use one of {sorted(ALLOWED_SUFFIXES)}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / f"source{suffix}"

    size = 0
    with dest.open("wb") as fh:
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                fh.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, f"clip is larger than {MAX_UPLOAD_BYTES // (1024*1024)} MB")
            fh.write(chunk)
    if size == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, "the uploaded file was empty")

    cfg = Config()
    cfg.speed.shot_distance_ft = shot_distance_ft
    cfg.speed.shooter_offset_ft = shooter_offset_ft
    if hfov_deg:
        cfg.camera.assumed_focal_frac = CameraConfig.frac_from_hfov(hfov_deg)
    if fps_override:
        cfg.fps_override = fps_override

    job = Job(id=job_id, video_path=str(dest))
    with JOBS_LOCK:
        JOBS[job_id] = job
    background.add_task(asyncio.to_thread, _run_job, job_id, cfg)
    return JSONResponse({"id": job_id}, status_code=202)


def _job_or_404(job_id: str) -> Job:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job_or_404(job_id).public()


@app.get("/api/jobs/{job_id}/chart.svg")
def get_chart(job_id: str):
    job = _job_or_404(job_id)
    if not job.chart_svg:
        raise HTTPException(409, "the analysis has not finished yet")
    return Response(job.chart_svg, media_type="image/svg+xml")


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str):
    job = _job_or_404(job_id)
    path = Path(job.video_path)
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


@app.get("/api/health")
def health():
    with JOBS_LOCK:
        n = len(JOBS)
    return {"ok": True, "jobs": n}


if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:  # pragma: no cover
    @app.get("/", response_class=HTMLResponse)
    def index():
        return "<h1>Shot Tracker</h1><p>The web UI is missing from this install.</p>"
