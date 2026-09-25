"""How well would this shot have been read at normal speed?

Real full-speed footage with known answers is hard to come by.  A slow-motion
clip makes it: keeping every 8th frame of a 240 fps clip is exactly what the
phone would have recorded at 30 fps, and every 4th frame is 60 fps.  Each of
the 8 (or 4) possible starting frames is a different, equally real, full-speed
clip of the same shot, so one slow-motion shot yields a dozen full-speed test
cases, all to be judged against the same slow-motion reading.

The slow-motion reading is the reference, not the truth: this measures how
much is lost by filming at normal speed, which is the question that decides
whether normal-speed video is good enough.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .pipeline import analyze, probe


@dataclass
class Reading:
    rate: float
    phase: int
    shots: int
    mph: float | None
    speed_err_pct: float | None
    mark_err_in: float | None
    outcome: str | None


def decimate(path: str, out_dir: str, rates=(30, 60), cfg: Config | None = None) -> tuple[float, list[tuple]]:
    """Write every full-speed version of a slow-motion clip. Returns (capture fps, [(rate, phase, path)])."""
    capture_fps = probe(path, cfg or Config()).fps
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames could be read from {path}")
    h, w = frames[0].shape[:2]
    out = []
    for rate in rates:
        step = capture_fps / rate
        if step < 2 or abs(step - round(step)) > 1e-6:
            continue  # only whole-frame steps are real recordings
        step = int(round(step))
        for phase in range(step):
            # Lossless, so what is measured is the frame rate and nothing else:
            # a lossy re-encode blurs thin goal pipes and small pucks further
            # than the phone's own encoder does.
            dest = os.path.join(out_dir, f"{rate:g}fps_start{phase}.mkv")
            writer = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*"FFV1"), rate, (w, h))
            for f in frames[phase::step]:
                writer.write(f)
            writer.release()
            out.append((rate, phase, dest))
    return capture_fps, out


def compare(path: str, cfg: Config, rates=(30, 60)) -> dict:
    """Read a slow-motion clip, and every full-speed version of it, and compare."""
    ref = analyze(path, cfg)
    if not ref.shots:
        raise RuntimeError("no shot was found in the slow-motion clip itself, so there is nothing to compare")
    shot = ref.shots[0]
    ref_xy = np.array(shot.impact_goal_in)
    ref_mph = shot.speed.mph if shot.speed else None

    readings = []
    with tempfile.TemporaryDirectory(prefix="shottracker-fullspeed-") as tmp:
        capture_fps, versions = decimate(path, tmp, rates, cfg)
        for rate, phase, p in versions:
            r = analyze(p, cfg)
            if not r.shots:
                readings.append(Reading(rate, phase, 0, None, None, None, None))
                continue
            s = min(r.shots, key=lambda s: float(np.hypot(*(np.array(s.impact_goal_in) - ref_xy))))
            mph = s.speed.mph if s.speed else None
            readings.append(Reading(
                rate, phase, len(r.shots), mph,
                100.0 * (mph / ref_mph - 1.0) if mph and ref_mph else None,
                float(np.hypot(*(np.array(s.impact_goal_in) - ref_xy))), s.outcome,
            ))
    return {"capture_fps": capture_fps, "reference_mph": ref_mph, "reference_xy": ref_xy.tolist(),
            "readings": readings}


def format_table(result: dict) -> str:
    lines = [
        f"Slow-motion reading ({result['capture_fps']:.0f} fps): "
        + (f"{result['reference_mph']:.1f} mph" if result["reference_mph"] else "no speed")
        + f" at ({result['reference_xy'][0]:+.1f}\", {result['reference_xy'][1]:.1f}\" up)",
        "",
        "  rate   start  shots   speed        vs slow-mo   mark vs slow-mo",
    ]
    for r in result["readings"]:
        if r.shots == 0:
            lines.append(f"  {r.rate:3.0f}    {r.phase:<5}  none")
            continue
        sp = f"{r.mph:5.1f} mph" if r.mph else "   -     "
        err = f"{r.speed_err_pct:+6.1f}%" if r.speed_err_pct is not None else "     -"
        lines.append(f"  {r.rate:3.0f}    {r.phase:<5}  {r.shots:<5}  {sp}   {err}      {r.mark_err_in:5.1f} in")
    lines.append("")
    for rate in sorted({r.rate for r in result["readings"]}):
        rs = [r for r in result["readings"] if r.rate == rate and r.speed_err_pct is not None]
        if rs:
            e = np.abs([r.speed_err_pct for r in rs])
            m = [r.mark_err_in for r in rs]
            lines.append(f"  at {rate:.0f} fps: speed within {np.median(e):.1f}% typically, {e.max():.1f}% at worst; "
                         f"marks within {np.median(m):.1f} in typically, {max(m):.1f} in at worst "
                         f"({len(rs)} of {sum(1 for r in result['readings'] if r.rate == rate)} versions read)")
    return "\n".join(lines)
