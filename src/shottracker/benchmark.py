"""Grade the tracker against clips whose answers are known exactly.

Every accuracy figure quoted in the README comes from running this.  It is
deliberately part of the package rather than a loose script, so the claims stay
reproducible as the code changes:

    shottracker benchmark
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

from .config import Config
from .pipeline import analyze
from .synth import SynthShot, render_session

# Top corner, bottom corner, five hole, and one that misses wide right.
BENCH_SHOTS = [
    dict(target_x_in=-30, target_y_in=42, flight_time_s=0.20, start_time_s=0.30),
    dict(target_x_in=30, target_y_in=6, flight_time_s=0.26, start_time_s=0.95),
    dict(target_x_in=0, target_y_in=8, flight_time_s=0.30, start_time_s=1.60),
    dict(target_x_in=48, target_y_in=30, flight_time_s=0.24, start_time_s=2.25),
]


@dataclass
class CameraScore:
    camera: str
    net_error_pct: float          # worst corner error, as % of goal width
    shots_found: int
    shots_expected: int
    on_net_error_in: float        # mean impact error for shots on the net
    off_net_error_in: float       # mean impact error for shots that missed
    zone_correct: int
    speed_error_pct: float        # mean absolute error
    speed_worst_pct: float
    within_error_bars: int
    lens_assumed: bool


def run_camera(camera: str, workdir: str, fps: float = 120.0, distance_ft: float = 20.0) -> CameraScore:
    path = os.path.join(workdir, f"bench-{camera}.mp4")
    truth = render_session(path, [SynthShot(**s) for s in BENCH_SHOTS], camera=camera, fps=fps)

    cfg = Config()
    cfg.speed.shot_distance_ft = distance_ft
    result = analyze(path, cfg)

    net_err = float("nan")
    if result.net is not None:
        truth_quad = np.asarray(truth["net_quad"])
        width = np.linalg.norm(truth_quad[1] - truth_quad[0])
        net_err = 100.0 * np.linalg.norm(result.net.quad - truth_quad, axis=1).max() / width

    from .geometry import build_zones, zone_for

    zones = build_zones(cfg.goal)
    on_net_errs, off_net_errs, speed_errs = [], [], []
    zone_ok = 0
    covered = 0

    measured = sorted(result.shots, key=lambda s: s.impact_frame)
    for shot, t in zip(measured, truth["shots"]):
        err = float(np.hypot(shot.impact_goal_in[0] - t["impact_x_in"],
                             shot.impact_goal_in[1] - t["impact_y_in"]))
        (on_net_errs if shot.outcome == "on_net" else off_net_errs).append(err)

        expected = zone_for(zones, t["impact_x_in"], t["impact_y_in"])
        if expected is None:
            zone_ok += shot.outcome != "on_net"
        else:
            zone_ok += shot.zone_key == expected.key

        if shot.speed:
            rel = 100.0 * abs(shot.speed.mph - t["release_speed_mph"]) / t["release_speed_mph"]
            speed_errs.append(rel)
            if shot.speed.uncertainty_mph is not None:
                covered += abs(shot.speed.mph - t["release_speed_mph"]) <= shot.speed.uncertainty_mph

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return CameraScore(
        camera=camera,
        net_error_pct=net_err,
        shots_found=len(result.shots),
        shots_expected=len(truth["shots"]),
        on_net_error_in=mean(on_net_errs),
        off_net_error_in=mean(off_net_errs),
        zone_correct=zone_ok,
        speed_error_pct=mean(speed_errs),
        speed_worst_pct=max(speed_errs) if speed_errs else float("nan"),
        within_error_bars=covered,
        lens_assumed=bool(result.camera and result.camera.focal_assumed),
    )


def run(cameras: list[str] | None = None, workdir: str | None = None) -> list[CameraScore]:
    cameras = cameras or ["side", "angled", "head_on"]
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        return [run_camera(c, workdir) for c in cameras]
    with tempfile.TemporaryDirectory(prefix="shottracker-bench-") as d:
        return [run_camera(c, d) for c in cameras]


def format_table(scores: list[CameraScore]) -> str:
    head = (
        f"{'camera':<10} {'net':>7} {'shots':>7} {'on-net':>8} {'off-net':>8} "
        f"{'zones':>7} {'speed':>8} {'worst':>7} {'in bars':>8}"
    )
    units = (
        f"{'':<10} {'% width':>7} {'found':>7} {'inches':>8} {'inches':>8} "
        f"{'right':>7} {'mean %':>8} {'%':>7} {'':>8}"
    )
    lines = [head, units, "-" * len(head)]
    for s in scores:
        lines.append(
            f"{s.camera:<10} {s.net_error_pct:>7.2f} {f'{s.shots_found}/{s.shots_expected}':>7} "
            f"{s.on_net_error_in:>8.1f} {s.off_net_error_in:>8.1f} "
            f"{f'{s.zone_correct}/{s.shots_expected}':>7} {s.speed_error_pct:>8.1f} "
            f"{s.speed_worst_pct:>7.1f} {f'{s.within_error_bars}/{s.shots_expected}':>8}"
            + ("   (lens assumed)" if s.lens_assumed else "")
        )
    return "\n".join(lines)
