"""Shared synthetic clips.

Rendering video is the slow part of these tests, so each clip is rendered once
per session and reused.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from shottracker.config import Config
from shottracker.pipeline import analyze
from shottracker.synth import SynthShot, render_session

# Four shots: top corner, bottom corner, five hole, and one that misses wide.
STANDARD_SHOTS = [
    dict(target_x_in=-30, target_y_in=42, flight_time_s=0.20, start_time_s=0.30),
    dict(target_x_in=30, target_y_in=6, flight_time_s=0.26, start_time_s=0.95),
    dict(target_x_in=0, target_y_in=8, flight_time_s=0.30, start_time_s=1.60),
    dict(target_x_in=48, target_y_in=30, flight_time_s=0.24, start_time_s=2.25),
]


@pytest.fixture(scope="session")
def workdir():
    with tempfile.TemporaryDirectory(prefix="shottracker-tests-") as d:
        yield d


def _render(workdir: str, name: str, camera: str, fps: float = 120.0, **kw):
    path = os.path.join(workdir, f"{name}.mp4")
    shots = [SynthShot(**s) for s in STANDARD_SHOTS]
    truth = render_session(path, shots, camera=camera, fps=fps, **kw)
    return truth


@pytest.fixture(scope="session")
def angled_clip(workdir):
    return _render(workdir, "angled", "angled")


@pytest.fixture(scope="session")
def side_clip(workdir):
    return _render(workdir, "side", "side")


@pytest.fixture(scope="session")
def head_on_clip(workdir):
    return _render(workdir, "head_on", "head_on")


def _analyze_truth(truth, *, distance_ft: float | None = 20.0):
    cfg = Config()
    cfg.speed.shot_distance_ft = distance_ft
    return analyze(truth["path"], cfg)


@pytest.fixture(scope="session")
def analyze_truth():
    """Analyze a rendered clip the way a player would: distance supplied."""
    return _analyze_truth


@pytest.fixture(scope="session")
def angled_result(angled_clip):
    return angled_clip, _analyze_truth(angled_clip)
