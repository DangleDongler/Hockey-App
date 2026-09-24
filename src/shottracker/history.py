"""Sessions over time: is the shot getting harder, and going where it is aimed?

The history works from saved session results, not from video, so it costs
nothing to show and survives the clips being deleted.  Each session becomes a
one-line record; progress compares the latest session with the ones before it.

A backyard session is often a handful of shots, so one session's average moves
around a lot on its own.  Progress is therefore stated against the average of
several earlier sessions, never against the single previous one, and the
record keeps each session's shot count so a reader can weigh it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import GoalSpec, TargetConfig
from .geometry import GoalPlane
from .targets import targeting_block

# How many earlier sessions the latest one is compared with.
BASELINE_SESSIONS = 5

METRICS = {
    # key: (label, unit, whether up is good)
    "speed_mean": ("Average speed", "mph", True),
    "accuracy_pct": ("On net", "%", True),
    "hit_pct": ("On target", "%", True),
}


def session_record(session_id: str, created_at: str, result: dict[str, Any]) -> dict[str, Any]:
    """One line of history from a saved session result."""
    s = result.get("summary") or {}
    speed = s.get("speed_mph") or {}
    targeting = result.get("targeting") or None
    ts = (targeting or {}).get("summary") or {}
    shots = int(s.get("shots") or 0)
    record: dict[str, Any] = {
        "id": session_id,
        "created_at": created_at,
        "clips": len(result.get("clips") or [None]),
        "shots": shots,
        "on_net": s.get("on_net"),
        "accuracy_pct": s.get("accuracy_pct"),
        "speed_mean": speed.get("mean"),
        "speed_max": speed.get("max"),
        "spread_in": (s.get("grouping") or {}).get("spread_in"),
        "target_label": (targeting or {}).get("label"),
        "target_hits": ts.get("hits"),
        "target_shots": ts.get("shots"),
        "hit_pct": None,
    }
    if targeting and ts.get("shots"):
        record["hit_pct"] = round(100.0 * ts["hits"] / ts["shots"], 1)
    return record


def progress(records: list[dict[str, Any]], baseline: int = BASELINE_SESSIONS) -> dict[str, Any]:
    """The latest session against the average of the few before it, per metric.

    ``records`` may come in any order; sessions without a value for a metric
    (no shots, no target) are simply not part of that metric's story.
    """
    ordered = sorted(records, key=lambda r: r["created_at"])
    out: dict[str, Any] = {}
    for key, (label, unit, up_is_good) in METRICS.items():
        series = [
            {"id": r["id"], "created_at": r["created_at"], "value": r[key], "shots": r["shots"]}
            for r in ordered
            if r.get(key) is not None and r.get("shots")
        ]
        if not series:
            continue
        latest = series[-1]
        before = series[-1 - baseline:-1]
        entry: dict[str, Any] = {
            "label": label,
            "unit": unit,
            "up_is_good": up_is_good,
            "latest": latest["value"],
            "series": series,
            "baseline": None,
            "baseline_sessions": len(before),
            "delta": None,
        }
        if before:
            base = float(np.mean([b["value"] for b in before]))
            entry["baseline"] = round(base, 1)
            entry["delta"] = round(float(latest["value"]) - base, 1)
        out[key] = entry
    return out


def rescore(result: dict[str, Any], goal: GoalSpec, target: TargetConfig) -> dict[str, Any]:
    """Score a saved session against a different target, without the video.

    Only the impact points matter, and they are in the saved result: the
    session's pooled shots for the numbers, and each clip's own shots and goal
    outline for the rings drawn over its footage.
    """
    shots = result.get("shots") or []
    block = targeting_block(
        [(s["impact_goal_in"][0], s["impact_goal_in"][1], s["outcome"]) for s in shots], goal, target
    )
    result["targeting"] = block
    for i, s in enumerate(shots):
        if block:
            s["vs_target"] = block["per_shot"][i]
        else:
            s.pop("vs_target", None)
    for clip in result.get("clips") or []:
        quad = (clip.get("net") or {}).get("quad")
        plane = GoalPlane(np.asarray(quad, dtype=float), goal) if quad else None
        cshots = clip.get("shots") or []
        cblock = targeting_block(
            [(s["impact_goal_in"][0], s["impact_goal_in"][1], s["outcome"]) for s in cshots], goal, target, plane
        )
        clip["targeting"] = cblock
        for i, s in enumerate(cshots):
            if cblock:
                s["vs_target"] = cblock["per_shot"][i]
            else:
                s.pop("vs_target", None)
    return result
