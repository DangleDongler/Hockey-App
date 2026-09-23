"""Aiming at something.

Knowing where a shot went is half of it; the other half is where it was meant
to go.  A player picks a target -- a top corner, the five hole, or "any
corner" -- and each shot is scored against it: did it land inside the target,
and if not, how far off and in which direction.

Direction is the part worth having.  Scatter is noise, but a *consistent*
offset -- every shot landing four inches under where it was aimed -- is a habit,
and it is the one thing a shooter can actually fix between sessions.  So the
bias is only reported when it stands clear of the scatter; a player told they
miss low on the strength of two shots would be chasing noise.

Everything here works on impact points in goal inches, so a session can be
re-scored against a different target without touching the video again.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import GoalSpec, TargetConfig
from .geometry import point_in_mouth


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    x_in: float        # aim point on the goal plane
    y_in: float
    radius_in: float

    def to_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(self).items()}


# Named targets, and the ones a player means when they say "any corner".
_SINGLE = {
    "top_left": "Top Shelf Left",
    "top_right": "Top Shelf Right",
    "bottom_left": "Bottom Left Corner",
    "bottom_right": "Bottom Right Corner",
    "five_hole": "Five Hole",
}
_GROUPS = {
    "top_shelf": ("Top Shelf", ["top_left", "top_right"]),
    "any_corner": ("Any Corner", ["top_left", "top_right", "bottom_left", "bottom_right"]),
    "low_corners": ("Low Corners", ["bottom_left", "bottom_right"]),
}

TARGET_CHOICES = sorted([*_SINGLE, *_GROUPS, "custom"])


def _aim_point(key: str, goal: GoalSpec, inset: float) -> tuple[float, float]:
    """Where to aim for a named target, set in from the pipe by ``inset``.

    Tied to the goal's real dimensions so a smaller training net gets
    proportionate targets rather than ones hanging off its posts.
    """
    hw = goal.mouth_width_in / 2.0
    top = goal.mouth_height_in
    x_edge = hw - inset
    return {
        "top_left": (-x_edge, top - inset),
        "top_right": (x_edge, top - inset),
        "bottom_left": (-x_edge, inset),
        "bottom_right": (x_edge, inset),
        "five_hole": (0.0, inset),
    }[key]


def resolve_targets(cfg: TargetConfig, goal: GoalSpec) -> list[Target]:
    """Turn a target setting into the concrete aim points it stands for."""
    if not cfg.kind:
        return []
    r = float(cfg.radius_in)
    if cfg.kind == "custom":
        if cfg.x_in is None or cfg.y_in is None:
            raise ValueError("a custom target needs an aim point: x and y in inches")
        return [Target("custom", "Custom", float(cfg.x_in), float(cfg.y_in), r)]
    if cfg.kind in _SINGLE:
        x, y = _aim_point(cfg.kind, goal, cfg.inset_in)
        return [Target(cfg.kind, _SINGLE[cfg.kind], x, y, r)]
    if cfg.kind in _GROUPS:
        _, keys = _GROUPS[cfg.kind]
        return [Target(k, _SINGLE[k], *_aim_point(k, goal, cfg.inset_in), r) for k in keys]
    raise ValueError(f"unknown target {cfg.kind!r}; choose from {TARGET_CHOICES}")


def target_label(cfg: TargetConfig) -> str | None:
    if not cfg.kind:
        return None
    if cfg.kind in _GROUPS:
        return _GROUPS[cfg.kind][0]
    return _SINGLE.get(cfg.kind, "Custom")


@dataclass
class ShotScore:
    target_key: str
    target_label: str
    hit: bool
    distance_in: float      # from the aim point
    dx_in: float            # + is right of the aim point, as the chart shows it
    dy_in: float            # + is above it

    def to_dict(self) -> dict:
        return {
            "target_key": self.target_key,
            "target_label": self.target_label,
            "hit": self.hit,
            "distance_in": round(self.distance_in, 1),
            "dx_in": round(self.dx_in, 1),
            "dy_in": round(self.dy_in, 1),
        }


def score_shot(
    x: float, y: float, outcome: str, targets: list[Target], goal: GoalSpec
) -> ShotScore | None:
    """Score one impact against the nearest target in the set.

    With a group such as "any corner" the player gets credit for whichever
    corner they were plainly going for.  A hit has to actually go in: a puck
    that clips the pipe inside the target circle rang off, it did not score.
    """
    if not targets:
        return None
    dists = [float(np.hypot(x - t.x_in, y - t.y_in)) for t in targets]
    i = int(np.argmin(dists))
    t = targets[i]
    went_in = outcome == "on_net" and point_in_mouth(goal, x, y)
    return ShotScore(
        target_key=t.key,
        target_label=t.label,
        hit=bool(went_in and dists[i] <= t.radius_in),
        distance_in=dists[i],
        dx_in=x - t.x_in,
        dy_in=y - t.y_in,
    )


def _bias_phrase(dx: float, dy: float, sig_x: bool, sig_y: bool) -> str | None:
    parts = []
    if sig_y:
        parts.append(f'{abs(dy):.0f}" {"high" if dy > 0 else "low"}')
    if sig_x:
        parts.append(f'{abs(dx):.0f}" {"right" if dx > 0 else "left"}')
    if not parts:
        return None
    return "shots land " + " and ".join(parts) + " of where they are aimed, on average"


def summarize_scores(scores: list[ShotScore], min_bias_in: float = 2.0) -> dict:
    """Hit rate, how close, and whether the misses lean one way."""
    scores = [s for s in scores if s is not None]
    n = len(scores)
    out: dict = {"shots": n, "hits": sum(s.hit for s in scores)}
    out["hit_rate_pct"] = round(100.0 * out["hits"] / n, 1) if n else None
    if not n:
        return out

    d = np.array([s.distance_in for s in scores])
    out["mean_distance_in"] = round(float(d.mean()), 1)
    out["best_distance_in"] = round(float(d.min()), 1)

    dx = np.array([s.dx_in for s in scores])
    dy = np.array([s.dy_in for s in scores])
    bias: dict = {
        "dx_in": round(float(dx.mean()), 1),
        "dy_in": round(float(dy.mean()), 1),
        "significant": False,
        "description": None,
    }
    if n >= 3:
        # A lean only counts if it clears twice its own standard error *and*
        # is big enough to matter on a hockey net.
        se_x = float(dx.std(ddof=1)) / np.sqrt(n)
        se_y = float(dy.std(ddof=1)) / np.sqrt(n)
        sig_x = abs(dx.mean()) >= max(2.0 * se_x, min_bias_in)
        sig_y = abs(dy.mean()) >= max(2.0 * se_y, min_bias_in)
        phrase = _bias_phrase(float(dx.mean()), float(dy.mean()), sig_x, sig_y)
        bias["significant"] = phrase is not None
        bias["description"] = phrase or "no consistent lean -- the misses scatter rather than drift one way"
    else:
        bias["description"] = "too few shots to tell whether the misses lean one way"
    out["bias"] = bias
    return out


def targeting_block(
    shots: list[tuple[float, float, str]],
    goal: GoalSpec,
    cfg: TargetConfig,
    plane=None,
) -> dict | None:
    """Everything the report and the browser need about aiming, in one place.

    ``shots`` is (x_in, y_in, outcome) per shot, in shot order.  ``plane``, if
    given, also yields each target's outline in image pixels for the overlay.
    """
    targets = resolve_targets(cfg, goal)
    if not targets:
        return None
    scores = [score_shot(x, y, o, targets, goal) for x, y, o in shots]

    target_dicts = []
    for t in targets:
        td = t.to_dict()
        if plane is not None:
            ang = np.linspace(0.0, 2.0 * np.pi, 40)
            ring = np.stack([t.x_in + t.radius_in * np.cos(ang), t.y_in + t.radius_in * np.sin(ang)], axis=1)
            td["image_outline"] = plane.to_image(ring).round(1).tolist()
        target_dicts.append(td)

    return {
        "kind": cfg.kind,
        "label": target_label(cfg),
        "radius_in": cfg.radius_in,
        "targets": target_dicts,
        "per_shot": [s.to_dict() if s else None for s in scores],
        "summary": summarize_scores(scores),
    }
