"""Turning a trajectory into a shot: where it hit, and how hard.

The impact point is the one number a shooter actually cares about, and it is
the last thing the track can tell us.  A puck stops dead when it hits the mesh,
so the trajectory ends at the net; we take the fitted path a fraction of a
frame past the final detection and map that point through the goal-plane
homography into inches on the net.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import CameraModel
from .config import Config
from .geometry import GoalPlane, Zone, build_zones, outer_outline, point_in_mouth, zone_for
from .speed import SpeedEstimate, estimate_speed
from .tracking import Track


@dataclass
class Shot:
    index: int
    # The first frame the puck was tracked in.  That is at or after the real
    # release -- the puck is often hidden by the player for the first few
    # frames -- so this is not a flight-time measurement.
    first_tracked_frame: int
    impact_frame: int
    impact_image: tuple[float, float]      # full-res pixels
    impact_goal_in: tuple[float, float]    # inches, origin on the ice at net centre
    outcome: str                           # "on_net" | "post" | "miss"
    zone_key: str | None
    zone_label: str | None
    miss_detail: str | None
    speed: SpeedEstimate | None
    track_length: int
    quality: float
    notes: list[str] = field(default_factory=list)

    @property
    def on_net(self) -> bool:
        return self.outcome == "on_net"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "first_tracked_frame": self.first_tracked_frame,
            "impact_frame": self.impact_frame,
            "impact_image": [round(float(v), 1) for v in self.impact_image],
            "impact_goal_in": [round(float(v), 1) for v in self.impact_goal_in],
            "outcome": self.outcome,
            "zone_key": self.zone_key,
            "zone_label": self.zone_label,
            "miss_detail": self.miss_detail,
            "speed": self.speed.to_dict() if self.speed else None,
            "track_length": self.track_length,
            "quality": round(float(self.quality), 3),
            "notes": list(self.notes),
        }


def _classify(
    x: float, y: float, plane: GoalPlane, zones: list[Zone], cfg: Config
) -> tuple[str, str | None, str | None]:
    """Decide whether the puck went in, hit iron, or missed."""
    goal = plane.goal
    hw = goal.mouth_width_in / 2.0
    top = goal.mouth_height_in
    tol = cfg.shot.post_tolerance_in

    inside = point_in_mouth(goal, x, y)

    # The bend where post meets crossbar comes first: it sits within a pipe's
    # width of both straight edges, so a plain "post" or "crossbar" test would
    # always claim it, and "rang it off the top corner" is the better answer.
    in_box = (-hw <= x <= hw) and (0.0 <= y <= top)
    if in_box and not inside:
        side = "left" if x < 0 else "right"
        return "post", None, f"{side} corner bend"

    # Within a pipe's width of the frame line, either side of it, is iron.
    near_left = abs(x + hw) <= tol
    near_right = abs(x - hw) <= tol
    near_bar = abs(y - top) <= tol
    if (near_left or near_right) and -tol <= y <= top + tol:
        return "post", None, "left post" if near_left else "right post"
    if near_bar and -hw - tol <= x <= hw + tol:
        return "post", None, "crossbar"

    if inside:
        z = zone_for(zones, x, y)
        return "on_net", (z.key if z else None), None

    parts = []
    if x < -hw:
        parts.append(f"{abs(x + hw) / 12.0:.1f} ft wide left")
    elif x > hw:
        parts.append(f"{abs(x - hw) / 12.0:.1f} ft wide right")
    if y > top:
        parts.append(f"{(y - top) / 12.0:.1f} ft high")
    elif y < 0:
        parts.append("into the ice")
    return "miss", None, " and ".join(parts) if parts else "wide"


def approaches_goal(track: Track, plane: GoalPlane, cfg: Config) -> bool:
    """Whether a trajectory arrives at the goal from somewhere else.

    Measured on screen, in goal widths outside the goal's outline.  Tracks
    that begin on the frame itself -- a post glinting, the netting swaying
    after a puck hits it, a rebound dropping out -- are not shots, however
    neatly they line up.
    """
    if len(track) == 0:
        return False
    outline = plane.to_image(outer_outline(plane.goal)).astype(np.float32).reshape(-1, 1, 2)
    width_px = float(np.linalg.norm(plane.image_quad[1] - plane.image_quad[0]))
    if width_px <= 0:
        return True

    def outside(p: np.ndarray) -> float:
        return -cv2.pointPolygonTest(outline, (float(p[0]), float(p[1])), True) / width_px

    start, end = outside(track.points[0]), outside(track.points[-1])
    return start >= cfg.shot.min_start_outside_goal and start - end >= cfg.shot.min_approach_goal


def shot_from_track(
    index: int,
    track: Track,
    plane: GoalPlane,
    cam: CameraModel | None,
    fps: float,
    cfg: Config,
    zones: list[Zone] | None = None,
) -> Shot | None:
    zones = zones or build_zones(plane.goal)

    # The puck is caught by the mesh, so the track ends at the net.  Step a
    # fraction of a frame past the last detection to land on the plane itself.
    last_frame = track.end_frame
    impact_frame_f = last_frame + cfg.shot.impact_extrapolation_frames
    impact_px = track.position_at(impact_frame_f)
    if not np.all(np.isfinite(impact_px)):
        impact_px = track.points[-1]
        impact_frame_f = float(last_frame)

    goal_xy = plane.to_goal(impact_px.reshape(1, 2))[0]
    x, y = float(goal_xy[0]), float(goal_xy[1])

    margin = cfg.shot.miss_margin_in
    hw = plane.goal.mouth_width_in / 2.0
    if not (-hw - margin <= x <= hw + margin and y <= plane.goal.mouth_height_in + margin):
        # Nowhere near the net: this trajectory was not a shot at this goal.
        return None
    if y < -cfg.shot.below_ice_margin_in:
        # Mapping below the ice means the puck was still well in front of the
        # goal plane, so this is a piece of a flight, not the end of one.
        return None

    outcome, zone_key, miss_detail = _classify(x, y, plane, zones, cfg)
    zone_label = None
    if zone_key:
        zone_label = next((z.label for z in zones if z.key == zone_key), None)

    speed = estimate_speed(track, plane, cam, np.array([x, y]), fps, cfg)

    notes: list[str] = []
    if len(track) < 6:
        notes.append("short track; the impact point is extrapolated from few detections")

    return Shot(
        index=index,
        first_tracked_frame=int(track.start_frame),
        impact_frame=int(round(impact_frame_f)),
        impact_image=(float(impact_px[0]), float(impact_px[1])),
        impact_goal_in=(x, y),
        outcome=outcome,
        zone_key=zone_key,
        zone_label=zone_label,
        miss_detail=miss_detail,
        speed=speed,
        track_length=len(track),
        quality=track.quality(),
        notes=notes,
    )


MERGED = "merged with a near-simultaneous trajectory (likely a rebound)"
AFTERMATH = "ignored movement at the net straight after the impact"


def _note(shot: Shot, text: str) -> None:
    if text not in shot.notes:
        shot.notes.append(text)


def dedupe_shots(shots: list[Shot], cfg: Config, fps: float) -> list[Shot]:
    """Collapse trajectories that are really one shot seen twice (e.g. a rebound)."""
    if not shots:
        return []
    shots = sorted(shots, key=lambda s: s.impact_frame)
    min_gap_frames = cfg.shot.min_shot_separation_s * fps
    kept: list[Shot] = [shots[0]]
    for s in shots[1:]:
        prev = kept[-1]
        # One player cannot have two pucks in the air at the same time, so
        # overlapping flights are one shot whose track came apart: keep
        # whichever piece was followed better.
        if s.first_tracked_frame <= prev.impact_frame:
            if s.quality > prev.quality:
                _note(s, MERGED)
                kept[-1] = s
            else:
                _note(prev, MERGED)
            continue
        # Something that starts after an impact and lands straight after it is
        # the impact's aftermath -- a rebound, the netting moving -- never a
        # better view of the shot, however cleanly it happened to be tracked.
        if s.impact_frame - prev.impact_frame < min_gap_frames:
            _note(prev, AFTERMATH)
            continue
        kept.append(s)
    for i, s in enumerate(kept):
        s.index = i
    return kept
