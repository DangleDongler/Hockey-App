"""Drawing the analysis back onto the footage.

Seeing the outline lock onto the net and the trail end in a dot is how a player
tells at a glance whether to believe the numbers.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .geometry import GoalPlane, mouth_rect
from .pipeline import SessionResult

OUTCOME_BGR = {"on_net": (94, 197, 34), "post": (11, 158, 245), "miss": (68, 68, 239)}
NET_BGR = (255, 210, 60)
TRAIL_BGR = (255, 255, 255)


def draw_net(frame: np.ndarray, plane: GoalPlane, *, zones: bool = True) -> None:
    outer = plane.image_quad.astype(np.int32)
    mouth = plane.to_image(mouth_rect(plane.goal)).astype(np.int32)
    cv2.polylines(frame, [outer], True, NET_BGR, 2, cv2.LINE_AA)
    cv2.polylines(frame, [mouth], True, NET_BGR, 1, cv2.LINE_AA)
    if not zones:
        return
    g = plane.goal
    for frac in (1 / 3, 2 / 3):
        x = -g.mouth_width_in / 2 + frac * g.mouth_width_in
        pts = plane.to_image([[x, 0.0], [x, g.mouth_height_in]]).astype(np.int32)
        cv2.line(frame, tuple(pts[0]), tuple(pts[1]), NET_BGR, 1, cv2.LINE_AA)
        y = frac * g.mouth_height_in
        pts = plane.to_image([[-g.mouth_width_in / 2, y], [g.mouth_width_in / 2, y]]).astype(np.int32)
        cv2.line(frame, tuple(pts[0]), tuple(pts[1]), NET_BGR, 1, cv2.LINE_AA)


def _label(frame: np.ndarray, text: str, org: tuple[int, int], color=(255, 255, 255), scale=0.5) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_overlay(result: SessionResult, out_path: str, *, trail_frames: int = 24, cfg: Config | None = None) -> str:
    """Write an annotated copy of the clip."""
    cfg = cfg or Config()
    plane = result.plane
    src = cv2.VideoCapture(result.video.path)
    if not src.isOpened():
        raise FileNotFoundError(result.video.path)

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        result.video.fps,
        (result.video.width, result.video.height),
    )
    if not writer.isOpened():
        src.release()
        raise RuntimeError(f"could not open {out_path} for writing")

    # Index track points by frame so drawing is a lookup, not a search.
    by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    for si, track in enumerate(result.tracks):
        for c in track.candidates:
            by_frame.setdefault(c.frame, []).append((si, np.array([c.x, c.y])))

    try:
        idx = 0
        while True:
            ok, frame = src.read()
            if not ok:
                break
            if plane is not None:
                draw_net(frame, plane)

            # Fading trail behind the puck.
            for si, track in enumerate(result.tracks):
                pts = [(c.frame, c.x, c.y) for c in track.candidates if idx - trail_frames <= c.frame <= idx]
                for j in range(1, len(pts)):
                    age = (idx - pts[j][0]) / max(trail_frames, 1)
                    shade = int(255 * (1.0 - 0.75 * age))
                    cv2.line(
                        frame,
                        (int(pts[j - 1][1]), int(pts[j - 1][2])),
                        (int(pts[j][1]), int(pts[j][2])),
                        (shade, shade, shade), 2, cv2.LINE_AA,
                    )
            for _, p in by_frame.get(idx, []):
                cv2.circle(frame, (int(p[0]), int(p[1])), 6, (255, 255, 255), 1, cv2.LINE_AA)

            # Impact marks persist once the shot has landed.
            for shot in result.shots:
                if idx < shot.impact_frame:
                    continue
                color = OUTCOME_BGR.get(shot.outcome, (180, 180, 180))
                px, py = int(shot.impact_image[0]), int(shot.impact_image[1])
                cv2.circle(frame, (px, py), 9, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 9, (20, 20, 20), 1, cv2.LINE_AA)
                tag = f"{shot.index + 1}"
                if shot.speed:
                    tag += f"  {shot.speed.mph:.0f} mph"
                _label(frame, tag, (px + 13, py - 8), color)

            hud = f"shots {sum(1 for s in result.shots if s.impact_frame <= idx)}/{len(result.shots)}"
            if result.shots:
                landed = [s for s in result.shots if s.impact_frame <= idx and s.speed]
                if landed:
                    hud += f"   best {max(s.speed.mph for s in landed):.0f} mph"
            _label(frame, hud, (14, 28), (255, 255, 255), scale=0.65)

            writer.write(frame)
            idx += 1
    finally:
        src.release()
        writer.release()
    return out_path
