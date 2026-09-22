"""Finding puck-shaped things in each frame.

The camera is a phone on a tripod, so the cheapest reliable foreground model is
a per-pixel median over the clip: a puck is in any given pixel for one or two
frames out of hundreds, so the median is a clean, puck-free plate of the rink.
Anything that differs from the plate is a candidate.

Candidates are *scored*, not filtered to one.  A stick blade, a skate and a
shadow all survive this stage; it is the trajectory stage that knows a puck
from a knee, because only one of them crosses the rink in a straight line at
eighty miles an hour.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config


@dataclass
class Candidate:
    frame: int
    x: float           # full-resolution image coordinates
    y: float
    area_px: float     # measured at working resolution
    score: float       # 0..1, higher is more puck-like
    aspect: float      # major/minor axis of the blur streak
    angle_deg: float   # streak orientation
    length_px: float   # major axis, i.e. how far it smeared
    darkness: float    # 0..1

    def point(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


def build_background(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median plate. Returns a grayscale image."""
    if not frames:
        raise ValueError("no frames to build a background from")
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    return np.median(grays, axis=0).astype(np.uint8)


class PuckDetector:
    """Per-frame candidate extraction against a fixed background plate."""

    def __init__(self, cfg: Config, background: np.ndarray | None, puck_px: float, scale: float):
        """
        ``puck_px`` is the puck's apparent diameter on the goal plane, in
        working-resolution pixels; it sets the size bounds.  ``scale`` converts
        working-resolution coordinates back to full resolution.
        """
        self.cfg = cfg
        self.background = background
        self.scale = scale
        self.puck_px = max(puck_px, 1.5)

        pcfg = cfg.puck
        puck_area = np.pi / 4.0 * self.puck_px * (self.puck_px * 0.45)
        self.min_area = max(pcfg.min_area_px, pcfg.min_area_mult * puck_area)
        self.max_area = max(self.min_area * 4.0, pcfg.max_area_mult * puck_area)

        self._mog = None
        if pcfg.method == "mog2" or background is None:
            self._mog = cv2.createBackgroundSubtractorMOG2(
                history=pcfg.mog_history, varThreshold=pcfg.mog_var_threshold, detectShadows=False
            )

    def foreground(self, small_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        if self._mog is not None:
            fg = self._mog.apply(small_bgr)
            fg = (fg > 200).astype(np.uint8) * 255
        else:
            diff = cv2.absdiff(gray, self.background)
            fg = (diff > self.cfg.puck.diff_threshold).astype(np.uint8) * 255
        # Close pinholes in the blur streak without merging separate objects.
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return fg

    def detect(self, small_bgr: np.ndarray, frame_idx: int) -> list[Candidate]:
        pcfg = self.cfg.puck
        gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        fg = self.foreground(small_bgr)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
        out: list[Candidate] = []
        for i in range(1, num):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area or area > self.max_area:
                continue

            # Work inside the component's own bounding box.  Comparing the
            # whole label image against i once per component is what makes
            # this loop quadratic in practice: a frame with a few hundred
            # movers would scan half a megapixel a few hundred times.
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            sub = labels[by : by + bh, bx : bx + bw] == i
            ys, xs = np.nonzero(sub)
            if len(xs) < 3:
                continue

            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            (_, _), (ew, eh), ang = cv2.minAreaRect(pts)
            major, minor = max(ew, eh), max(min(ew, eh), 1e-3)
            aspect = major / minor
            if aspect > pcfg.max_streak_aspect:
                continue

            mean_val = float(gray[by : by + bh, bx : bx + bw][sub].mean())
            darkness = float(np.clip((pcfg.dark_value_max - mean_val) / pcfg.dark_value_max, 0.0, 1.0))

            # Size score peaks at the puck's expected blurred footprint and
            # falls off logarithmically either side, so a slightly-off blob
            # still competes but a jersey does not.
            ideal = np.pi / 4.0 * self.puck_px * (self.puck_px * 0.45) * 3.0
            size_score = float(np.exp(-0.5 * (np.log(area / ideal) / 1.1) ** 2))
            # A compact-ish blob or a clean streak both score well; ragged
            # many-holed regions do not.
            fill = area / max(major * minor, 1e-3)
            fill_score = float(np.clip(fill / 0.65, 0.0, 1.0))

            score = float(
                np.clip(
                    (1.0 - pcfg.dark_weight) * (0.6 * size_score + 0.4 * fill_score)
                    + pcfg.dark_weight * darkness,
                    0.0,
                    1.0,
                )
            )

            cx, cy = centroids[i]
            out.append(
                Candidate(
                    frame=frame_idx,
                    x=float(cx) / self.scale,
                    y=float(cy) / self.scale,
                    area_px=area,
                    score=score,
                    aspect=float(aspect),
                    angle_deg=float(ang),
                    length_px=float(major),
                    darkness=darkness,
                )
            )

        out.sort(key=lambda c: c.score, reverse=True)
        return out[: pcfg.max_candidates_per_frame]
