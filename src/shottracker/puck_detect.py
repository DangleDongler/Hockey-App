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
    return build_background_and_noise(frames)[0]


def build_background_and_noise(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median plate, and how much each pixel normally strays from it.

    The spread is the median absolute deviation over the same frames, scaled
    to a standard deviation.  Gravel and concrete just in front of the lens
    shimmer by a few grey levels frame to frame from sensor noise alone; with
    one threshold for the whole picture that shimmer broke into ~1,600
    puck-sized specks a frame on a real 4K clip, and the puck was cut from
    the list before anything could tell it apart.
    """
    if not frames:
        raise ValueError("no frames to build a background from")
    grays = np.stack([f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    plate = np.median(grays, axis=0)
    spread = 1.4826 * np.median(np.abs(grays.astype(np.float32) - plate), axis=0)
    return plate.astype(np.uint8), spread.astype(np.float32)


class PuckDetector:
    """Per-frame candidate extraction against a fixed background plate."""

    def __init__(self, cfg: Config, background: np.ndarray | None, puck_px: float, scale: float,
                 noise: np.ndarray | None = None):
        """
        ``puck_px`` is the puck's apparent diameter on the goal plane, in
        working-resolution pixels; it sets the size bounds.  ``scale`` converts
        working-resolution coordinates back to full resolution.  ``noise`` is
        each pixel's usual spread about the plate (see
        build_background_and_noise); where it is large the threshold rises.
        """
        self.cfg = cfg
        self.background = background
        self.scale = scale
        self.threshold = None
        if background is not None and noise is not None and cfg.puck.noise_threshold_k > 0:
            self.threshold = np.maximum(float(cfg.puck.diff_threshold),
                                        cfg.puck.noise_threshold_k * noise).astype(np.float32)
        self.puck_px = max(puck_px, 1.5)

        pcfg = cfg.puck
        puck_area = np.pi / 4.0 * self.puck_px * (self.puck_px * 0.45)
        self.min_area = max(pcfg.min_area_px, pcfg.min_area_mult * puck_area)
        self.max_area = max(self.min_area * 4.0, pcfg.max_area_mult * puck_area)

        self._mog = None
        # Whether detect() carries state from one frame to the next, and so
        # must see frames one at a time, in order.
        self.stateful = pcfg.method == "mog2" or background is None
        if self.stateful:
            self._mog = cv2.createBackgroundSubtractorMOG2(
                history=pcfg.mog_history, varThreshold=pcfg.mog_var_threshold, detectShadows=False
            )

    def foreground(self, small_bgr: np.ndarray, gray: np.ndarray | None = None) -> np.ndarray:
        """Pixels that differ from the plate.  ``small_bgr`` may already be grey."""
        if gray is None:
            gray = small_bgr if small_bgr.ndim == 2 else cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        if self._mog is not None:
            fg = self._mog.apply(small_bgr)
            fg = (fg > 200).astype(np.uint8) * 255
        else:
            diff = cv2.absdiff(gray, self.background)
            if self.threshold is not None:
                fg = (diff.astype(np.float32) > self.threshold).astype(np.uint8) * 255
            else:
                fg = (diff > self.cfg.puck.diff_threshold).astype(np.uint8) * 255
        # Close pinholes in the blur streak without merging separate objects.
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        return fg

    def detect(self, small_bgr: np.ndarray, frame_idx: int, limit: int | None = None) -> list[Candidate]:
        """Candidates in one frame, best first; at most ``limit`` (default: the configured cap)."""
        pcfg = self.cfg.puck
        gray = small_bgr if small_bgr.ndim == 2 else cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        fg = self.foreground(small_bgr, gray)

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
        return out[: limit or pcfg.max_candidates_per_frame]


def recurring(
    cands_by_frame: dict[int, list[Candidate]],
    radius_px: float,
    window: int,
    gap: int,
    min_hits: int,
    chunk: int = 256,
) -> set[tuple[int, int]]:
    """Which candidates sit where something was seen both just before and just after.

    A candidate counts as recurring when, within ``radius_px``, other
    candidates turn up in at least ``min_hits`` distinct frames among the
    ``window`` frames before it, and again among the ``window`` after --
    ignoring the ``gap`` frames either side, where a slow puck can still
    overlap itself.  Returns (frame, index in that frame's list) pairs.
    """
    from scipy.spatial import cKDTree

    frames = sorted(f for f, v in cands_by_frame.items() if v)
    if not frames:
        return set()
    F = np.concatenate([np.full(len(cands_by_frame[f]), f) for f in frames])
    K = np.concatenate([np.arange(len(cands_by_frame[f])) for f in frames])
    P = np.array([(c.x, c.y) for f in frames for c in cands_by_frame[f]], dtype=np.float64)
    out: set[tuple[int, int]] = set()
    # In time chunks, so memory follows the chunk, not the clip.
    for lo in range(frames[0], frames[-1] + 1, chunk):
        core = np.nonzero((F >= lo) & (F < lo + chunk))[0]
        if not len(core):
            continue
        ext = np.nonzero((F >= lo - window) & (F < lo + chunk + window))[0]
        pairs = cKDTree(P[core]).sparse_distance_matrix(cKDTree(P[ext]), radius_px, output_type="ndarray")
        if not len(pairs):
            continue
        a, b = core[pairs["i"]], ext[pairs["j"]]
        df = F[b] - F[a]
        for side in (-1, 1):
            sel = (side * df >= gap) & (side * df <= window)
            # Distinct frames per candidate: several fragments of one leaf in
            # one frame are one sighting.
            keys = np.unique(a[sel] * (2 * window + 1) + (df[sel] + window))
            hits = np.bincount(keys // (2 * window + 1), minlength=len(F))
            if side == -1:
                before = hits
            else:
                both = np.nonzero(np.minimum(before, hits) >= min_hits)[0]
                out.update(zip(F[both].tolist(), K[both].tolist()))
    return out


def crowded(cands: list[Candidate], radius_px: float, max_neighbours: int) -> np.ndarray:
    """Which candidates have more than ``max_neighbours`` others within ``radius_px``."""
    if len(cands) <= max_neighbours + 1:
        return np.zeros(len(cands), dtype=bool)
    p = np.array([(c.x, c.y) for c in cands], dtype=np.float64)
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    return ((d > 1.0) & (d < radius_px)).sum(axis=1) > max_neighbours


def demote_recurring(
    cands_by_frame: dict[int, list[Candidate]],
    radius_px: float,
    fps: float,
    cfg: Config,
    crowd_radius_px: float | None = None,
) -> tuple[dict[int, list[Candidate]], int, int]:
    """Rank recurring clutter last in every frame, then apply the per-frame cap.

    With ``crowd_radius_px``, candidates in a crowd (see ``crowded``) rank
    between the ones standing alone and the recurring clutter.

    Returns (candidates, how many were demoted, how many frames were still
    full of candidates that were not clutter).
    """
    pcfg = cfg.puck
    window = max(pcfg.recurring_min_window_frames, int(round(pcfg.recurring_window_s * fps)))
    gap = max(2, int(round(0.0125 * fps)))
    flagged = recurring(cands_by_frame, radius_px, window, gap, pcfg.recurring_min_hits)
    cap = pcfg.max_candidates_per_frame
    out: dict[int, list[Candidate]] = {}
    busy = 0
    for f, lst in cands_by_frame.items():
        clear = [c for k, c in enumerate(lst) if (f, k) not in flagged]
        clutter = [c for k, c in enumerate(lst) if (f, k) in flagged]
        busy += len(clear) >= cap
        if crowd_radius_px and len(clear) > cap:
            # Only matters when the cap would cut something that is not clutter.
            dense = crowded(lst, crowd_radius_px, cfg.track.clutter_max_neighbours)
            alone = [c for k, c in enumerate(lst) if (f, k) not in flagged and not dense[k]]
            crowd = [c for k, c in enumerate(lst) if (f, k) not in flagged and dense[k]]
            clear = alone + crowd
        kept = (clear + clutter)[:cap]
        if kept:
            out[f] = kept
    return out, len(flagged), busy
