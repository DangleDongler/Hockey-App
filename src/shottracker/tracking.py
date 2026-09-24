"""Linking per-frame candidates into shot trajectories.

Nothing in a single frame tells you whether a dark blob is a puck.  What tells
you is what it does over the next tenth of a second: a shot crosses the rink
fast, in one direction, along a path that is very nearly a straight line in the
image — because the projection of a straight 3-D line is a straight 2-D line,
and over a two-tenths-of-a-second flight gravity only bends it slightly.

A knee, a stick blade and a shadow all fail at least one of those.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .puck_detect import Candidate


@dataclass
class Track:
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def frames(self) -> np.ndarray:
        return np.array([c.frame for c in self.candidates], dtype=np.float64)

    @property
    def points(self) -> np.ndarray:
        return np.array([[c.x, c.y] for c in self.candidates], dtype=np.float64)

    @property
    def scores(self) -> np.ndarray:
        return np.array([c.score for c in self.candidates], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def start_frame(self) -> int:
        return self.candidates[0].frame

    @property
    def end_frame(self) -> int:
        return self.candidates[-1].frame

    @property
    def span_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def displacement_px(self) -> float:
        p = self.points
        return float(np.linalg.norm(p[-1] - p[0]))

    @property
    def mean_step_px(self) -> float:
        """Average image motion per frame."""
        if self.span_frames <= 0:
            return 0.0
        return self.displacement_px / self.span_frames

    def direction_coherence(self) -> float:
        """How consistently the steps point the same way. 1.0 is a ruler."""
        p = self.points
        if len(p) < 3:
            return 1.0
        overall = p[-1] - p[0]
        n = np.linalg.norm(overall)
        if n < 1e-6:
            return 0.0
        overall = overall / n
        steps = np.diff(p, axis=0)
        mags = np.linalg.norm(steps, axis=1)
        good = mags > 1e-6
        if not good.any():
            return 0.0
        units = steps[good] / mags[good, None]
        return float(np.mean(units @ overall))

    def path_residual_px(self) -> float:
        """RMS distance of the detections from the straight line they should lie on.

        This is measured in space, not against time.  The projection of a
        straight 3-D line is a straight 2-D line no matter how the puck is
        parameterised along it -- which matters, because perspective makes a
        receding puck crawl across the image near the net and race across it
        near the camera.  Fitting position against *time* would call that
        curvature; fitting the shape does not.  What is left is the slight arc
        gravity puts on the flight.
        """
        p = self.points
        if len(p) < 3:
            return 0.0
        centred = p - p.mean(axis=0)
        # Total least squares: the smallest singular value is the RMS spread
        # perpendicular to the best-fit line.
        sv = np.linalg.svd(centred, compute_uv=False)
        return float(sv[-1] / np.sqrt(len(p)))

    def fit_poly(self, deg: int | None = None, window: int | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        """Polynomial fit of x(frame) and y(frame). Returns (cx, cy, t0).

        ``window`` restricts the fit to the last N detections, which is what
        you want when extrapolating past the end of the track: perspective
        makes a single global polynomial a poor description of a long flight.
        """
        t, p = self.frames, self.points
        if window is not None and len(t) > window:
            t, p = t[-window:], p[-window:]
        if deg is None:
            deg = 2 if len(t) >= 6 else 1
        deg = min(deg, len(t) - 1)
        t0 = float(t[0])
        cx = np.polyfit(t - t0, p[:, 0], deg)
        cy = np.polyfit(t - t0, p[:, 1], deg)
        return cx, cy, t0

    def position_at(self, frame: float, window: int | None = 6) -> np.ndarray:
        cx, cy, t0 = self.fit_poly(window=window)
        return np.array([np.polyval(cx, frame - t0), np.polyval(cy, frame - t0)], dtype=np.float64)

    def quality(self) -> float:
        return float(self.scores.mean() * min(len(self) / 8.0, 1.0))


def _velocity(track: Track, direction: int = +1, lookback: int = 4) -> np.ndarray:
    """Per-frame image velocity at the end of the track we are extending.

    Perspective makes image speed vary a lot along one flight -- a puck heading
    away from the camera slows down on screen by several fold -- so extending
    backwards has to use the velocity at the *start* of the track, not the end.
    """
    pts = track.points[-lookback:] if direction > 0 else track.points[:lookback]
    frs = track.frames[-lookback:] if direction > 0 else track.frames[:lookback]
    if len(pts) < 2:
        return np.zeros(2)
    return (pts[-1] - pts[0]) / max(frs[-1] - frs[0], 1.0)


def build_tracks(
    cands_by_frame: dict[int, list[Candidate]],
    goal_width_px: float,
    cfg: Config,
    notes: list[str] | None = None,
) -> list[Track]:
    """Grow trajectories out of scored per-frame candidates.

    Seeds are tried best-first and claim the detections they consume, so the
    strongest evidence wins and the search stays linear in practice -- as long
    as the number of candidates per frame stays small.  When the foreground
    model fails, every frame fills with candidates and the seed set grows with
    their square, so it is capped.
    """
    notes = notes if notes is not None else []
    tcfg = cfg.track
    gw = max(goal_width_px, 1.0)
    max_gap = tcfg.max_frame_gap
    max_step = 0.9 * gw            # per frame; beyond this nothing is a puck
    gate_base = tcfg.gate_base_frac * gw
    gate_vel = tcfg.gate_vel_frac

    frames = sorted(cands_by_frame)
    index = {f: cands_by_frame[f] for f in frames}
    claimed: set[tuple[int, int]] = set()
    tried_seeds: set[tuple[int, int, int, int]] = set()

    # Build seed pairs and try the most promising first.
    seeds: list[tuple[float, int, int, int, int]] = []
    for fi, f in enumerate(frames):
        for gap_i in range(1, max_gap + 1):
            f2 = f + gap_i
            if f2 not in index:
                continue
            for i, c1 in enumerate(index[f]):
                for j, c2 in enumerate(index[f2]):
                    step = np.hypot(c2.x - c1.x, c2.y - c1.y) / gap_i
                    if step > max_step or step < 1e-3:
                        continue
                    seeds.append((-(c1.score + c2.score) / 2.0, f, i, f2, j))
    seeds.sort()
    if len(seeds) > tcfg.max_seeds:
        notes.append(
            f"{len(seeds):,} possible puck pairings were found, far more than a clean clip "
            f"produces; only the {tcfg.max_seeds:,} strongest were followed, so a shot may have "
            "been missed among them."
        )
        seeds = seeds[: tcfg.max_seeds]

    def find_next(track: Track, direction: int) -> tuple[int, int] | None:
        """Best unclaimed candidate continuing the track forward/backward."""
        v = _velocity(track, direction) * direction
        anchor_pt = track.points[-1] if direction > 0 else track.points[0]
        anchor_f = track.end_frame if direction > 0 else track.start_frame
        best = None
        best_cost = np.inf
        for gap_i in range(1, max_gap + 1):
            f = anchor_f + direction * gap_i
            if f not in index:
                continue
            predicted = anchor_pt + v * gap_i
            gate = gate_base + gate_vel * np.linalg.norm(v) * gap_i
            for k, c in enumerate(index[f]):
                if (f, k) in claimed:
                    continue
                d = float(np.hypot(c.x - predicted[0], c.y - predicted[1]))
                if d > gate:
                    continue
                cost = d / max(gate, 1e-6) - 0.3 * c.score + 0.12 * (gap_i - 1)
                if cost < best_cost:
                    best_cost, best = cost, (f, k)
            if best is not None:
                break  # prefer the nearest frame that offers anything
        return best

    tracks: list[Track] = []
    for _, f1, i1, f2, i2 in seeds:
        key = (f1, i1, f2, i2)
        if key in tried_seeds:
            continue
        tried_seeds.add(key)
        if (f1, i1) in claimed or (f2, i2) in claimed:
            continue

        track = Track([index[f1][i1], index[f2][i2]])
        newly: list[tuple[int, int]] = [(f1, i1), (f2, i2)]
        for c in newly:
            claimed.add(c)

        while True:
            nxt = find_next(track, +1)
            if nxt is None:
                break
            track.candidates.append(index[nxt[0]][nxt[1]])
            claimed.add(nxt)
            newly.append(nxt)
        while True:
            nxt = find_next(track, -1)
            if nxt is None:
                break
            track.candidates.insert(0, index[nxt[0]][nxt[1]])
            claimed.add(nxt)
            newly.append(nxt)

        if _accept(track, gw, cfg):
            tracks.append(track)
        else:
            # Put the detections back so a better seed can use them.
            for c in newly:
                claimed.discard(c)

    tracks.sort(key=lambda t: t.start_frame)
    return tracks


def _accept(track: Track, goal_width_px: float, cfg: Config) -> bool:
    tcfg = cfg.track
    if len(track) < tcfg.min_track_length:
        return False
    if track.span_frames <= 0:
        return False
    if track.direction_coherence() < 0.90:
        return False
    if track.path_residual_px() > tcfg.max_line_residual_frac * goal_width_px:
        return False
    return True


def filter_by_speed(tracks: list[Track], goal_width_px: float, fps: float, cfg: Config) -> list[Track]:
    """Drop trajectories too slow to be a shot."""
    min_step = cfg.track.min_mean_speed_goalwidths_per_sec * goal_width_px / max(fps, 1e-6)
    return [t for t in tracks if t.mean_step_px >= min_step]
