"""Linking per-frame candidates into shot trajectories.

Nothing in a single frame tells you whether a dark blob is a puck.  What tells
you is what it does over the next tenth of a second: a shot crosses the rink
fast, in one direction, along a path that is very nearly a straight line in the
image — because the projection of a straight 3-D line is a straight 2-D line,
and over a two-tenths-of-a-second flight gravity only bends it slightly.

A knee, a stick blade and a shadow all fail at least one of those.
"""

from __future__ import annotations

import math
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
    cs = track.candidates[-lookback:] if direction > 0 else track.candidates[:lookback]
    if len(cs) < 2:
        return np.zeros(2)
    a, b = cs[0], cs[-1]
    return np.array([b.x - a.x, b.y - a.y]) / max(b.frame - a.frame, 1.0)


def _seed_pairs(
    xy: dict[int, np.ndarray], score: dict[int, np.ndarray], max_gap: int, max_step: float
) -> np.ndarray:
    """Every plausible pairing of detections a few frames apart, best first.

    Rows are (frame1, index1, frame2, index2).  Ties in score fall back to
    frame and index order, so the result never depends on dict ordering.
    """
    rows: list[np.ndarray] = []
    for f in sorted(xy):
        for gap_i in range(1, max_gap + 1):
            f2 = f + gap_i
            if f2 not in xy:
                continue
            d = np.hypot(xy[f2][None, :, 0] - xy[f][:, None, 0], xy[f2][None, :, 1] - xy[f][:, None, 1]) / gap_i
            i, j = np.nonzero((d <= max_step) & (d >= 1e-3))
            if len(i):
                s = -(score[f][i] + score[f2][j]) / 2.0
                rows.append(np.column_stack([s, np.full(len(i), f), i, np.full(len(i), f2), j]))
    if not rows:
        return np.zeros((0, 4), dtype=np.int64)
    allr = np.concatenate(rows)
    order = np.lexsort((allr[:, 4], allr[:, 3], allr[:, 2], allr[:, 1], allr[:, 0]))
    return allr[order, 1:].astype(np.int64)


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

    Seed pairs are found with array arithmetic, since there can be a million of
    them; continuing a track looks at a dozen detections, where plain Python
    is quicker.
    """
    notes = notes if notes is not None else []
    tcfg = cfg.track
    gw = max(goal_width_px, 1.0)
    max_gap = tcfg.max_frame_gap
    max_step = 0.9 * gw            # per frame; beyond this nothing is a puck
    gate_base = tcfg.gate_base_frac * gw
    gate_vel = tcfg.gate_vel_frac

    index = {f: cands_by_frame[f] for f in sorted(cands_by_frame) if cands_by_frame[f]}
    xy = {f: np.array([[c.x, c.y] for c in cs], dtype=np.float64) for f, cs in index.items()}
    score = {f: np.array([c.score for c in cs], dtype=np.float64) for f, cs in index.items()}
    claimed = {f: [False] * len(cs) for f, cs in index.items()}

    seeds = _seed_pairs(xy, score, max_gap, max_step)
    if len(seeds) > tcfg.max_seeds:
        notes.append(
            f"{len(seeds):,} possible puck pairings were found, far more than a clean clip "
            f"produces; only the {tcfg.max_seeds:,} strongest were followed, so a shot may have "
            "been missed among them."
        )
        seeds = seeds[: tcfg.max_seeds]

    # Plain floats: a frame holds a dozen detections at most, and at that size
    # Python arithmetic beats numpy's per-call overhead several times over.
    rows = {f: [(c.x, c.y, c.score) for c in cs] for f, cs in index.items()}

    def find_next(track: Track, direction: int) -> tuple[int, int] | None:
        """Best unclaimed candidate continuing the track forward/backward."""
        # Velocity at the end being extended (see _velocity), pointed the way
        # the search is going: backwards in time when extending the start.
        cs = track.candidates[-4:] if direction > 0 else track.candidates[:4]
        a, b = cs[0], cs[-1]
        dt = max(b.frame - a.frame, 1.0)
        vx, vy = direction * (b.x - a.x) / dt, direction * (b.y - a.y) / dt
        if len(cs) < 2:
            vx = vy = 0.0
        anchor = track.candidates[-1] if direction > 0 else track.candidates[0]
        speed = math.hypot(vx, vy)
        for gap_i in range(1, max_gap + 1):
            f = anchor.frame + direction * gap_i
            row = rows.get(f)
            if row is None:
                continue
            gate = gate_base + gate_vel * speed * gap_i
            px, py = anchor.x + vx * gap_i, anchor.y + vy * gap_i
            taken = claimed[f]
            best, best_cost = -1, math.inf
            for k, (x, y, sc) in enumerate(row):
                if taken[k]:
                    continue
                d = math.hypot(x - px, y - py)
                if d > gate:
                    continue
                cost = d / max(gate, 1e-6) - 0.3 * sc + 0.12 * (gap_i - 1)
                if cost < best_cost:
                    best_cost, best = cost, k
            if best >= 0:
                return f, best  # prefer the nearest frame that offers anything
        return None

    tracks: list[Track] = []
    for f1, i1, f2, i2 in seeds.tolist():
        if claimed[f1][i1] or claimed[f2][i2]:
            continue

        track = Track([index[f1][i1], index[f2][i2]])
        newly: list[tuple[int, int]] = [(f1, i1), (f2, i2)]
        claimed[f1][i1] = claimed[f2][i2] = True

        for direction in (+1, -1):
            while True:
                nxt = find_next(track, direction)
                if nxt is None:
                    break
                c = index[nxt[0]][nxt[1]]
                if direction > 0:
                    track.candidates.append(c)
                else:
                    track.candidates.insert(0, c)
                claimed[nxt[0]][nxt[1]] = True
                newly.append(nxt)

        if not _accept(track, gw, cfg):
            # Put the detections back so a better seed can use them.
            for f, k in newly:
                claimed[f][k] = False
        else:
            tracks.append(track)

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
    """Drop trajectories too slow to be a shot, or seen too briefly to be one."""
    min_step = cfg.track.min_mean_speed_goalwidths_per_sec * goal_width_px / max(fps, 1e-6)
    min_span = cfg.track.min_track_s * fps
    return [t for t in tracks if t.mean_step_px >= min_step and t.span_frames >= min_span]
