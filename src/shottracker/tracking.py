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
        # Total least squares: the smaller eigenvalue of the points' 2x2
        # covariance is the mean squared spread perpendicular to the best-fit
        # line.  In closed form, because this runs hundreds of thousands of
        # times on a long clip.
        d = p - p.mean(axis=0)
        sxx, syy, sxy = float(d[:, 0] @ d[:, 0]), float(d[:, 1] @ d[:, 1]), float(d[:, 0] @ d[:, 1])
        half = 0.5 * (sxx + syy)
        low = half - np.sqrt(max(0.25 * (sxx - syy) ** 2 + sxy * sxy, 0.0))
        return float(np.sqrt(max(low, 0.0) / len(p)))

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
    fps: float | None = None,
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
    searched: dict = {}
    # Slower on screen than this a frame, nothing salvaged would be a shot.
    min_step = (tcfg.min_mean_speed_goalwidths_per_sec * gw / fps) if fps else 0.0
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

        if fps is not None and fps >= cfg.track.straggler_min_fps:
            track = _without_stragglers(track, cfg)
        if not _accept(track, gw, cfg):
            track = _straight_part(track, gw, cfg, min_step, searched)
        # Put back whatever was not kept, so a better seed can use it.
        keep = {id(c) for c in track.candidates} if track is not None else set()
        for f, k in newly:
            if id(index[f][k]) not in keep:
                claimed[f][k] = False
        if track is not None:
            tracks.append(track)

    tracks.sort(key=lambda t: t.start_frame)
    return tracks


def _without_stragglers(track: Track, cfg: Config) -> Track:
    """Drop a lone detection or two cut off from either end by a gap.

    Bridging a gap of a couple of frames keeps a track whole when the puck
    blinks out mid-flight.  At the ends it can do harm.  Before the release,
    the thing picked up across the gap is as likely a puck lying in the pile
    as the one being shot, so a stray start is always dropped.  After the
    impact, the first thing near where the puck *would* have gone -- the
    netting springing back, a bounce -- is taken as the puck and the mark is
    read from it: on a real 60 fps shot one such point moved the mark 17
    inches.  But a puck hidden behind a post for two frames reappears on its
    line at the impact, and that point is the most valuable one there is.  So
    a stray end is dropped only when it is off the line the track was on.

    Not at 30 fps, where a flight is a handful of frames and the puck is
    routinely missed for two of them: there the end detections are real far
    more often than not, and dropping them cost more than it saved.
    """
    tcfg = cfg.track
    cs = list(track.candidates)
    gap, run = tcfg.straggler_gap_frames, tcfg.max_straggler_run

    def off_line(body: list[Candidate], tail: list[Candidate]) -> bool:
        ref = body[-5:]
        d = np.array([ref[-1].x - ref[0].x, ref[-1].y - ref[0].y])
        span = max(ref[-1].frame - ref[0].frame, 1)
        norm = float(np.linalg.norm(d))
        if norm < 1e-6:
            return True
        u = d / norm
        anchor = ref[-1]
        for c in tail:
            v = np.array([c.x - anchor.x, c.y - anchor.y])
            travel = norm / span * abs(c.frame - anchor.frame)
            if abs(v[0] * u[1] - v[1] * u[0]) > max(3.0, tcfg.straggler_off_line * travel):
                return True
        return False

    for at_end in (True, False):
        cs.reverse()  # reversed: the end first; reversed back: the start
        removed = 0
        while removed < run:
            k = next((k for k in range(1, run - removed + 1)
                      if len(cs) - k >= tcfg.min_track_length and abs(cs[k].frame - cs[k - 1].frame) >= gap), None)
            if k is None:
                break
            # cs runs from this end inward: cs[:k] is the stray piece.
            if at_end and not off_line(list(reversed(cs[k:])), cs[:k][::-1]):
                break
            cs, removed = cs[k:], removed + k
    cs.sort(key=lambda c: c.frame)
    return track if len(cs) == len(track.candidates) else Track(cs)


def until_impact(track: Track, fps: float, cfg: Config, near_goal=None) -> Track:
    """End the track where the puck stopped flying.

    In flight a puck's path on screen is smooth: it bends only as gravity
    and perspective bend it, and its speed on screen changes slowly as it
    nears or leaves the camera.  At the net that ends abruptly -- it drops
    into the mesh, bounces off the bar, rides the netting back -- and a
    detector sensitive enough to follow the flight follows that too.  On real
    60 fps copies those few extra points moved the mark by 12 to 28 inches.

    Steps are compared at 60 fps spacing whatever the frame rate, so pixel
    jitter between 240 fps frames is not mistaken for a turn.  A break only
    counts where ``near_goal(point)`` says the puck was at the net: a wobble
    mid-flight, far from it, once cut a real track in half.
    """
    if fps < cfg.track.straggler_min_fps:
        return track
    tcfg = cfg.track
    cs = track.candidates
    stride = max(1, int(round(fps / 60.0)))
    # Thin to one detection per stride, keeping the last.
    picked = [len(cs) - 1]
    for i in range(len(cs) - 2, -1, -1):
        if cs[picked[-1]].frame - cs[i].frame >= stride:
            picked.append(i)
    picked.reverse()
    if len(picked) < 6:
        return track
    pts = np.array([[cs[i].x, cs[i].y] for i in picked])
    fr = np.array([cs[i].frame for i in picked], dtype=float)
    vel = np.diff(pts, axis=0) / np.diff(fr)[:, None]
    speed = np.linalg.norm(vel, axis=1)
    for k in range(max(3, len(vel) // 2), len(vel)):
        ref = vel[k - 2:k].mean(axis=0)
        ref_speed = float(np.linalg.norm(ref))
        if ref_speed < tcfg.impact_min_step_px or speed[k] < 1e-6:
            continue
        cos = float(ref @ vel[k]) / (ref_speed * speed[k])
        turned = cos < np.cos(np.radians(tcfg.impact_turn_deg))
        ratio = speed[k] / ref_speed
        # Slowing down gently is not a sign: a puck flying away from the
        # camera slows on screen all the way, by a quarter across a missed
        # frame.  Stopping short is: the mesh takes most of its speed at once.
        stopped = ratio < tcfg.impact_stop_ratio
        if turned or stopped or ratio > tcfg.impact_speed_jump:
            cut = picked[k]  # the last point before the break
            if near_goal is not None and not near_goal(pts[k]):
                continue
            if cut + 1 >= tcfg.min_track_length:
                return Track(list(cs[: cut + 1]))
            break
    return track


def _straight_part(track: Track, goal_width_px: float, cfg: Config, min_step: float = 0.0,
                   seen: dict | None = None) -> Track | None:
    """The clean flight inside a track that bends at either end.

    Growing a track outward follows the puck wherever it goes: before a wrist
    shot, along the ice as the blade drags it; after the impact, down the
    netting.  Those bends make the whole track fail the straight-line test,
    and the flight -- the part that matters -- would be thrown away with
    them.  So the longest unbroken stretch that passes is kept, the one
    nearest the net when two are as long.  Peeling one point at a time off
    whichever end looked worse was tried first; on a real shot bent at both
    ends it peeled the wrong end and kept a stretch that stopped halfway to
    the net.  Less than half the track is not the same flight, so nothing
    shorter is looked for.

    A long busy clip hands this hundreds of thousands of failed tracks, many
    of them the same wandering track regrown from another seed, so: tracks
    with no stretch fast enough to be a shot (``min_step`` px a frame) are
    skipped, ones already searched are remembered in ``seen``, and for each
    start the longest clean end is found by bisection.
    """
    cs = track.candidates
    n = len(cs)
    n_min = cfg.track.min_track_length
    lo = max(n_min, (n + 1) // 2)
    if n - 1 < lo:
        return None
    key = None
    if seen is not None:
        key = tuple(id(c) for c in cs)
        if key in seen:
            return seen[key]
    # Every stretch is judged in constant time from running sums -- a long
    # clip can hand this hundreds of thousands of failed tracks.
    p = np.array([[c.x, c.y] for c in cs], dtype=np.float64)
    fr = np.array([c.frame for c in cs], dtype=np.float64)
    step = np.diff(p, axis=0)
    mag = np.linalg.norm(step, axis=1)
    # No stretch of lo points covers more ground than the whole path, nor
    # fewer than lo - 1 frames: if even that is too slow, none would pass.
    if mag.sum() / (lo - 1) < min_step:
        return None
    good = mag > 1e-6
    unit = np.where(good[:, None], step / np.where(good, mag, 1.0)[:, None], 0.0)
    cu = np.vstack([[0.0, 0.0], np.cumsum(unit, axis=0)])            # over steps
    cg = np.concatenate([[0], np.cumsum(good)])
    c1 = np.vstack([[0.0, 0.0], np.cumsum(p, axis=0)])                 # over points
    cxx = np.concatenate([[0.0], np.cumsum(p[:, 0] ** 2)])
    cyy = np.concatenate([[0.0], np.cumsum(p[:, 1] ** 2)])
    cxy = np.concatenate([[0.0], np.cumsum(p[:, 0] * p[:, 1])])
    limit = cfg.track.max_line_residual_frac * goal_width_px

    def clean(a: int, b: int) -> bool:           # points a .. b-1
        if fr[b - 1] - fr[a] <= 0:
            return False
        overall = p[b - 1] - p[a]
        norm = float(np.hypot(*overall))
        count = int(cg[b - 1] - cg[a])
        if norm < 1e-6 or count == 0:
            return False
        if float((cu[b - 1] - cu[a]) @ overall) / (norm * count) < 0.90:
            return False
        m = b - a
        mx, my = (c1[b] - c1[a]) / m
        sxx = (cxx[b] - cxx[a]) - m * mx * mx
        syy = (cyy[b] - cyy[a]) - m * my * my
        sxy = (cxy[b] - cxy[a]) - m * mx * my
        half = 0.5 * (sxx + syy)
        low = half - np.sqrt(max(0.25 * (sxx - syy) ** 2 + sxy * sxy, 0.0))
        return float(np.sqrt(max(low, 0.0) / m)) <= limit

    best: tuple[int, int] | None = None
    for start in range(0, n - lo + 1):
        end = start + lo
        if not clean(start, end):
            continue
        top = min(n, start + n - 1)
        while end < top:
            mid = (end + top + 1) // 2
            if clean(start, mid):
                end = mid
            else:
                top = mid - 1
        if best is None or end - start >= best[1] - best[0]:
            best = (start, end)
    found = None
    if best is not None:
        t = Track(list(cs[best[0] : best[1]]))
        found = t if _accept(t, goal_width_px, cfg) else None   # the same test, spelled out
    if seen is not None:
        seen[key] = found
    return found


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


def clear_sightings(
    track: Track, raw_xy: dict[int, np.ndarray], radius_px: float, max_neighbours: int
) -> int:
    """How many of a track's detections were made in uncluttered surroundings.

    ``raw_xy`` holds every candidate found in each frame, before any cap, so
    it measures how busy the background really was there.
    """
    n = 0
    for c in track.candidates:
        pts = raw_xy.get(c.frame)
        if pts is None or not len(pts):
            n += 1
            continue
        d = np.hypot(pts[:, 0] - c.x, pts[:, 1] - c.y)
        n += int(((d > 1.0) & (d < radius_px)).sum() <= max_neighbours)
    return n


def revisits(track: Track, raw_xy: dict[int, np.ndarray], radius_px: float, window: int, gap: int) -> float:
    """How often something turns up again where a track's sightings were.

    The fraction of frames between ``gap`` and ``window`` either side of each
    sighting that have a candidate within ``radius_px`` of it.  A puck passes
    a spot once; a line of things that flicker in place -- lamps, a post, the
    edge of a house while the camera settles at the start of a clip -- keeps
    being seen where it is.  Unlike the recurring-clutter test, either side
    counts, so it still works in a clip's first and last moments.
    """
    total = 0
    for c in track.candidates:
        for df in range(gap, window + 1):
            for f in (c.frame - df, c.frame + df):
                pts = raw_xy.get(f)
                if pts is not None and len(pts):
                    total += bool((np.hypot(pts[:, 0] - c.x, pts[:, 1] - c.y) < radius_px).any())
    return total / max(len(track) * 2 * (window - gap + 1), 1)


def filter_by_clutter(
    tracks: list[Track], raw_xy: dict[int, np.ndarray], goal_width_px: float, fps: float, cfg: Config
) -> tuple[list[Track], int]:
    """Drop tracks seen almost only inside busy background.  Returns (kept, dropped)."""
    tcfg = cfg.track
    need = max(tcfg.min_clear_sightings, math.ceil(tcfg.min_clear_sightings_s * fps - 1e-9))
    if need <= 0:
        return tracks, 0
    radius = tcfg.clutter_radius_frac * goal_width_px
    kept = [t for t in tracks if clear_sightings(t, raw_xy, radius, tcfg.clutter_max_neighbours) >= need]
    return kept, len(tracks) - len(kept)
