"""Automatic detection of the goal outline.

A hockey goal is the one thing in the frame that is bright red, rigid and
shaped like an upside-down U standing on the ice.  We lean on all three:

1.  Segment the red pipe in HSV (red straddles the hue wrap, so two bands).
2.  Recover the structure: the *outer* edge of each post and the *top* edge of
    the crossbar, each fitted with RANSAC so that mesh, skate marks and the
    net's rearward skirt do not drag the lines around.
3.  Intersect those lines to get the outer corners, and find each post's foot.

The goal does not move, so we do this on frames sampled across the whole clip
and keep the median of the ones that agree.  A single bad frame — a player
standing in front of the post, a camera flash — cannot move the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Config, NetDetectConfig
from .geometry import GoalPlane, order_quad, quad_diagonal


@dataclass
class NetDetection:
    quad: np.ndarray                  # 4x2 outer-pipe corners, full-res image px
    confidence: float                 # 0..1
    method: str                       # how it was found
    frames_used: int = 0
    frames_tried: int = 0
    notes: list[str] = field(default_factory=list)

    def plane(self, cfg: Config) -> GoalPlane:
        return GoalPlane(self.quad, cfg.goal)

    def to_dict(self) -> dict:
        return {
            "quad": np.asarray(self.quad, dtype=float).round(2).tolist(),
            "confidence": round(float(self.confidence), 3),
            "method": self.method,
            "frames_used": self.frames_used,
            "frames_tried": self.frames_tried,
            "notes": list(self.notes),
        }


# --- small geometric helpers ------------------------------------------------


def _fit_line_ransac(
    pts: np.ndarray, iters: int = 80, thresh: float = 2.5, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit a line to 2-D points, ignoring outliers.

    Returns (origin, unit_direction, inlier_mask).
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return None
    if len(pts) == 2:
        d = pts[1] - pts[0]
        n = np.linalg.norm(d)
        if n < 1e-9:
            return None
        return pts[0], d / n, np.ones(2, dtype=bool)

    rng = rng or np.random.default_rng(12345)
    best_inliers = None
    best_count = -1
    n = len(pts)
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        d = pts[j] - pts[i]
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            continue
        d = d / norm
        normal = np.array([-d[1], d[0]])
        dist = np.abs((pts - pts[i]) @ normal)
        inliers = dist < thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_inliers is None or best_count < 2:
        return None

    # Polish with a total-least-squares fit over the inliers.
    inlier_pts = pts[best_inliers].astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64), best_inliers


def _intersect(o1: np.ndarray, d1: np.ndarray, o2: np.ndarray, d2: np.ndarray) -> np.ndarray | None:
    """Intersection of two lines given in point-direction form."""
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float64)
    det = np.linalg.det(a)
    if abs(det) < 1e-9:
        return None
    t = np.linalg.solve(a, (o2 - o1))
    return o1 + t[0] * d1


def red_pipe_mask(bgr: np.ndarray, cfg: NetDetectConfig) -> np.ndarray:
    """Binary mask of goal-red pixels, closed into continuous pipe."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, (0, cfg.sat_min, cfg.val_min), (cfg.hue_low_max, 255, 255))
    hi = cv2.inRange(hsv, (cfg.hue_high_min, cfg.sat_min, cfg.val_min), (179, 255, 255))
    mask = cv2.bitwise_or(lo, hi)
    k = cfg.close_kernel | 1  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def _largest_frame_component(mask: np.ndarray, cfg: NetDetectConfig) -> np.ndarray | None:
    """Keep the connected component most likely to be the goal frame."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None
    frame_area = mask.shape[0] * mask.shape[1]
    best, best_score = None, 0.0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < cfg.min_pipe_area_frac * frame_area:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w < 12 or h < 8:
            continue
        # A goal is roughly 1.5 : 1.  Rink markings are the main false
        # positive and they are ruler-thin, so require real vertical extent.
        if h < 0.18 * w or w < 0.35 * h:
            continue
        # The frame is wide, hollow and spans a lot of the scene: reward
        # bounding-box coverage, penalise solid blobs (a jersey, a bench).
        fill = area / float(max(w * h, 1))
        if fill > cfg.max_fill:
            continue
        score = (w * h) * (1.0 - min(fill, 0.95))
        if score > best_score:
            best_score, best = score, i
    if best is None:
        return None
    return (labels == best).astype(np.uint8) * 255


def _structural_quad(comp: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float] | None:
    """Recover the outer corners from post and crossbar edges."""
    ys, xs = np.nonzero(comp)
    if len(xs) < 50:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if w < 20 or h < 15:
        return None

    tol = max(2.0, 0.006 * max(w, h))

    # Crossbar: topmost pipe pixel in each column, across the middle span so
    # the post shoulders do not dominate the fit.
    cols = np.arange(x0, x1 + 1)
    top_pts = []
    for cx in cols:
        col = np.nonzero(comp[:, cx])[0]
        if len(col):
            top_pts.append((cx, col[0]))
    if len(top_pts) < 10:
        return None
    top_pts = np.array(top_pts, dtype=np.float64)
    inner = (top_pts[:, 0] > x0 + 0.12 * w) & (top_pts[:, 0] < x1 - 0.12 * w)
    crossbar = _fit_line_ransac(top_pts[inner] if inner.sum() >= 10 else top_pts, thresh=tol, rng=rng)
    if crossbar is None:
        return None

    # Posts: outer-most pipe pixel in each row, left half and right half.
    mid = 0.5 * (x0 + x1)
    left_pts, right_pts = [], []
    for ry in range(y0, y1 + 1):
        row = np.nonzero(comp[ry, :])[0]
        if not len(row):
            continue
        left_pts.append((row[0], ry))
        right_pts.append((row[-1], ry))
    if len(left_pts) < 10 or len(right_pts) < 10:
        return None
    left_pts = np.array(left_pts, dtype=np.float64)
    right_pts = np.array(right_pts, dtype=np.float64)
    # Drop rows whose "outer" pixel is really on the other side of the goal
    # (happens above the crossbar, or where the frame is broken up).
    left_pts = left_pts[left_pts[:, 0] < mid]
    right_pts = right_pts[right_pts[:, 0] > mid]
    if len(left_pts) < 8 or len(right_pts) < 8:
        return None

    left = _fit_line_ransac(left_pts, thresh=tol, rng=rng)
    right = _fit_line_ransac(right_pts, thresh=tol, rng=rng)
    if left is None or right is None:
        return None

    tl = _intersect(left[0], left[1], crossbar[0], crossbar[1])
    tr = _intersect(right[0], right[1], crossbar[0], crossbar[1])
    if tl is None or tr is None:
        return None

    # Feet: the lowest pipe pixel that still sits on each post line.
    def foot(line, pts) -> np.ndarray | None:
        o, d, _ = line
        normal = np.array([-d[1], d[0]])
        dist = np.abs((pts - o) @ normal)
        on_line = pts[dist < max(tol * 3.0, 4.0)]
        if len(on_line) < 3:
            return None
        lowest = on_line[np.argmax(on_line[:, 1])]
        # Project it back onto the fitted line so the quad stays rectilinear.
        t = float((lowest - o) @ d)
        return o + t * d

    bl = foot(left, left_pts)
    br = foot(right, right_pts)
    if bl is None or br is None:
        return None

    quad = np.array([tl, tr, br, bl], dtype=np.float64)
    if not np.all(np.isfinite(quad)):
        return None

    # Confidence: how much of each edge was actually supported by pipe pixels.
    support = float(min(left[2].mean(), right[2].mean(), crossbar[2].mean()))
    return quad, support


def _hull_quad(comp: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Fallback: the convex hull of the pipe, squeezed down to four corners."""
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    pts = np.vstack(cnts).reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.12, 24):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float64), 0.55
    rect = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float64)
    return rect, 0.35


def detect_net_in_frame(
    bgr: np.ndarray, cfg: Config, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, float, str] | None:
    """Find the goal outline in a single full-resolution frame.

    Returns (quad in full-res pixels, confidence, method) or None.
    """
    ncfg = cfg.net
    rng = rng or np.random.default_rng(7)
    h, w = bgr.shape[:2]
    scale = min(1.0, ncfg.work_width / float(w))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr

    mask = red_pipe_mask(small, ncfg)
    comp = _largest_frame_component(mask, ncfg)
    if comp is None:
        return None

    result = _structural_quad(comp, rng)
    method = "pipe_edges"
    if result is None:
        result = _hull_quad(comp)
        method = "pipe_hull"
    if result is None:
        return None

    quad, support = result
    if method == "pipe_edges" and support < ncfg.min_edge_support:
        result = _hull_quad(comp)
        if result is None:
            return None
        quad, support = result
        method = "pipe_hull"
    quad = order_quad(quad)

    area = abs(cv2.contourArea(quad.astype(np.float32)))
    if area < ncfg.min_quad_area_frac * small.shape[0] * small.shape[1]:
        return None

    plane = GoalPlane(quad, cfg.goal)
    if not plane.is_plausible(ncfg.min_aspect, ncfg.max_aspect):
        return None

    return quad / scale, float(np.clip(support, 0.0, 1.0)), method


def consensus_quad(
    quads: list[np.ndarray], tol_frac: float
) -> tuple[np.ndarray, list[int]] | None:
    """Median outline over frames, keeping only the frames that agree with it."""
    if not quads:
        return None
    stack = np.stack([order_quad(q) for q in quads])  # (n, 4, 2)
    median = np.median(stack, axis=0)
    tol = tol_frac * quad_diagonal(median)
    err = np.linalg.norm(stack - median[None], axis=2).max(axis=1)
    keep = np.nonzero(err <= max(tol, 2.0))[0]
    if len(keep) == 0:
        return median, []
    refined = np.median(stack[keep], axis=0)
    return refined, keep.tolist()


def detect_net(
    frames: list[np.ndarray], cfg: Config, notes: list[str] | None = None
) -> NetDetection | None:
    """Vote on one static goal outline using frames sampled across the clip.

    Returns None when the evidence is too weak to name an outline.  Anything
    appended to ``notes`` explains why, because "I could not find it" is only
    useful to a player if it also says what to change.
    """
    notes = notes if notes is not None else []
    rng = np.random.default_rng(7)
    quads: list[np.ndarray] = []
    confs: list[float] = []
    methods: list[str] = []

    for f in frames:
        got = detect_net_in_frame(f, cfg, rng)
        if got is None:
            continue
        q, c, m = got
        quads.append(q)
        confs.append(c)
        methods.append(m)

    if not quads:
        notes.append(
            "nothing in the clip looked like a goal frame. The detector keys on the red pipe, so a "
            "goal in deep shade, a faded or non-red frame, or one hidden behind a shooter tutor will "
            "not be found. Mark the corners by hand instead."
        )
        return None

    agreed = consensus_quad(quads, cfg.net.consensus_tol_frac)
    if agreed is None:
        notes.append("the candidate outlines did not agree well enough to combine")
        return None
    quad, keep = agreed

    n_used = len(keep)
    if n_used < cfg.net.min_agreeing_frames:
        notes.append(
            f"only {n_used} of {len(frames)} sampled frames agreed on where the goal is, so no "
            "outline could be trusted. Usually this means the camera moved during the clip, or the "
            "goal's frame is too dark or washed out to pick out from the background. "
            "Marking the four corners by hand solves it for good on a net that does not move."
        )
        return None

    method_counts: dict[str, int] = {}
    for i in keep or range(len(methods)):
        method_counts[methods[i]] = method_counts.get(methods[i], 0) + 1
    method = max(method_counts, key=method_counts.get) if method_counts else methods[0]
    if method == "pipe_hull":
        notes.append("fell back to hull fitting; the outline may sit slightly outside the pipe")

    base_conf = float(np.mean([confs[i] for i in keep])) if keep else float(np.mean(confs))
    agreement = n_used / float(max(len(frames), 1))
    confidence = float(np.clip(0.35 * base_conf + 0.65 * agreement, 0.0, 1.0))

    plane = GoalPlane(quad, cfg.goal)
    if plane.obliquity < 0.06:
        notes.append(
            "camera is nearly square to the net; pixel-motion speed will be unreliable, "
            "prefer time-of-flight with a known shot distance"
        )

    if confidence < cfg.net.min_confidence:
        notes.append(
            f"a possible goal outline was found but only at {confidence:.0%} confidence, which is too "
            "low to build measurements on -- a red jacket, a flowering shrub or a rink marking can all "
            "look like goal pipe. Nothing was reported rather than something wrong. "
            "Marking the corners by hand is the reliable fix."
        )
        return None

    return NetDetection(
        quad=order_quad(quad),
        confidence=confidence,
        method=method,
        frames_used=n_used,
        frames_tried=len(frames),
        notes=[n for n in notes],
    )
