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


def _colour_quad(
    small: np.ndarray, cfg: Config, rng: np.random.Generator
) -> tuple[np.ndarray, float, str] | None:
    """The original method: segment saturated red pipe and fit its edges."""
    ncfg = cfg.net
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
    if not _plausible(quad, small, cfg):
        return None
    return quad, float(np.clip(support, 0.0, 1.0)), method


def _plausible(quad: np.ndarray, small: np.ndarray, cfg: Config) -> bool:
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    if area < cfg.net.min_quad_area_frac * small.shape[0] * small.shape[1]:
        return False
    return GoalPlane(quad, cfg.goal).is_plausible(cfg.net.min_aspect, cfg.net.max_aspect)


# --- finding the goal by its shape ------------------------------------------


@dataclass
class _Post:
    x0: int
    x1: int
    y0: int
    y1: int
    xs: np.ndarray
    ys: np.ndarray

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1


def _post_mask(small: np.ndarray, ncfg: NetDetectConfig) -> np.ndarray:
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(
        hsv, (0, ncfg.post_low_hue_sat_min, ncfg.post_val_min), (ncfg.post_hue_low_max, 255, 255)
    )
    hi = cv2.inRange(hsv, (ncfg.post_hue_high_min, ncfg.post_sat_min, ncfg.post_val_min), (179, 255, 255))
    return cv2.bitwise_or(lo, hi)


def find_posts(small: np.ndarray, ncfg: NetDetectConfig) -> list[_Post]:
    """Thin upright reddish bars: anything that could be a goal post.

    A post is many times taller than it is wide.  Legs, shrubs and jackets
    can be the same colour but none of them are that shape, which is why the
    colour band here can afford to be loose.
    """
    H, W = small.shape[:2]
    mask = _post_mask(small, ncfg)
    k = max(9, int(round(0.025 * H)))
    tall = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, tall)
    # Net mesh crosses the posts and breaks them into pieces; bridge that.
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, tall)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(vert, connectivity=8)
    posts: list[_Post] = []
    min_h = max(20.0, ncfg.post_min_height_frac * H)
    max_w = max(3.0, ncfg.post_max_width_frac * W)
    for i in range(1, num):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if h < min_h:
            continue
        sub = labels[y : y + h, x : x + w] == i
        # Judge thinness row by row and keep the longest thin stretch.  Red
        # flowers resting on top of a post, or a shadow at its foot, fuse with
        # it into one blob that is wide somewhere; the post is still the long
        # run of narrow rows inside it.
        row_w = sub.sum(axis=1)
        thin = (row_w > 0) & (row_w <= max_w)
        cols = np.arange(sub.shape[1])
        centre = np.where(row_w > 0, (sub * cols).sum(axis=1) / np.maximum(row_w, 1), np.nan)
        # A post is also straight: a row whose centre jumps sideways has left
        # the post for whatever is touching it, even if that row is narrow.
        best_len, best_end, run = 0, -1, 0
        prev = np.nan
        for r, t in enumerate(thin):
            straight = np.isnan(prev) or abs(centre[r] - prev) <= 2.5
            if t and straight:
                run += 1
            elif t:
                run = 1
            else:
                run = 0
            prev = centre[r] if t else np.nan
            if run > best_len:
                best_len, best_end = run, r
        if best_len < min_h:
            continue
        r0, r1 = best_end - best_len + 1, best_end
        ys, xs = np.nonzero(sub[r0 : r1 + 1])
        ys = ys + r0
        width = float(np.median(row_w[r0 : r1 + 1]))
        if best_len / max(width, 1.0) < ncfg.post_min_aspect:
            continue
        px0, px1 = x + int(xs.min()), x + int(xs.max())
        if px0 <= 1 or px1 >= W - 2:
            continue  # the frame edge: a door jamb or a wall corner, not a post
        posts.append(_Post(px0, px1, y + r0, y + r1, xs + x, ys + y))
    return posts


def _edge_line(post: _Post, side: str, rng: np.random.Generator):
    """Line along the outer edge of a post: leftmost pixel per row, or rightmost."""
    edge = []
    for ry in np.unique(post.ys):
        row = post.xs[post.ys == ry]
        edge.append((row.min() if side == "left" else row.max(), ry))
    return _fit_line_ransac(np.array(edge, dtype=np.float64), thresh=1.5, rng=rng)


def _point_on_line_at_y(line, y: float) -> np.ndarray | None:
    o, d, _ = line
    if abs(d[1]) < 1e-9:
        return None
    t = (y - o[1]) / d[1]
    return o + t * d


def _ridge_image(gray: np.ndarray, pipe_px: float) -> np.ndarray:
    """How much each pixel looks like the middle of a horizontal bar.

    Centre band minus the rows either side: positive for a bar lighter than
    what surrounds it (a white crossbar seen through dark mesh), negative for
    one darker (red pipe against bright ice).
    """
    g = gray.astype(np.float32)
    half = max(1, int(round(pipe_px / 2.0)))
    centre = cv2.blur(g, (1, 2 * half + 1))
    off = max(2, int(round(0.9 * pipe_px)))
    up = np.roll(g, off, axis=0)
    down = np.roll(g, -off, axis=0)
    return centre - 0.5 * (up + down)


def _line_score(ridge: np.ndarray, p0: np.ndarray, p1: np.ndarray, n: int = 120) -> tuple[float, float]:
    """Continuity-weighted ridge strength along a segment.

    A crossbar runs the whole way from post to post.  Mesh strands sag and
    wander, so they never keep the same sign over the full span; the bar does.
    Returns (score, fraction of the span that agrees).
    """
    xs = np.linspace(p0[0], p1[0], n).astype(np.float32).reshape(1, -1)
    ys = np.linspace(p0[1], p1[1], n).astype(np.float32).reshape(1, -1)
    vals = cv2.remap(ridge, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
    med = float(np.median(vals))
    if abs(med) < 1e-6:
        return 0.0, 0.0
    agree = float(((np.sign(vals) == np.sign(med)) & (np.abs(vals) > 6.0)).mean())
    return abs(med) * agree, agree


def _find_crossbar(
    gray: np.ndarray,
    ll,
    rl,
    bl: np.ndarray,
    br: np.ndarray,
    seen: tuple[float, float],
    pipe_px: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Locate the crossbar's top edge between two posts.

    Heights are measured up each post from its foot.  Two directions for the
    bar are tried: parallel to the ground line, which is right square-on and
    survives one post's top being hidden; and through the two visible post
    tops, which is right from the side, where perspective tips the bar well
    away from the ground line.  In each, the bar is the *first* clear straight
    line met going up; the best candidate is then refined with its two ends
    free.  Returns the (left, right) top-edge points, or None if nothing
    bar-like spans the gap.
    """
    ridge = _ridge_image(gray, pipe_px)
    seen_l, seen_r = seen
    tallest = max(seen_l, seen_r)

    def ends(c_left: float, c_right: float) -> tuple[np.ndarray, np.ndarray] | None:
        a = _point_on_line_at_y(ll, bl[1] - c_left)
        b = _point_on_line_at_y(rl, br[1] - c_right)
        if a is None or b is None:
            return None
        return a, b

    def score(c_left: float, c_right: float) -> tuple[float, float]:
        e = ends(c_left, c_right)
        if e is None:
            return 0.0, 0.0
        a, b = e
        # Keep clear of the posts themselves, and of the bends in the corners.
        return _line_score(ridge, a + 0.12 * (b - a), a + 0.88 * (b - a))

    def first_peak(offset_l: float, offset_r: float, lo: float, hi: float):
        """Shift a line of fixed direction upward; return its lowest clear peak."""
        shifts = np.arange(lo, hi, 1.0)
        vals = np.zeros(len(shifts))
        for k, t in enumerate(shifts):
            sc, agree = score(offset_l + t, offset_r + t)
            vals[k] = sc if agree >= 0.6 else 0.0
        if not len(vals) or vals.max() <= 0:
            return None
        # The crossbar is the first bar met going up from the posts.  Anything
        # higher -- the top of a backstop frame, a fence rail, a roofline -- can
        # be straighter and brighter, which is why the strongest line is the
        # wrong choice.
        floor = 0.5 * float(vals.max())
        for k in range(len(vals)):
            if vals[k] >= floor and (k == 0 or vals[k] >= vals[k - 1]) and (
                k == len(vals) - 1 or vals[k] >= vals[k + 1]
            ):
                t = float(shifts[k])
                return float(vals[k]), (offset_l + t, offset_r + t)
        return None

    found = []
    parallel = first_peak(0.0, 0.0, 0.85 * tallest, 2.0 * tallest)
    if parallel:
        found.append(parallel)
    through_tops = first_peak(seen_l, seen_r, -0.15 * tallest, 1.0 * tallest)
    if through_tops:
        found.append(through_tops)
    if not found:
        return None

    best_sc, start = max(found, key=lambda f: f[0])
    best, best_agree = start, score(*start)[1]
    span = max(2.0, 0.10 * max(start))
    for dl in np.arange(-span, span + 0.5, 1.0):
        for dr in np.arange(-span, span + 0.5, 1.0):
            cand = (start[0] + dl, start[1] + dr)
            sc, agree = score(*cand)
            if agree >= 0.6 and sc > best_sc:
                best_sc, best_agree, best = sc, agree, cand
    if best_agree < 0.6:
        return None
    c_l, c_r = best

    # The ridge sits on the bar's centreline; the outline wants its top edge.
    half = pipe_px / 2.0
    return ends(c_l + half, c_r + half)


def _quad_from_posts(
    left: _Post,
    right: _Post,
    rng: np.random.Generator,
    gray: np.ndarray | None = None,
    pipe_frac: float = 2.375 / 76.75,
) -> np.ndarray | None:
    """Outer corners from two posts and the crossbar joining them.

    The feet are reliable -- the posts stand on the ground.  The tops often
    are not: a hanging target, glare off the bend or the mesh can hide the
    last stretch of a post.  So the top edge comes from the crossbar itself,
    found as a straight bar spanning post to post, and with no such bar there
    is no goal to report.
    """
    ll = _edge_line(left, "left", rng)
    rl = _edge_line(right, "right", rng)
    if ll is None or rl is None:
        return None

    bl = _point_on_line_at_y(ll, left.y1)
    br = _point_on_line_at_y(rl, right.y1)
    if bl is None or br is None:
        return None

    tops = None
    if gray is not None:
        # How thick the pipe looks follows from the goal's own proportions: it
        # is 2 3/8" on a frame 76 3/4" wide.  The posts' segmented width is no
        # guide -- blending with the mesh fattens them by a third or more.
        pipe_px = max(2.0, float(np.linalg.norm(br - bl)) * pipe_frac)
        seen = (float(bl[1] - left.y0), float(br[1] - right.y0))
        tops = _find_crossbar(gray, ll, rl, bl, br, seen, pipe_px)

    if tops is None:
        # Two upright bars with nothing spanning their tops are not a goal --
        # or at least not one whose height can be read, which comes to the same.
        return None

    quad = np.array([tops[0], tops[1], br, bl], dtype=np.float64)
    return quad if np.all(np.isfinite(quad)) else None


def _contains(outer: np.ndarray, inner: np.ndarray, slack: float = 2.0) -> bool:
    poly = outer.astype(np.float32).reshape(-1, 1, 2)
    return all(cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -slack for x, y in inner)


def _post_pair_quad(
    small: np.ndarray, cfg: Config, rng: np.random.Generator
) -> tuple[np.ndarray, float, str] | None:
    """Find the goal as two posts whose feet line up and whose spacing fits."""
    ncfg = cfg.net
    posts = find_posts(small, ncfg)
    if len(posts) < 2:
        return None
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    goal = cfg.goal
    head_on = goal.outer_width_in / goal.outer_height_in

    candidates: list[tuple[float, np.ndarray]] = []
    for i, a in enumerate(posts):
        for b in posts[i + 1 :]:
            left, right = (a, b) if a.x0 < b.x0 else (b, a)
            if right.x0 - left.x1 < 3:
                continue
            tall = max(left.height, right.height)
            short = min(left.height, right.height)
            # Both posts stand on the same ground.
            if abs(left.y1 - right.y1) > 0.2 * tall:
                continue
            # One may be partly hidden, but not mostly.
            if short / tall < 0.5:
                continue
            ratio = (right.x1 - left.x0) / tall
            if not (0.5 <= ratio <= 3.0):
                continue
            quad = _quad_from_posts(left, right, rng, gray, goal.post_diameter_in / goal.outer_width_in)
            if quad is None:
                continue
            quad = order_quad(quad)
            if not _plausible(quad, small, cfg):
                continue
            # Prefer matched posts with level feet and head-on-ish proportions.
            score = (
                (short / tall)
                * (1.0 - abs(left.y1 - right.y1) / (0.2 * tall) * 0.5)
                * float(np.exp(-0.5 * (np.log(ratio / head_on) / 0.45) ** 2))
            )
            candidates.append((score, quad))

    if not candidates:
        return None

    # A goal in a backyard often sits inside a backstop frame of the same pipe.
    # The goal is always the inner rectangle, never the outer one, and taking
    # the outer by mistake would scale every result down by the difference.
    inner = [
        (sc, q) for sc, q in candidates
        if not any(_contains(q, other) for _, other in candidates if other is not q)
    ]
    score, quad = max(inner or candidates, key=lambda c: c[0])
    # This method infers the crossbar rather than seeing it, so it never claims
    # the confidence the colour method earns on clean red pipe.
    return quad, float(np.clip(0.4 + 0.45 * score, 0.0, 0.85)), "posts"


def detect_net_in_frame(
    bgr: np.ndarray, cfg: Config, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, float, str] | None:
    """Find the goal outline in a single full-resolution frame.

    Tries the colour method first; when that has not clearly locked onto red
    pipe -- a faded, shaded or washed-out frame -- falls back to finding the
    goal by its shape.  Returns (quad in full-res pixels, confidence, method)
    or None.
    """
    ncfg = cfg.net
    rng = rng or np.random.default_rng(7)
    h, w = bgr.shape[:2]
    scale = min(1.0, ncfg.work_width / float(w))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr

    if ncfg.method == "posts":
        best = _post_pair_quad(small, cfg, rng)
    elif ncfg.method == "colour":
        best = _colour_quad(small, cfg, rng)
    else:
        colour = _colour_quad(small, cfg, rng)
        if colour is not None and colour[2] == "pipe_edges" and colour[1] >= ncfg.colour_method_trust:
            best = colour
        else:
            best = _post_pair_quad(small, cfg, rng) or colour
    if best is None:
        return None
    quad, conf, method = best
    return quad / scale, conf, method


def _largest_group(points: np.ndarray, tol: float) -> np.ndarray:
    """Indices of the biggest set of readings that agree with one of them.

    ``points`` is (n, k, 2): k corners per reading.  Two readings agree when
    every corner is within ``tol`` pixels.  Voting like this, rather than
    comparing everything to the median, still finds the majority when the
    readings split two ways, where a median can land between the camps and
    agree with neither.
    """
    d = np.linalg.norm(points[:, None] - points[None, :], axis=3).max(axis=2)
    agree = d <= tol
    return np.nonzero(agree[int(np.argmax(agree.sum(axis=1)))])[0]


def consensus_quad(
    quads: list[np.ndarray], tol_frac: float
) -> tuple[np.ndarray, list[int], list[int]] | None:
    """One outline from many frames' readings of it.

    Settled in two steps, because the goal's two edges go wrong in different
    ways.  The posts' feet stand on the ground and are found the same way in
    every frame, so they say whether the camera held still.  The crossbar is
    the edge that gets misread -- a band of netting just under it can pass
    for its edge in some frames -- so among the frames where the feet agree,
    it is settled by the reading most of them share.

    Returns (outline, frames used for it, frames in which the camera was
    where it is for most of the clip).
    """
    if not quads:
        return None
    stack = np.stack([order_quad(q) for q in quads])  # (n, 4, 2): TL, TR, BR, BL
    tol = max(tol_frac * quad_diagonal(np.median(stack, axis=0)), 2.0)
    steady = _largest_group(stack[:, 2:], tol)
    keep = steady[_largest_group(stack[steady, :2], tol)]
    refined = np.median(stack[keep], axis=0)
    return refined, keep.tolist(), steady.tolist()


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
    quad, keep, steady = agreed

    n_used, n_steady = len(keep), len(steady)
    # Moved means the frames that found the goal disagree about where it is.
    # Frames that found nothing -- the shooter in the way, a thin pipe lost in
    # glare -- say nothing about the camera.
    if n_used >= cfg.net.min_agreeing_frames and n_steady < cfg.net.min_agreement_frac * len(quads):
        notes.append(
            f"the goal was found, but only {n_steady} of the {len(quads)} frames it was found in put it in "
            "the same place, so no one outline fits the whole clip -- the camera most likely moved. Propping "
            "the phone fixes it; meanwhile the net can be marked on a frame near the shots that matter."
        )
        return None
    if n_used < cfg.net.min_agreeing_frames:
        notes.append(
            f"only {n_used} of {len(frames)} sampled frames agreed on where the goal is, so no "
            "outline could be trusted. Usually this means the camera moved during the clip, or the "
            "goal's frame is too dark or washed out to pick out from the background. "
            "Marking the four corners by hand solves it for good on a net that does not move."
        )
        return None
    if n_used < cfg.net.min_agreement_frac * n_steady:
        notes.append(
            f"the crossbar was hard to pick out: {n_used} of {n_steady} frames agreed on where its top "
            "edge is. Check the outline sits on the top of the bar; if not, mark the corners by hand."
        )

    method_counts: dict[str, int] = {}
    for i in keep or range(len(methods)):
        method_counts[methods[i]] = method_counts.get(methods[i], 0) + 1
    method = max(method_counts, key=method_counts.get) if method_counts else methods[0]
    if method == "pipe_hull":
        notes.append("fell back to hull fitting; the outline may sit slightly outside the pipe")

    base_conf = float(np.mean([confs[i] for i in keep])) if keep else float(np.mean(confs))
    # How much of the clip the outline holds for, discounted by how
    # unanimous the crossbar reading was.
    agreement = n_steady / float(max(len(frames), 1)) * (0.5 + 0.5 * n_used / float(max(n_steady, 1)))
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
