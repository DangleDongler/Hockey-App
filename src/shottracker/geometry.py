"""The goal-plane coordinate system.

Everything the tracker reports — where a shot hit, how fast it was going — is
anchored to a single plane: the vertical plane containing the goal mouth.

Goal coordinates are inches, with the origin on the ice at the centre of the
mouth, ``x`` increasing to image-right and ``y`` increasing upward.  So the
mouth is x in [-36, +36], y in [0, 48], and the four corners a coach yells
about live at the extremes.

A homography maps that plane to the image, which is all we need because the
goal really is planar and a puck's impact really does happen on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import GoalSpec

# Canonical corner order used everywhere: top-left, top-right, bottom-right,
# bottom-left, as seen in the image.
TL, TR, BR, BL = 0, 1, 2, 3


@dataclass(frozen=True)
class Zone:
    key: str
    label: str
    x0: float
    x1: float
    y0: float
    y1: float

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))


def build_zones(goal: GoalSpec) -> list[Zone]:
    """A 3x3 target grid over the mouth, named the way players name them."""
    w, h = goal.mouth_width_in, goal.mouth_height_in
    xs = [-w / 2, -w / 6, w / 6, w / 2]
    ys = [0.0, h / 3, 2 * h / 3, h]
    names = [
        # (row from top, col from left) -> key, label
        [("top_left", "Top Shelf Left"), ("top_mid", "Top Shelf Middle"), ("top_right", "Top Shelf Right")],
        [("mid_left", "Mid Left"), ("mid_mid", "Centre Mass"), ("mid_right", "Mid Right")],
        [("low_left", "Bottom Left Corner"), ("five_hole", "Five Hole"), ("low_right", "Bottom Right Corner")],
    ]
    zones: list[Zone] = []
    for row in range(3):
        y1 = ys[3 - row]
        y0 = ys[2 - row]
        for col in range(3):
            key, label = names[row][col]
            zones.append(Zone(key, label, xs[col], xs[col + 1], y0, y1))
    return zones


def zone_for(zones: list[Zone], x: float, y: float) -> Zone | None:
    for z in zones:
        if z.contains(x, y):
            return z
    # Points exactly on the top/right edge belong to the adjacent zone.
    for z in zones:
        if z.x0 <= x <= z.x1 and z.y0 <= y <= z.y1:
            return z
    return None


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Put four image points into TL, TR, BR, BL order.

    Sorting by angle around the centroid gives a consistent winding; we then
    rotate so the corner nearest the image origin leads.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)          # counter-clockwise in maths axes,
    pts = pts[order]                 # clockwise on screen (y grows downward)
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def outer_rect(goal: GoalSpec) -> np.ndarray:
    """The goal's outer pipe rectangle in goal coordinates, in TL/TR/BR/BL order."""
    hw = goal.outer_width_in / 2.0
    top = goal.outer_height_in
    return np.array(
        [[-hw, top], [hw, top], [hw, 0.0], [-hw, 0.0]],
        dtype=np.float64,
    )


def mouth_rect(goal: GoalSpec) -> np.ndarray:
    hw = goal.mouth_width_in / 2.0
    top = goal.mouth_height_in
    return np.array(
        [[-hw, top], [hw, top], [hw, 0.0], [-hw, 0.0]],
        dtype=np.float64,
    )


class GoalPlane:
    """A calibrated mapping between the goal plane and the image.

    Built from the four image corners of the goal's *outer* pipe rectangle,
    whose real dimensions are known, which is what fixes the scale.
    """

    def __init__(self, image_quad: np.ndarray, goal: GoalSpec | None = None):
        self.goal = goal or GoalSpec()
        self.image_quad = order_quad(image_quad)
        src = outer_rect(self.goal).astype(np.float32)
        dst = self.image_quad.astype(np.float32)
        self.H = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        self.H_inv = np.linalg.inv(self.H)

    # -- mapping ----------------------------------------------------------

    def to_image(self, pts_goal: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_goal, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def to_goal(self, pts_image: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_image, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H_inv)
        return out.reshape(-1, 2)

    # -- derived quantities ----------------------------------------------

    @property
    def mouth_image_quad(self) -> np.ndarray:
        return self.to_image(mouth_rect(self.goal))

    def px_per_inch_at(self, x: float, y: float) -> float:
        """Isotropic image scale at a point ON the goal plane.

        Used for sizing detection thresholds and for diagnostics; distances are
        measured by mapping endpoints, not by multiplying through this.
        """
        eps = 0.5
        p = self.to_image(np.array([[x, y], [x + eps, y], [x, y + eps]]))
        jx = (p[1] - p[0]) / eps
        jy = (p[2] - p[0]) / eps
        det = abs(jx[0] * jy[1] - jx[1] * jy[0])
        return float(np.sqrt(det))

    @property
    def quad_area_px(self) -> float:
        return float(abs(cv2.contourArea(self.image_quad.astype(np.float32))))

    @property
    def obliquity(self) -> float:
        """How far off-axis the camera is, in [0, 1).

        0 means the camera looks straight down the shot line, where a puck
        flying at the net barely moves in the image and its speed cannot be
        read from pixel motion.  Larger values mean a more side-on view, where
        it can.  Computed from how unequally perspective foreshortens the two
        posts.
        """
        q = self.image_quad
        left = float(np.linalg.norm(q[BL] - q[TL]))
        right = float(np.linalg.norm(q[BR] - q[TR]))
        lo, hi = min(left, right), max(left, right)
        if hi <= 1e-6:
            return 0.0
        return float(1.0 - lo / hi)

    def contains_image_point(self, pt, margin_in: float = 0.0) -> bool:
        g = self.to_goal(np.array([pt]))[0]
        hw = self.goal.mouth_width_in / 2.0 + margin_in
        return (-hw <= g[0] <= hw) and (-margin_in <= g[1] <= self.goal.mouth_height_in + margin_in)

    def is_plausible(self, min_aspect: float, max_aspect: float) -> bool:
        q = self.image_quad
        if not cv2.isContourConvex(q.astype(np.float32)):
            return False
        top = float(np.linalg.norm(q[TR] - q[TL]))
        bottom = float(np.linalg.norm(q[BR] - q[BL]))
        left = float(np.linalg.norm(q[BL] - q[TL]))
        right = float(np.linalg.norm(q[BR] - q[TR]))
        if min(top, bottom, left, right) < 4.0:
            return False
        width = 0.5 * (top + bottom)
        height = 0.5 * (left + right)
        if height <= 1e-6:
            return False
        aspect = width / height
        if not (min_aspect <= aspect <= max_aspect):
            return False
        # Perspective can shrink the far edge, but not arbitrarily.
        if max(top, bottom) / max(min(top, bottom), 1e-6) > 3.0:
            return False
        if max(left, right) / max(min(left, right), 1e-6) > 3.0:
            return False
        return True


def scale_quad(quad: np.ndarray, factor: float) -> np.ndarray:
    return np.asarray(quad, dtype=np.float64) * float(factor)


def quad_diagonal(quad: np.ndarray) -> float:
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    return float(max(np.linalg.norm(q[0] - q[2]), np.linalg.norm(q[1] - q[3])))
