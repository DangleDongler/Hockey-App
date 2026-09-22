"""Recovering the camera from the goal outline.

Once the goal's four corners are known, the goal is a rectangle of known size
lying on a plane — which is the classic single-plane calibration problem.  The
two constraints that the rotation's first two columns are orthonormal are
enough to solve for focal length, and from there for the camera's full pose.

That buys two things the homography alone cannot give:

* the camera's position in goal coordinates, so we can say how square-on the
  shot line it is, and
* the ability to project any 3-D point — in particular the spot the player is
  shooting from — into the image, which is how the release instant is pinned
  down instead of guessed.

World coordinates match :mod:`shottracker.geometry`, extended by ``z``, the
distance out in front of the goal plane toward the shooter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraModel:
    K: np.ndarray            # 3x3 intrinsics
    R: np.ndarray            # 3x3 rotation, world -> camera
    t: np.ndarray            # 3, translation, world -> camera
    focal_px: float
    residual_px: float       # reprojection error on the four goal corners
    focal_assumed: bool = False   # True when the view could not determine it

    @property
    def position(self) -> np.ndarray:
        """Camera centre in goal coordinates (inches)."""
        return -self.R.T @ self.t

    def project(self, pts_world: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
        cam = (self.R @ pts.T).T + self.t
        z = np.clip(cam[:, 2], 1e-6, None)
        img = (self.K @ (cam / z[:, None]).T).T
        return img[:, :2]

    def depth(self, pts_world: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
        return ((self.R @ pts.T).T + self.t)[:, 2]

    def shot_line_angle_deg(self, from_point: np.ndarray) -> float:
        """Angle between the camera's view direction and the shot line.

        0 means the camera is looking straight down the barrel, where image
        motion carries almost no speed information; 90 means fully side-on.
        """
        shot_dir = np.array([0.0, 0.0, -1.0])           # shooter -> net
        to_cam = self.position - np.asarray(from_point, dtype=np.float64)
        n = np.linalg.norm(to_cam)
        if n < 1e-9:
            return 0.0
        cos = abs(float(np.dot(shot_dir, to_cam / n)))
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def calibrate_from_homography(
    H: np.ndarray,
    image_size: tuple[int, int],
    corners_goal=None,
    corners_image=None,
    assumed_focal_px: float | None = None,
) -> CameraModel | None:
    """Solve for intrinsics and pose from a plane-to-image homography.

    ``H`` maps goal-plane (x, y) in inches to image pixels.  The principal
    point is assumed to be the image centre and pixels square, which is a good
    approximation for a phone camera and leaves focal length as the only
    unknown intrinsic.

    A view square-on to the net carries no focal-length information at all --
    the two orthonormality constraints degenerate.  Passing
    ``assumed_focal_px`` lets the solve continue on a typical phone lens; the
    pose is then only as good as that guess, and the model says so through
    ``focal_assumed``.  With no assumption to fall back on, this returns None,
    which is a statement that the geometry is not observable rather than a
    failure to try.
    """
    H = np.asarray(H, dtype=np.float64)
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0

    # Move the principal point to the origin so omega is diag(1/f^2, 1/f^2, 1).
    T = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    Hc = T @ H
    h1, h2 = Hc[:, 0], Hc[:, 1]

    # r1 . r2 = 0  and  |r1| = |r2|, written in terms of f^2.
    num_a = -(h1[0] * h2[0] + h1[1] * h2[1])
    den_a = h1[2] * h2[2]
    num_b = (h2[0] ** 2 + h2[1] ** 2) - (h1[0] ** 2 + h1[1] ** 2)
    den_b = h1[2] ** 2 - h2[2] ** 2

    candidates = []
    # Prefer the better-conditioned of the two constraints.
    for num, den, weight in ((num_a, den_a, abs(den_a)), (num_b, den_b, abs(den_b))):
        if abs(den) < 1e-12:
            continue
        f2 = num / den
        if f2 > 1.0:
            candidates.append((weight, float(np.sqrt(f2))))
    if not candidates and assumed_focal_px is None:
        return None
    candidates.sort(reverse=True)
    f = candidates[0][1] if candidates else 0.0
    focal_assumed = False

    # A phone lens is roughly 0.5-2.5 image widths of focal length.  Well
    # outside that, the fronto-parallel degeneracy has made the solve garbage.
    if not (0.25 * w <= f <= 6.0 * w):
        if assumed_focal_px is None:
            return None
        f, focal_assumed = float(assumed_focal_px), True

    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    Kinv = np.linalg.inv(K)
    M = Kinv @ H
    lam = 1.0 / np.linalg.norm(M[:, 0])
    r1 = lam * M[:, 0]
    r2 = lam * M[:, 1]
    t = lam * M[:, 2]
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)

    # Nearest true rotation.
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, 2] *= -1
        R = U @ Vt

    # Cheirality: the goal must be in front of the camera, and the camera must
    # be on the shooter's side of the goal plane.
    if (R @ np.array([0.0, 24.0, 0.0]) + t)[2] < 0:
        R, t = -R, -t
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, 2] *= -1
            R = U @ Vt
    cam_pos = -R.T @ t
    if cam_pos[2] < 0:
        return None

    residual = 0.0
    if corners_goal is not None and corners_image is not None:
        cg = np.asarray(corners_goal, dtype=np.float64)
        pts3d = np.hstack([cg, np.zeros((len(cg), 1))])
        model = CameraModel(K=K, R=R, t=t, focal_px=f, residual_px=0.0)
        proj = model.project(pts3d)
        residual = float(np.linalg.norm(proj - np.asarray(corners_image, dtype=np.float64), axis=1).mean())

    return CameraModel(K=K, R=R, t=t, focal_px=f, residual_px=residual, focal_assumed=focal_assumed)
