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

import cv2
import numpy as np

# The goal outline is located to about a pixel.  A focal length that swings
# more than this, as a fraction of itself, when the corners are jittered by
# that much was never really measured.
FOCAL_STABILITY_TOL = 0.15
FOCAL_JITTER_PX = 1.0
FOCAL_JITTER_SAMPLES = 24


@dataclass
class CameraModel:
    K: np.ndarray            # 3x3 intrinsics
    R: np.ndarray            # 3x3 rotation, world -> camera
    t: np.ndarray            # 3, translation, world -> camera
    focal_px: float
    residual_px: float       # reprojection error on the four goal corners
    focal_assumed: bool = False   # True when the view could not determine it
    focal_spread: float = 0.0     # relative scatter of the focal solve, 0 when assumed
    # Where the focal length came from: "goal" (solved from the outline),
    # "known" (the phone's own record, or the player's), or "assumed".
    focal_source: str = "goal"

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


    def flight_view_angle_deg(self, release: np.ndarray, target: np.ndarray) -> float:
        """How obliquely the camera sees the flight from ``release`` to ``target``.

        The angle between the line of sight to the middle of the flight and
        the flight itself: near 0 from straight behind the shooter, where the
        puck's progress toward the net barely shows, 90 from the side.
        """
        release, target = np.asarray(release, dtype=np.float64), np.asarray(target, dtype=np.float64)
        flight = target - release
        sight = (release + target) / 2.0 - self.position
        n = float(np.linalg.norm(flight) * np.linalg.norm(sight))
        if n < 1e-9:
            return 0.0
        cos = abs(float(np.dot(flight, sight))) / n
        return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


def _focal_from_constraints(H: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float]:
    """Focal length implied by each of the two orthonormality constraints.

    Returns (from r1.r2 = 0, from |r1| = |r2|); either may be nan when its
    denominator vanishes, which is what happens as the view goes
    fronto-parallel.
    """
    w, h = image_size
    T = np.array([[1.0, 0.0, -w / 2.0], [0.0, 1.0, -h / 2.0], [0.0, 0.0, 1.0]])
    Hc = T @ np.asarray(H, dtype=np.float64)
    h1, h2 = Hc[:, 0], Hc[:, 1]

    pairs = (
        (-(h1[0] * h2[0] + h1[1] * h2[1]), h1[2] * h2[2]),
        ((h2[0] ** 2 + h2[1] ** 2) - (h1[0] ** 2 + h1[1] ** 2), h1[2] ** 2 - h2[2] ** 2),
    )
    out = []
    for num, den in pairs:
        if abs(den) < 1e-12:
            out.append(float("nan"))
            continue
        f2 = num / den
        out.append(float(np.sqrt(f2)) if f2 > 1.0 else float("nan"))
    return out[0], out[1]


def _stable_focal(
    corners_goal: np.ndarray,
    corners_image: np.ndarray,
    image_size: tuple[int, int],
    jitter_px: float = FOCAL_JITTER_PX,
) -> tuple[float, float] | None:
    """Pick whichever constraint survives jittering the corners, if either does.

    Neither constraint is reliable in general: the first is unstable at almost
    every realistic camera position, and the second is undefined when the view
    is square-on.  Rather than pick one analytically, perturb the marked
    corners by the precision they are actually known to and keep the estimate
    that barely moves.  Returns (focal, relative spread).
    """
    src = np.asarray(corners_goal, dtype=np.float32)
    dst = np.asarray(corners_image, dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        return None

    rng = np.random.default_rng(11)
    samples: list[list[float]] = [[], []]
    for i in range(FOCAL_JITTER_SAMPLES + 1):
        jitter = 0.0 if i == 0 else rng.normal(0.0, jitter_px, dst.shape)
        try:
            H = cv2.getPerspectiveTransform(src, (dst + jitter).astype(np.float32))
        except cv2.error:
            continue
        fa, fb = _focal_from_constraints(H, image_size)
        for k, f in enumerate((fa, fb)):
            if np.isfinite(f):
                samples[k].append(f)

    best: tuple[float, float] | None = None
    for vals in samples:
        # Every jittered draw must give an answer, or the constraint is only
        # sometimes defined and cannot be trusted.
        if len(vals) < FOCAL_JITTER_SAMPLES:
            continue
        arr = np.array(vals)
        med = float(np.median(arr))
        if med <= 0:
            continue
        spread = float(np.percentile(arr, 84) - np.percentile(arr, 16)) / 2.0
        rel = spread / med
        if rel <= FOCAL_STABILITY_TOL and (best is None or rel < best[1]):
            best = (med, rel)
    return best


def calibrate_from_homography(
    H: np.ndarray,
    image_size: tuple[int, int],
    corners_goal=None,
    corners_image=None,
    assumed_focal_px: float | None = None,
    corner_jitter_px: float = FOCAL_JITTER_PX,
    known_focal_px: float | None = None,
    known_focal_spread: float = 0.0,
) -> CameraModel | None:
    """Solve for intrinsics and pose from a plane-to-image homography.

    ``H`` maps goal-plane (x, y) in inches to image pixels.  The principal
    point is assumed to be the image centre and pixels square, which is a good
    approximation for a phone camera and leaves focal length as the only
    unknown intrinsic.

    Which of the two orthonormality constraints is usable depends entirely on
    where the camera is, so rather than choose analytically the corners are
    jittered by ``corner_jitter_px`` -- the precision the outline is actually
    known to -- and the estimate that barely moves is kept.  Pass 0 to ask the
    noise-free question instead.  A view square-on to the net carries no
    reliable focal-length information at all.  Passing
    ``assumed_focal_px`` lets the solve continue on a typical phone lens; the
    pose is then only as good as that guess, and the model says so through
    ``focal_assumed``.  With no assumption to fall back on, this returns None,
    which is a statement that the geometry is not observable rather than a
    failure to try.  ``known_focal_px`` -- the lens as the phone recorded it,
    or as the player gave it -- skips the solve altogether.
    """
    H = np.asarray(H, dtype=np.float64)
    w, h = image_size
    cx, cy = w / 2.0, h / 2.0

    f = 0.0
    focal_assumed = True
    focal_spread = 0.0
    source = "assumed"
    if known_focal_px is not None and known_focal_px > 0:
        f, focal_spread, focal_assumed, source = float(known_focal_px), float(known_focal_spread), False, "known"
    elif corners_goal is not None and corners_image is not None:
        stable = _stable_focal(corners_goal, corners_image, image_size, corner_jitter_px)
        if stable is not None and 0.25 * w <= stable[0] <= 6.0 * w:
            f, focal_spread = stable
            focal_assumed = False
            source = "goal"

    if focal_assumed:
        if assumed_focal_px is None:
            return None
        f = float(assumed_focal_px)

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

    return CameraModel(
        K=K, R=R, t=t, focal_px=f, residual_px=residual,
        focal_assumed=focal_assumed, focal_spread=focal_spread, focal_source=source,
    )
