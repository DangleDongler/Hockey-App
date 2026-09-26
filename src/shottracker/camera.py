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
# The pose is fitted to the goal's corners only when they fit a goal of the
# size given, seen through this lens: the corners' RMS miss under this
# fraction of the goal's width in the picture, and the camera no lower than
# CAMERA_MIN_HEIGHT_IN.  Synthetic outlines found to a pixel or two fit to
# 0.3-0.5%; the real backyard clips, with the net entered as 72 x 48, missed by
# 1.1-1.4% -- and fitting the pose to those put the camera nine feet below
# the ground, reading speeds 20-65% off.
PLANAR_FIT_TOL = 0.0075
CAMERA_MIN_HEIGHT_IN = -12.0
# ...and only where a pixel's error in the crossbar's height (or the posts'
# feet) moves the fitted camera less than this.  That is the outline's
# commonest error on real clips -- the bar or the netting just under it,
# grass over the feet -- and the homography's pose, scaled by the goal's
# width, hardly feels it (1-2 in), where a fit from far and low swings by a
# foot or more: with it, the real clips' 60 fps copies spread 7.1% on speed
# at the median instead of 5.4%, and 43% at worst instead of 25%.
PLANAR_MAX_HEIGHT_SENSITIVITY_IN = 6.0


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
    # True when the pose was fitted to the goal's corners and passed every
    # check that it is well settled by them (see _planar_pose).
    pose_from_corners: bool = False

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


def planar_fit(corners_goal, corners_image, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """The pose that best fits the goal's corners through lens ``K``, and its RMS miss in px.

    Of OpenCV's two mirror-image answers (IPPE), the one that puts the camera
    on the shooter's side of the goal, looking at it, is kept.
    """
    cg = np.asarray(corners_goal, dtype=np.float64).reshape(-1, 2)
    ci = np.asarray(corners_image, dtype=np.float64).reshape(-1, 1, 2)
    if len(cg) < 4 or len(cg) != len(ci):
        return None
    obj = np.hstack([cg, np.zeros((len(cg), 1))]).reshape(-1, 1, 3)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, ci, K, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    errs = np.ravel(errs) if errs is not None else np.full(n, np.nan)
    for rvec, tvec, err in zip(rvecs[:n], tvecs[:n], errs[:n]):
        R, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec, dtype=np.float64).ravel()
        if (-R.T @ t)[2] > 0 and (R @ np.array([0.0, 24.0, 0.0]) + t)[2] > 0:
            return R, t, float(err)
    return None


def _planar_pose(corners_goal, corners_image, K: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Pose from the goal's corners and a known lens, fitted to the corners themselves.

    Reading the pose straight out of the homography scales it by one column
    and squares up the rotation afterwards, which spreads any error in the
    corners into the camera's position: on synthetic views with the lens
    known, a pixel of error in the outline put the camera 21-35 in from where
    it was, and every speed with it.  Fitting the pose to the corners directly
    lands 6-11 in away from the same corners.

    But only if the corners fit a goal of the size given at all (see
    PLANAR_FIT_TOL): when they do not, the best fit explains the mismatch by
    tilting the view, and the camera ends up feet underground.  And only if
    the view is not so far and low that an error in the crossbar's height
    swings the fit (see PLANAR_MAX_HEIGHT_SENSITIVITY_IN).  Otherwise the
    homography's answer, which takes its scale from the goal's width, is the
    better one, and None says to keep it.
    """
    fit = planar_fit(corners_goal, corners_image, K)
    if fit is None:
        return None
    R, t, rms = fit
    ci = np.asarray(corners_image, dtype=np.float64).reshape(-1, 2)
    width_px = 0.5 * (np.linalg.norm(ci[1] - ci[0]) + np.linalg.norm(ci[2] - ci[3]))
    if not np.isfinite(rms) or rms > PLANAR_FIT_TOL * width_px:
        return None
    if (-R.T @ t)[1] < CAMERA_MIN_HEIGHT_IN:
        return None
    lowered = ci.copy()
    lowered[:2, 1] += 1.0                          # the top corners, a pixel down
    moved = planar_fit(corners_goal, lowered, K)
    if moved is None or np.linalg.norm((-moved[0].T @ moved[1]) - (-R.T @ t)) > PLANAR_MAX_HEIGHT_SENSITIVITY_IN:
        return None
    return R, t


def hidden_feet(corners_image, K: np.ndarray, goal, max_hidden_in: float = 15.0,
                min_hidden_in: float = 1.5) -> tuple[np.ndarray, float] | None:
    """Where the posts' feet are when grass hides them, or None if nothing is hidden.

    From a phone on the ground, the lawn between it and the net hides the
    bottom of the posts, and the outline stops where the red stops.  On the
    real backyard clips that was 6-8 in of a 48 in NHL net: the outline fitted
    a goal 86% as tall to a pixel, and put the camera nine feet underground as
    72 x 48.  With the lens known, the outline's proportions are measured, so
    the missing strip can be too: fit the goal with its bottom ``hidden``
    inches cut off, and take the cut that fits best.  (A strip missing from
    the top would fit just as well -- the shape cannot tell -- but the top is
    the crossbar, which nothing in a backyard hides.)  Returns the outline
    with the feet put back, and how many inches were hidden.
    """
    ci = np.asarray(corners_image, dtype=np.float64).reshape(-1, 2)
    width_px = 0.5 * (np.linalg.norm(ci[1] - ci[0]) + np.linalg.norm(ci[2] - ci[3]))
    if width_px <= 0:
        return None
    hw, top = goal.outer_width_in / 2.0, goal.outer_height_in

    def fit(hidden: float):
        rect = np.array([[-hw, top], [hw, top], [hw, hidden], [-hw, hidden]])
        f = planar_fit(rect, ci, K)
        if f is None or (-f[0].T @ f[1])[1] < CAMERA_MIN_HEIGHT_IN:
            return None
        return f[2] / width_px, f[0], f[1]

    as_seen = fit(0.0)
    if as_seen is not None and as_seen[0] <= PLANAR_FIT_TOL:
        return None
    fits = [(f[0], h, f) for h in np.arange(0.0, max_hidden_in + 1e-9, 0.25) if (f := fit(float(h))) is not None]
    if not fits:
        return None
    best, hidden, (_, R, t) = min(fits, key=lambda x: x[0])
    if hidden < min_hidden_in or best > 0.5 * PLANAR_FIT_TOL:
        return None
    feet = np.array([[hw, 0.0, 0.0], [-hw, 0.0, 0.0]])
    cam = (R @ feet.T).T + t
    if np.any(cam[:, 2] <= 0):
        return None
    img = (K @ (cam / cam[:, 2:3]).T).T[:, :2]
    return np.array([ci[0], ci[1], img[0], img[1]]), float(hidden)


# A phone "on the ground" -- propped up in portrait, its lens a few inches
# above the grass or gravel -- is held at this height (see ground_pose).  On
# the real clips anything from -6 to +12 in moved the speeds by 1-2%.
ON_GROUND_HEIGHT_IN = 4.0
# The held fit is kept only while the outline fits a goal of the size given,
# seen from that height, to this fraction of the goal's width in the picture.
# The real slow-motion clips it was made for fitted to 1.8%.
GROUND_FIT_TOL = 0.03


def _look_at(centre: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-to-camera rotation for a camera at ``centre`` looking at ``target``, held level."""
    z = target - centre
    z = z / np.linalg.norm(z)
    x = np.cross(np.array([0.0, -1.0, 0.0]), z)
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def ground_pose(corners_image, K: np.ndarray, goal, height_in: float,
                max_hidden_in: float = 15.0) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """The pose of a camera at a known height, and how much of the posts the ground hides.

    From a phone on the ground, a goal 25 ft away is a fifth of the frame
    wide, and its outline cannot say how high the phone is: on two real
    slow-motion clips an outline 2-4 px out put it 14 and 48 in up instead
    of on the gravel, the puck 10-32 in in the air as it left the stick, and
    read the shots 6% and 20% fast.  Holding the height where the player
    says it was leaves the outline to settle the rest -- which way the phone
    faced, how far out and to the side it was, how much grass hides the
    feet -- and on those clips landed within 2.5% of the speeds the
    background itself gave (the camera placed by matching the house and
    fence to a 4K clip from the same spot).

    The fit starts from a fan of placements around the goal, so it does not
    settle on the first answer near a poor guess.  Returns (R, t, hidden
    inches, RMS miss in px) or None.
    """
    from scipy.optimize import least_squares

    ci = np.asarray(corners_image, dtype=np.float64).reshape(-1, 2)
    if ci.shape != (4, 2):
        return None
    width_px = 0.5 * (np.linalg.norm(ci[1] - ci[0]) + np.linalg.norm(ci[2] - ci[3]))
    if width_px <= 0:
        return None
    hw, top = goal.outer_width_in / 2.0, goal.outer_height_in

    def rect(hidden: float) -> np.ndarray:
        return np.array([[-hw, top, 0.0], [hw, top, 0.0], [hw, hidden, 0.0], [-hw, hidden, 0.0]])

    def pose(p) -> tuple[np.ndarray, np.ndarray]:
        R = cv2.Rodrigues(np.asarray(p[:3], dtype=np.float64))[0]
        return R, -R @ np.array([p[3], height_in, p[4]])

    def miss(p) -> np.ndarray:
        R, t = pose(p)
        cam = (R @ rect(p[5]).T).T + t
        if np.any(cam[:, 2] <= 1.0):
            return np.full(8, 1e3)
        return ((K @ (cam / cam[:, 2:3]).T).T[:, :2] - ci).ravel()

    reach = float(K[0, 0]) * goal.outer_width_in / width_px   # how far a goal this wide is, head on
    aim = np.array([0.0, top / 2.0, 0.0])
    lower = [-np.inf] * 3 + [-10.0 * reach, 24.0, 0.0]
    upper = [np.inf] * 3 + [10.0 * reach, 10.0 * reach, max_hidden_in]
    best = None
    for az in np.radians(np.arange(-70.0, 71.0, 10.0)):
        for d in (0.8 * reach, reach, 1.25 * reach):
            c = np.array([d * np.sin(az), height_in, d * np.cos(az)])
            x0 = np.r_[cv2.Rodrigues(_look_at(c, aim))[0].ravel(), c[0], c[2], 5.0]
            try:
                sol = least_squares(miss, x0, bounds=(lower, upper))
            except ValueError:
                continue
            if best is None or sol.cost < best.cost:
                best = sol
    if best is None or best.cost >= 0.5 * 8 * 1e6:
        return None
    R, t = pose(best.x)
    rms = float(np.sqrt(np.mean(np.sum(best.fun.reshape(-1, 2) ** 2, axis=1))))
    return R, t, float(best.x[5]), rms


def outline_height_fit(corners_image, K: np.ndarray, goal) -> tuple[float, float, float] | None:
    """How well the outline fits the goal as entered, and which mouth height fits it best.

    A rectangle's proportions can be read from its outline once the lens is
    known (its size cannot: a bigger goal further away looks the same).  Only
    the height is searched -- the width is what the player is surest of.
    Returns (RMS miss as entered, best mouth height, RMS miss at that height),
    misses as fractions of the goal's width in the picture, or None.
    """
    from dataclasses import replace

    from .geometry import outer_rect

    ci = np.asarray(corners_image, dtype=np.float64).reshape(-1, 2)
    width_px = 0.5 * (np.linalg.norm(ci[1] - ci[0]) + np.linalg.norm(ci[2] - ci[3]))
    if width_px <= 0:
        return None

    def miss(height: float) -> float | None:
        fit = planar_fit(outer_rect(replace(goal, mouth_height_in=height)), ci, K)
        if fit is None or (-fit[0].T @ fit[1])[1] < CAMERA_MIN_HEIGHT_IN:
            return None
        return fit[2] / width_px

    entered = miss(goal.mouth_height_in)
    heights = np.arange(0.6 * goal.mouth_height_in, 1.3 * goal.mouth_height_in, 0.5)
    fits = [(m, hgt) for hgt in heights if (m := miss(float(hgt))) is not None]
    if not fits:
        return None
    best, height = min(fits)
    return (entered if entered is not None else float("inf")), float(height), float(best)


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

    # With the lens known, fit the pose to the corners.  With the lens solved
    # from this same outline, the homography's pose is the one consistent with
    # that solve: fitting afresh on top of it made the benchmark's speeds
    # worse (1.1-1.9% to 2.0-3.4%).
    settled = False
    if source == "known" and corners_goal is not None and corners_image is not None:
        better = _planar_pose(corners_goal, corners_image, K)
        if better is not None:
            (R, t), settled = better, True

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
        pose_from_corners=settled,
    )
