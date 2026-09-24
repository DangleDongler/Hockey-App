"""Shot speed.

A single camera cannot see depth, and a puck flying at a net is mostly moving
in depth.  That is the whole difficulty, and no amount of tracking precision
makes it go away.  So rather than pretend one number is authoritative, the
tracker runs up to three estimators, each with a different thing it assumes,
and reports which one it trusted and why.

``time_of_flight``
    Needs the distance the player is shooting from — one number, typed once.
    Given that, the puck's 3-D flight line is known end to end: the release
    point is (offset, stick height, distance) and the impact point comes from
    the goal-plane homography.  Each detection back-projects to a ray, and
    where that ray meets the flight line says how far along the flight the puck
    was.  Those fractions advance linearly with time, and the slope of that
    line is the flight time.  Nothing is extrapolated past what was seen.

``ballistic_3d``
    Needs nothing.  Fits a 3-D parabola whose projection matches the observed
    track.  Uniform motion along a ray bundle is famously ambiguous — a slow
    near puck and a fast far puck look identical — and it is gravity alone that
    breaks the tie.  Over a two-tenths-of-a-second flight that is a faint
    signal, so this one carries a real uncertainty and says so.

``goal_plane``
    Needs nothing, assumes little, and is the roughest: measure the puck's
    motion in the last frames before impact, where it is close enough to the
    goal plane that the plane's scale is nearly the puck's own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import CameraModel
from .config import GRAVITY_IN_S2, IN_PER_SEC_TO_MPH, Config
from .geometry import GoalPlane
from .tracking import Track


@dataclass
class SpeedEstimate:
    mph: float
    method: str
    confidence: float                      # 0..1
    uncertainty_mph: float | None = None
    notes: list[str] = field(default_factory=list)
    alternatives: dict[str, float] = field(default_factory=dict)
    # Estimators whose answer was physically impossible and so set aside.  Kept
    # for diagnosis; it says nothing about the speed that was reported.
    rejected: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mph": round(float(self.mph), 1),
            "method": self.method,
            "confidence": round(float(self.confidence), 3),
            "uncertainty_mph": None if self.uncertainty_mph is None else round(float(self.uncertainty_mph), 1),
            "notes": list(self.notes),
            "alternatives_mph": {k: round(float(v), 1) for k, v in self.alternatives.items()},
            "rejected_mph": {k: round(float(v), 1) for k, v in self.rejected.items()},
        }


def _rays(cam: CameraModel, pts_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera centre and unit ray directions in world coordinates."""
    uv = np.asarray(pts_image, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([uv, np.ones((len(uv), 1))])
    dirs_cam = (np.linalg.inv(cam.K) @ homog.T).T
    dirs = (cam.R.T @ dirs_cam.T).T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return cam.position, dirs


def _closest_param_on_segment(origin, dirs, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each ray, the parameter s where P(s) = a + s(b-a) comes closest."""
    ab = b - a
    denom_ab = float(ab @ ab)
    w0 = origin - a
    out = np.empty(len(dirs))
    for i, d in enumerate(dirs):
        # Closest approach between ray(origin, d) and line(a, ab).  Setting the
        # derivatives of |origin + lam*d - (a + s*ab)|^2 to zero gives
        #   [ d.d    -d.ab  ] [lam]   [ -d.w0  ]
        #   [ ab.d  -ab.ab  ] [ s ] = [ -ab.w0 ]
        A = np.array([[float(d @ d), -float(d @ ab)], [float(ab @ d), -denom_ab]])
        rhs = np.array([-float(d @ w0), -float(ab @ w0)])
        det = np.linalg.det(A)
        if abs(det) < 1e-12:
            out[i] = np.nan
            continue
        out[i] = np.linalg.solve(A, rhs)[1]
    return out


def goal_line_crossing(
    track: Track, plane: GoalPlane, cam: CameraModel, cfg: Config, first_guess: np.ndarray, iterations: int = 8
) -> tuple[np.ndarray, float] | None:
    """When the puck reached the goal line, and so where on the net it was.

    The goal-plane mapping is exact for a point on the goal line and wrong for
    one in front of it or behind it -- and at 30 fps the last sighting can be a
    yard short of the line, while a puck that goes in keeps sliding to the back
    of the net.  So work out *when* it crossed, from how fast it was closing
    on the line (the same fractions-along-the-flight as ``time_of_flight``),
    and map where the puck was in the picture at that moment.  Sightings
    already past the line are left out of the timing: in the mesh the puck
    slows down.
    """
    scfg = cfg.speed
    if scfg.shot_distance_ft is None or len(track) < 3:
        return None
    release = np.array([scfg.shooter_offset_ft * 12.0, scfg.release_height_in, scfg.shot_distance_ft * 12.0])
    origin, dirs = _rays(cam, track.points)
    frames = track.frames.astype(float)
    xy = np.asarray(first_guess, dtype=float)
    f_cross = float(frames[-1])
    for _ in range(iterations):
        s = _closest_param_on_segment(origin, dirs, release, np.array([xy[0], xy[1], 0.0]))
        use = np.isfinite(s) & (s <= 1.02)
        if use.sum() < 3:
            return None
        slope, intercept = np.polyfit(frames[use], s[use], 1)
        if slope <= 1e-9:
            return None
        f_cross = float(np.clip((1.0 - intercept) / slope, frames[0], frames[-1] + 1.5))
        if f_cross <= frames[-1]:
            img = np.array([np.interp(f_cross, frames, track.points[:, 0]),
                            np.interp(f_cross, frames, track.points[:, 1])])
        else:
            img = track.position_at(f_cross)
        new = plane.to_goal(img.reshape(1, 2))[0]
        if not np.all(np.isfinite(new)):
            return None
        done = np.hypot(*(new - xy)) < 0.05
        xy = new
        if done:
            break
    return xy, f_cross


def estimate_time_of_flight(
    track: Track,
    cam: CameraModel,
    impact_goal: np.ndarray,
    fps: float,
    cfg: Config,
) -> SpeedEstimate | None:
    """Speed from a known shooting distance, without extrapolating the track."""
    scfg = cfg.speed
    if scfg.shot_distance_ft is None:
        return None

    dist_in = scfg.shot_distance_ft * 12.0
    release = np.array([scfg.shooter_offset_ft * 12.0, scfg.release_height_in, dist_in])
    impact = np.array([float(impact_goal[0]), float(impact_goal[1]), 0.0])
    flight_in = float(np.linalg.norm(impact - release))
    if flight_in < 12.0:
        return None

    origin, dirs = _rays(cam, track.points)
    s = _closest_param_on_segment(origin, dirs, release, impact)
    frames = track.frames
    ok = np.isfinite(s)
    if ok.sum() < 3:
        return None
    s, frames = s[ok], frames[ok]

    # s should advance linearly with time; its slope is 1 / flight_time.
    slope, intercept = np.polyfit(frames, s, 1)
    if slope <= 1e-9:
        return None
    pred = slope * frames + intercept
    ss_res = float(((s - pred) ** 2).sum())
    ss_tot = float(((s - s.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    speed_in_s = flight_in * slope * fps
    mph = speed_in_s * IN_PER_SEC_TO_MPH

    # Slope uncertainty from the residual scatter, propagated straight through.
    n = len(frames)
    unc = None
    if n > 2:
        var_f = float(((frames - frames.mean()) ** 2).sum())
        if var_f > 1e-9:
            sigma = np.sqrt(ss_res / (n - 2))
            unc = float(flight_in * (sigma / np.sqrt(var_f)) * fps * IN_PER_SEC_TO_MPH)

    # Detection scatter is rarely the limit; not knowing the shooting distance
    # is.  Speed is directly proportional to it, so fold that in.
    rel_dist = scfg.distance_uncertainty_ft / max(scfg.shot_distance_ft, 1e-6)
    rel_pose = scfg.assumed_lens_uncertainty_frac if cam.focal_assumed else scfg.pose_uncertainty_frac
    sys_unc = abs(mph) * float(np.hypot(rel_dist, rel_pose))
    unc = float(np.hypot(unc or 0.0, sys_unc))

    notes = []
    if r2 < 0.97:
        notes.append(
            f"the puck's progress along the assumed flight line was not quite linear (R2={r2:.3f}); "
            "the shooting distance may be off"
        )
    conf = float(np.clip(0.45 + 0.55 * max(r2, 0.0), 0.0, 0.97))
    if cam.focal_assumed:
        conf *= 0.8
        notes.append("the lens was assumed rather than measured from the goal; see the error bar")
    return SpeedEstimate(mph=mph, method="time_of_flight", confidence=conf, uncertainty_mph=unc, notes=notes)


def estimate_ballistic_3d(
    track: Track, cam: CameraModel, fps: float, cfg: Config, seed_distance_in: float = 240.0
) -> SpeedEstimate | None:
    """Fit a 3-D parabola to the observed track; gravity sets the scale."""
    try:
        from scipy.optimize import least_squares
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        return None

    frames = track.frames
    obs = track.points
    n = len(frames)
    if n < 5:
        return None
    t = (frames - frames[0]) / fps

    # Seed: a straight shot from roughly the usual distance to where it ended.
    origin, dirs = _rays(cam, obs[[0, -1]])
    # Put the last detection on the goal plane and the first at the seed depth.
    def ray_to_z(d, z):
        if abs(d[2]) < 1e-9:
            return origin + d * seed_distance_in
        return origin + d * ((z - origin[2]) / d[2])

    p_start = ray_to_z(dirs[0], seed_distance_in)
    p_end = ray_to_z(dirs[1], 0.0)
    v_seed = (p_end - p_start) / max(t[-1], 1e-3)
    x0 = np.concatenate([p_start, v_seed])

    def residuals(params):
        p0, v0 = params[:3], params[3:]
        pos = p0[None, :] + v0[None, :] * t[:, None]
        pos[:, 1] -= 0.5 * GRAVITY_IN_S2 * t ** 2
        return (cam.project(pos) - obs).ravel()

    try:
        sol = least_squares(residuals, x0, method="lm", max_nfev=4000)
    except Exception:
        return None
    if not sol.success and sol.status <= 0:
        return None

    v0 = sol.x[3:]
    speed_in_s = float(np.linalg.norm(v0))
    mph = speed_in_s * IN_PER_SEC_TO_MPH
    if not np.isfinite(mph) or mph <= 0:
        return None

    # Uncertainty from the fit's covariance, projected onto the speed direction.
    unc = None
    conf = 0.25
    try:
        J = sol.jac
        dof = max(len(sol.fun) - len(sol.x), 1)
        s2 = float((sol.fun ** 2).sum()) / dof
        cov = np.linalg.pinv(J.T @ J) * s2
        grad = np.concatenate([np.zeros(3), v0 / max(speed_in_s, 1e-9)])
        var = float(grad @ cov @ grad)
        if var >= 0:
            unc = float(np.sqrt(var) * IN_PER_SEC_TO_MPH)
    except Exception:
        pass
    if unc is not None:
        # The fit only knows how well the curve matches the dots -- and at
        # 240 fps there are so many dots that this alone says +-1%.  Every ray
        # it fits comes from the camera model, so the camera's own error comes
        # with it: an assumed lens moves this estimate as much as it moves the
        # time-of-flight one.
        rel_cam = cfg.speed.assumed_lens_uncertainty_frac if cam.focal_assumed else cfg.speed.pose_uncertainty_frac
        unc = float(np.hypot(unc, rel_cam * mph))
        rel = unc / max(mph, 1e-6)
        conf = float(np.clip(0.9 * np.exp(-6.0 * rel), 0.02, 0.85))

    notes = []
    if unc is not None and unc > 0.15 * mph:
        notes.append(
            "single-camera depth is weakly constrained over a flight this short; "
            "supply the shooting distance for a much tighter number"
        )
    return SpeedEstimate(mph=mph, method="ballistic_3d", confidence=conf, uncertainty_mph=unc, notes=notes)


def estimate_goal_plane(track: Track, plane: GoalPlane, fps: float, cfg: Config) -> SpeedEstimate | None:
    """Rough speed from image motion in the final frames before impact."""
    window = max(2, cfg.speed.plane_window)
    pts = track.points[-window:]
    frames = track.frames[-window:]
    if len(pts) < 2:
        return None

    goal_pts = plane.to_goal(pts)
    steps = np.linalg.norm(np.diff(goal_pts, axis=0), axis=1)
    dts = np.diff(frames) / fps
    good = dts > 1e-9
    if not good.any():
        return None
    speeds = steps[good] / dts[good]
    speed_in_s = float(np.median(speeds))
    mph = speed_in_s * IN_PER_SEC_TO_MPH

    # This method only works when the camera can actually see the puck move.
    obliquity = plane.obliquity
    conf = float(np.clip(0.15 + 1.9 * obliquity, 0.05, 0.55))
    notes = [
        "measured from pixel motion at the goal plane; treat as a rough figure "
        "unless the camera is well off to the side"
    ]
    if obliquity < 0.06:
        notes.append("the camera is nearly square to the net, so this estimate is close to meaningless")
    return SpeedEstimate(mph=mph, method="goal_plane", confidence=conf, notes=notes)


def estimate_speed(
    track: Track,
    plane: GoalPlane,
    cam: CameraModel | None,
    impact_goal: np.ndarray,
    fps: float,
    cfg: Config,
) -> SpeedEstimate | None:
    """Run every estimator that applies and return the most trustworthy."""
    results: list[SpeedEstimate] = []
    if cam is not None:
        tof = estimate_time_of_flight(track, cam, impact_goal, fps, cfg)
        if tof:
            results.append(tof)
        seed = (cfg.speed.shot_distance_ft or 20.0) * 12.0
        bal = estimate_ballistic_3d(track, cam, fps, cfg, seed_distance_in=seed)
        if bal:
            results.append(bal)
    gp = estimate_goal_plane(track, plane, fps, cfg)
    if gp:
        results.append(gp)
    if not results:
        return None

    # A degenerate fit can produce a number like 87,000 mph.  Such a value is
    # not a speed with a large error bar, it is a failed solve, and it must not
    # be preferred over a workable estimate just because it scored well.
    scfg_band = cfg.speed
    absurd = [
        r for r in results
        if not (0.2 * scfg_band.plausible_min_mph <= r.mph <= 5.0 * scfg_band.plausible_max_mph)
    ]
    usable = [r for r in results if r not in absurd]
    if usable:
        results = usable
    else:
        absurd = []

    wanted = cfg.speed.method
    chosen = None
    if wanted != "auto":
        chosen = next((r for r in results if r.method == wanted), None)
        if chosen is None:
            chosen = max(results, key=lambda r: r.confidence)
            chosen.notes.append(f"requested method {wanted!r} was not available here; used {chosen.method!r}")
    else:
        chosen = max(results, key=lambda r: r.confidence)

    chosen.alternatives = {r.method: r.mph for r in results if r is not chosen}
    chosen.rejected = {r.method: r.mph for r in absurd}

    scfg = cfg.speed
    if not (scfg.plausible_min_mph <= chosen.mph <= scfg.plausible_max_mph):
        chosen.notes.append(
            f"{chosen.mph:.0f} mph is outside the {scfg.plausible_min_mph:.0f}-"
            f"{scfg.plausible_max_mph:.0f} mph band; check that the clip's frame rate "
            "is right (slow-motion footage often mislabels it)"
        )
        chosen.confidence *= 0.5
    return chosen
