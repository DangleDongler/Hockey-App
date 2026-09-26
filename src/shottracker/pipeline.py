"""Video in, session out.

Two passes over the clip.  The first samples frames to find the goal and build
the background plate; the second walks every frame looking for the puck.  The
goal is found before the puck on purpose — its size in pixels is what tells the
puck detector how big a puck should look, and its plane is what turns a pixel
into a spot on the net.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import (
    GROUND_FIT_TOL,
    PLANAR_FIT_TOL,
    CameraModel,
    calibrate_from_homography,
    ground_pose,
    hidden_feet,
    outline_height_fit,
)
from .config import PUCK_DIAMETER_IN, CameraConfig, Config
from .container import Lens, detect_slow_motion, read_lens, read_timing
from .geometry import GoalPlane, build_zones, mouth_outline, outer_outline, outer_rect
from .net_detect import NetDetection, detect_net
from .puck_detect import Candidate, PuckDetector, build_background_and_noise, demote_recurring
from .shots import Shot, approaches_goal, dedupe_shots, shot_from_track
from .stabilize import CameraMotion, estimate_motion, interpolate, measure
from .tracking import Track, build_tracks, filter_by_clutter, filter_by_speed, revisits, until_impact
from .video import finish, iter_frames, iter_raw_frames, keyframes


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float                  # frames per second of real time: what timing uses
    frame_count: int
    fps_source: str = "container"
    # The rate the file plays at, when that differs from the rate it was filmed
    # at (slow motion with the slowdown baked in).  Anything that seeks in or
    # re-encodes the file itself goes by this, not by ``fps``.
    playback_fps: float | None = None
    # The lens, as the phone recorded it in the file (ordinary video only).
    lens: str | None = None
    focal_35mm: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "playback_fps": round(self.playback_fps, 3) if self.playback_fps else None,
            "frame_count": self.frame_count,
            "duration_s": round(self.frame_count / self.fps, 2) if self.fps else None,
            "fps_source": self.fps_source,
            "lens": self.lens,
            "focal_35mm": self.focal_35mm,
        }


@dataclass
class SessionResult:
    video: VideoInfo
    net: NetDetection | None
    shots: list[Shot] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    camera: CameraModel | None = None
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    # The settings this run used, so reporting describes the goal that was
    # actually measured rather than assuming a regulation one.
    config: Config = field(default_factory=Config)

    @property
    def plane(self) -> GoalPlane | None:
        return None if self.net is None else GoalPlane(self.net.quad, self.config.goal)

    def targeting(self) -> dict | None:
        from .targets import targeting_block

        return targeting_block(
            [(s.impact_goal_in[0], s.impact_goal_in[1], s.outcome) for s in self.shots],
            self.config.goal,
            self.config.target,
            self.plane,
        )

    def to_dict(self) -> dict:
        from .report import summarize

        targeting = self.targeting()
        shots = [s.to_dict() for s in self.shots]
        if targeting:
            for sd, score in zip(shots, targeting["per_shot"]):
                sd["vs_target"] = score

        return {
            "video": self.video.to_dict(),
            "net": self.net.to_dict() if self.net else None,
            "camera": (
                {
                    "focal_px": round(float(self.camera.focal_px), 1),
                    "position_in": [round(float(v), 1) for v in self.camera.position],
                    "reprojection_error_px": round(float(self.camera.residual_px), 2),
                }
                if self.camera
                else None
            ),
            "shots": shots,
            "targeting": targeting,
            "tracks": [
                {
                    "shot_index": i,
                    "frames": [int(c.frame) for c in t.candidates],
                    "points": [[round(float(c.x), 1), round(float(c.y), 1)] for c in t.candidates],
                }
                for i, t in enumerate(self.tracks)
            ],
            "goal": {
                "mouth_width_in": self.config.goal.mouth_width_in,
                "mouth_height_in": self.config.goal.mouth_height_in,
                "post_diameter_in": self.config.goal.post_diameter_in,
                "corner_radius_in": self.config.goal.corner_radius_in,
                "mouth_quad": (
                    self.plane.mouth_image_quad.round(1).tolist() if self.net else None
                ),
                # The opening's real shape in image pixels, bends included, so
                # the browser draws the goal rather than a box.
                "mouth_outline": (
                    self.plane.to_image(mouth_outline(self.config.goal)).round(1).tolist()
                    if self.net
                    else None
                ),
                "outer_outline": (
                    self.plane.to_image(outer_outline(self.config.goal)).round(1).tolist()
                    if self.net
                    else None
                ),
            },
            "summary": summarize(self),
            "warnings": list(self.warnings),
            "elapsed_s": round(self.elapsed_s, 2),
        }


def _capture_warnings(info: VideoInfo, cfg: Config) -> list[str]:
    """Check, before doing any work, whether the clip *can* resolve a shot.

    A puck only exists in the footage for distance / speed seconds.  At 30 fps
    and a short backyard distance that is three or four frames, which is below
    what any tracker can work with -- and no amount of processing recovers a
    puck that was never captured.
    """
    out: list[str] = []
    speed_in_s = cfg.speed.nominal_speed_mph * 17.6
    dist_in = (cfg.speed.shot_distance_ft or 20.0) * 12.0
    frames_of_flight = dist_in / speed_in_s * info.fps
    needed = cfg.track.min_track_length

    if frames_of_flight < needed + 2:
        out.append(
            f"at {info.fps:.0f} fps, a {cfg.speed.nominal_speed_mph:.0f} mph shot from "
            f"{dist_in/12:.0f} ft is only in the air for about {frames_of_flight:.0f} frames, and the "
            f"tracker needs at least {needed} sightings to call something a shot. Record in "
            "slow-motion -- 120 or 240 fps, which any recent phone offers -- or move further back. "
            "This is the single thing that most decides whether a clip can be read at all."
        )
    elif frames_of_flight < 2 * needed:
        out.append(
            f"at {info.fps:.0f} fps there are only about {frames_of_flight:.0f} frames of puck flight "
            "to work with; speeds will be rough. Slow-motion capture would tighten them considerably."
        )
    return out


def _busy_scene_warning(
    saturated: float, samples: list[np.ndarray], scale: float, already_aligned: bool, cfg: Config
) -> str:
    """Say why most frames were full of moving things, having checked.

    Two different problems look the same to the puck detector: a camera that
    moved, so the whole scene slid against the background plate, and a steady
    camera watching a scene that will not keep still -- wind in leaves and
    netting, flickering sunlight, people.  The fixes differ, so measure which
    it was on the frames already sampled for the plate instead of guessing.
    """
    head = f"{saturated:.0%} of frames were full of moving objects, so the puck is hard to pick out. "
    drift = None
    if not already_aligned and len(samples) >= 2:
        s2 = min(1.0, 320.0 / float(samples[0].shape[1]))
        grays = {}
        for i, f in enumerate(samples):
            small = cv2.resize(f, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA)
            grays[i] = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        ref = grays[len(grays) // 2]
        est = estimate_motion(ref, grays, scale=scale * s2)
        weak = sum(1 for _, resp in est.values() if resp < 0.12)
        shifts = sorted((abs(dx) + abs(dy) for (dx, dy), resp in est.values() if resp >= 0.12), reverse=True)
        # A camera that moved shows it in many of the samples.  One sample
        # thrown off by something big crossing the view -- the shooter, a
        # stick -- is not the camera: on a synthetic clip from a camera that
        # never moved, that alone read as 12 px of drift.
        moved = shifts[len(shifts) // 4] if shifts else 0.0
        drift = (moved, weak / max(len(est), 1))

    if drift is not None and drift[0] <= cfg.puck.stabilize_drift_threshold_px and drift[1] < 0.25:
        return head + (
            "The camera held still, so it was the scene itself moving: wind in trees or netting, "
            "sunlight flickering, people walking through. Filming with the sun behind the phone and "
            "less moving background in frame helps most."
        )
    if drift is None:
        how = "The camera moved during the clip"
    elif drift[1] >= 0.25:
        how = "The view kept changing, as when a phone is carried or panned"
    else:
        how = f"The camera moved during the clip, by {drift[0]:.0f} px or more"
    return head + how + "; prop the phone against something steady rather than holding it."


def _held_camera(quad: np.ndarray, K: np.ndarray, cfg: Config,
                 warnings: list[str]) -> tuple[CameraModel, np.ndarray, str | None] | None:
    """The camera fitted at the height the player gave, when the outline alone cannot place it.

    Returns the camera, the outline with the posts' hidden feet put back, and
    a note on them -- or None, saying why, when the outline does not fit a
    goal of the size given seen from that height.
    """
    g, height = cfg.goal, float(cfg.camera.height_in)
    fit = ground_pose(quad, K, g, height)
    q = np.asarray(quad, dtype=np.float64)
    width_px = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    if fit is None or fit[3] > GROUND_FIT_TOL * width_px:
        warnings.append(
            f"the net's outline does not fit a {g.mouth_width_in:.0f} x {g.mouth_height_in:.0f} in goal seen from "
            f"{height:.0f} in off the ground, so where the phone was is worked out from the outline alone. If the "
            "phone was raised, say so in the form; if not, check the net's size"
        )
        return None
    R, t, hidden, rms = fit
    hw, top = g.outer_width_in / 2.0, g.outer_height_in
    corners = np.array([[-hw, top, 0.0], [hw, top, 0.0], [hw, 0.0, 0.0], [-hw, 0.0, 0.0]])
    cam_pts = (R @ corners.T).T + t
    full = (K @ (cam_pts / cam_pts[:, 2:3]).T).T[:, :2]
    note = None
    if hidden >= 1.5:
        note = (f"the bottom {hidden:.0f} in of the posts look hidden -- grass, seen from low down -- so the goal's "
                f"feet were put where a {g.mouth_width_in:.0f} x {g.mouth_height_in:.0f} in goal's must be")
    warnings.append(
        f"the net is too small in the picture to show how high the phone was, so it was taken as {height:.0f} in "
        "off the ground, as entered"
    )
    cam = CameraModel(K=K, R=R, t=t, focal_px=float(K[0, 0]), residual_px=rms, focal_source="known")
    return cam, full, note


def _goal_size_note(quad: np.ndarray, K: np.ndarray, cfg: Config) -> str | None:
    """Say so when the net's outline does not fit a goal of the size entered.

    With the lens known, the outline's proportions are measured, not assumed.
    An outline that is short is usually posts hidden in grass, and is put
    right before this (see camera.hidden_feet); one that is taller than the
    size entered, or short by more than grass could hide, is a net of another
    size.
    """
    fit = outline_height_fit(quad, K, cfg.goal)
    if fit is None:
        return None
    entered, height, best = fit
    if entered <= PLANAR_FIT_TOL or best > 0.5 * PLANAR_FIT_TOL or abs(height - cfg.goal.mouth_height_in) < 2.0:
        return None
    g = cfg.goal
    return (
        f"the net's outline fits a goal about {g.mouth_width_in:.0f} x {height:.0f} in far better than the "
        f"{g.mouth_width_in:.0f} x {g.mouth_height_in:.0f} in entered. Measure the opening, and if it is "
        f"{g.mouth_width_in:.0f} x {height:.0f}, enter that: every mark's height and the speed depend on it"
    )


def _too_slow_for_a_shot(shot: Shot, cfg: Config) -> bool:
    """Whether a timed flight was too slow to have been a shot.

    Only the timed flight (a shooting distance was given) is trusted for
    this: the fallback estimators read real shots at a third of their speed
    on the real clips, and must not throw them away.
    """
    sp = shot.speed
    return bool(sp is not None and sp.method == "time_of_flight" and sp.mph < cfg.shot.min_shot_mph)


def _near_goal(plane: GoalPlane, goal_width_px: float, cfg: Config):
    """A test for whether an image point is at the net, give or take."""
    outline = plane.to_image(outer_outline(plane.goal)).astype(np.float32).reshape(-1, 1, 2)
    reach = cfg.track.impact_near_goal_frac * goal_width_px

    def near(p) -> bool:
        return cv2.pointPolygonTest(outline, (float(p[0]), float(p[1])), True) >= -reach

    return near


def _distance_check(shots: list[Shot], cfg: Config) -> str | None:
    """Say so when the flights start further out than the distance given.

    Only a flight seen from the stick onward can say where it started; one
    first seen mid-flight says nothing, so this only ever reports a distance
    that is *longer* than the one given, and only when most such shots agree.
    """
    given = cfg.speed.shot_distance_ft
    if not given or not shots:
        return None
    implied = [s.speed.implied_distance_ft for s in shots if s.speed and s.speed.implied_distance_ft]
    if not implied or len(implied) * 2 < len(shots):
        return None
    median = float(np.median(implied))
    if median < given * (1.0 + cfg.speed.distance_warn_frac):
        return None
    return (
        f"the puck seems to leave the stick about {median:.0f} ft from the goal line "
        f"({len(implied)} of {len(shots)} shot(s)), not the {given:g} ft given. If the shooting spot was "
        f"further back, run it again with that distance: every speed scales with it, and this one would "
        f"read about {100 * (median / given - 1):.0f}% faster. From a camera far off or low, the estimate is rough."
    )


def decoded_frame_rate(path: str, frames: int = 40) -> float | None:
    """The rate the frames that actually play are spaced at, from their timestamps.

    A phone's header can disagree with its own frames.  iPhone 60 fps video
    starts with a few frames at 30 fps that the file's edit list then trims,
    and the header's average still counts them: one real clip says 52.1 fps
    while every frame that plays is 1/60 s apart -- a 15% error in every
    speed if the header is believed.
    """
    cap = cv2.VideoCapture(path)
    stamps: list[float] = []
    try:
        while len(stamps) < frames and cap.grab():
            stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        cap.release()
    gaps = np.diff(stamps)
    gaps = gaps[gaps > 0]
    if len(gaps) < 3:
        return None
    return 1000.0 / float(np.median(gaps))


def probe(path: str, cfg: Config) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if fps > 1.0 and not cfg.fps_override:
        decoded = decoded_frame_rate(path)
        if decoded and abs(decoded / fps - 1.0) > 0.02:
            note = (f"the file says {fps:.1f} fps, but the frames that play are 1/{decoded:.0f} s apart; "
                    f"timing uses {decoded:.0f} fps")
            fps = decoded
        else:
            note = None
    else:
        note = None
    info = VideoInfo(path=path, width=width, height=height, fps=fps, frame_count=max(count, 0))
    if note:
        info.notes.append(note)
        info.fps_source = "frame timestamps"
    if cfg.fps_override:
        info.fps, info.fps_source = float(cfg.fps_override), "override"
    elif fps <= 1.0:
        info.fps, info.fps_source = 30.0, "assumed"
    else:
        slow = detect_slow_motion(read_timing(path), fps)
        if slow.capture_fps:
            info.fps, info.fps_source = slow.capture_fps, "slow-motion"
        if slow.note:
            info.notes.append(slow.note)
    if fps > 1.0 and abs(info.fps - fps) > 0.01 * fps:
        # Timing now runs at a different rate from the file itself.
        info.playback_fps = fps
    lens = read_lens(path)
    if lens is not None:
        info.lens, info.focal_35mm = lens.model, lens.focal_35mm
    return info


def known_focal_px(info: VideoInfo, cfg: Config) -> tuple[float, float, str] | None:
    """The lens's focal length in pixels when it is known rather than solved for.

    Returns (focal, relative uncertainty, where it came from), or None.
    """
    if cfg.camera.hfov_deg:
        return CameraConfig.frac_from_hfov(cfg.camera.hfov_deg) * info.width, 0.03, "the field of view given"
    if cfg.camera.use_lens_metadata and info.focal_35mm:
        f = Lens(focal_35mm=info.focal_35mm).focal_px(info.width, info.height)
        if f:
            what = f"{info.lens or 'the phone lens'}, {info.focal_35mm:g} mm equivalent, as the phone recorded it"
            return f, cfg.camera.lens_metadata_spread, what
    if cfg.camera.focal_35mm:
        f = Lens(focal_35mm=cfg.camera.focal_35mm).focal_px(info.width, info.height)
        if f:
            return f, 0.06, f"the lens chosen ({cfg.camera.focal_35mm:g} mm equivalent)"
    return None


# Seeking in phone video decodes forward from the last keyframe -- a quarter
# of a second a seek for 1080p HEVC -- so when the wanted frames are this close
# together it is cheaper to decode straight through and keep the ones needed.
SEQUENTIAL_SAMPLE_STRIDE = 48


def sample_frames(path: str, n: int, total: int, exact: bool = True,
                  size: tuple[int, int] | None = None, gray: bool = False,
                  fast: bool = False) -> list[np.ndarray]:
    """Evenly spaced frames: read straight through a short clip, jump through a long one.

    ``exact`` asks for frames at exactly evenly spaced indices.  Without it, a
    long clip is sampled at its keyframes, which decode on their own and are
    about a second apart -- seconds instead of minutes on a long 4K clip.

    ``fast`` reads a short clip with the fast reader (video.iter_frames), so
    the frames are exactly what the per-frame pass sees; the background plate
    must be.  Otherwise a short clip is read with OpenCV itself: the frames
    the net is found on decide where the camera is worked out to be, and on
    a small far net even the fast reader's HDR colour mapping (a few grey
    levels off OpenCV's) moved the outline enough to shift a speed by 7%.
    ``size`` and ``gray`` are as for iter_frames.
    """
    def shaped(f: np.ndarray) -> np.ndarray:
        if size is not None and (f.shape[1], f.shape[0]) != tuple(size):
            f = cv2.resize(f, tuple(size), interpolation=cv2.INTER_AREA)
        if gray and f.ndim == 3:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return f

    def decoded():
        """Frames from the fast reader, each still to be finished."""
        if gray:
            return ((f, None) for f in iter_frames(path, size, gray))
        return ((None, raw) for raw in iter_raw_frames(path))

    def done(item) -> np.ndarray:
        f, raw = item
        return shaped(f if raw is None else finish(raw, size))

    frames: list[np.ndarray] = []
    if total <= 0:
        for item in decoded():
            frames.append(done(item))
            if len(frames) >= n:
                break
        return frames
    wanted = np.linspace(0, max(total - 1, 0), min(n, max(total, 1))).astype(int)
    keep = set(int(i) for i in wanted)
    last = int(wanted[-1])
    if total / max(len(wanted), 1) <= SEQUENTIAL_SAMPLE_STRIDE:
        if gray or fast:
            # Grey frames come scaled from the decoder; colour ones are read
            # full size and finished exactly as the per-frame pass does.
            for idx, item in enumerate(decoded()):
                if idx in keep:
                    frames.append(done(item))
                if idx >= last:
                    break
            return frames
        cap = cv2.VideoCapture(path)
        try:
            idx = 0
            while idx <= last and cap.grab():
                if idx in keep:
                    ok, f = cap.retrieve()
                    if ok:
                        frames.append(shaped(f))
                idx += 1
        finally:
            cap.release()
        return frames
    if not exact:
        got = keyframes(path, len(wanted), size if gray else None, gray)
        if got:
            return [shaped(f) for f in got]
    cap = cv2.VideoCapture(path)
    try:
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                frames.append(shaped(f))
    finally:
        cap.release()
    return frames


def analyze(
    path: str,
    cfg: Config | None = None,
    *,
    net_quad: np.ndarray | None = None,
    progress=None,
) -> SessionResult:
    """Run the full pipeline over a clip."""
    cfg = cfg or Config()
    started = time.perf_counter()
    info = probe(path, cfg)
    warnings: list[str] = []

    if info.fps_source == "assumed":
        warnings.append("the file did not report a usable frame rate; assumed 30 fps, so speeds are unreliable")
    warnings.extend(info.notes)

    def report(stage: str, frac: float) -> None:
        if progress:
            progress(stage, frac)

    # --- pass 1: goal outline and background plate ---------------------
    report("sampling", 0.02)
    n_sample = max(cfg.net.sample_frames, cfg.puck.bg_sample_frames)
    samples = sample_frames(path, n_sample, info.frame_count, exact=cfg.puck.stabilize)
    if not samples:
        raise RuntimeError(f"no frames could be read from {path}")

    report("finding the net", 0.12)
    warnings.extend(_capture_warnings(info, cfg))
    if net_quad is not None:
        from .geometry import order_quad

        net = NetDetection(quad=order_quad(np.asarray(net_quad, dtype=np.float64)),
                           confidence=1.0, method="manual",
                           frames_used=len(samples), frames_tried=len(samples))
    else:
        net_samples = samples[:: max(1, len(samples) // max(cfg.net.sample_frames, 1))][: cfg.net.sample_frames]
        net_notes: list[str] = []
        net = detect_net(net_samples, cfg, net_notes)
        net_samples = None
        warnings.extend(net_notes)

    if net is None:
        return SessionResult(video=info, net=None, warnings=warnings, config=cfg,
                             elapsed_s=time.perf_counter() - started)
    warnings.extend(net.notes)

    known = known_focal_px(info, cfg)
    K = (np.array([[known[0], 0.0, info.width / 2.0], [0.0, known[0], info.height / 2.0], [0.0, 0.0, 1.0]])
         if known else None)
    seen_quad = np.array(net.quad, dtype=np.float64)
    feet_note = None
    if known and cfg.net.restore_hidden_feet:
        restored = hidden_feet(net.quad, K, cfg.goal)
        if restored is not None:
            net.quad, hidden = restored
            g = cfg.goal
            feet_note = (
                f"the bottom {hidden:.0f} in of the posts look hidden -- grass, seen from low down -- so the goal's "
                f"feet were put where a {g.mouth_width_in:.0f} x {g.mouth_height_in:.0f} in goal's must be. If the "
                f"opening really is about {g.mouth_width_in:.0f} x {g.mouth_height_in - hidden:.0f} in, enter that"
            )

    plane = GoalPlane(net.quad, cfg.goal)
    cam = calibrate_from_homography(
        plane.H,
        (info.width, info.height),
        outer_rect(cfg.goal),
        plane.image_quad,
        assumed_focal_px=cfg.camera.assumed_focal_frac * info.width,
        known_focal_px=known[0] if known else None,
        known_focal_spread=known[1] if known else 0.0,
    )
    if cfg.camera.height_in is not None:
        if not known:
            warnings.append("the phone's height is only used with the lens known; pick the lens you filmed with")
        elif cam is not None and cam.pose_from_corners:
            # The outline settled where the phone was on its own; a height
            # misremembered must not override it (from 56 in up, entered as
            # on the ground, a synthetic view read speeds 26-69% fast).
            if abs(float(cam.position[1]) - cfg.camera.height_in) > 24.0:
                warnings.append(
                    f"the net's outline puts the phone about {cam.position[1]:.0f} in off the ground, not the "
                    f"{cfg.camera.height_in:.0f} in entered; the outline was clear enough to go by"
                )
        else:
            held = _held_camera(seen_quad, K, cfg, warnings)
            if held is not None:
                cam, quad, feet_note = held
                net.quad = quad
                plane = GoalPlane(net.quad, cfg.goal)
    if feet_note:
        warnings.append(feet_note)
    if cam is not None and known:
        warnings.append(f"lens: {known[2]}")
        note = _goal_size_note(plane.image_quad, cam.K, cfg)
        if note:
            warnings.append(note)
    if cam is None:
        warnings.append(
            "could not recover the camera geometry from the goal outline; speed falls back to "
            "pixel motion at the goal plane"
        )
    elif cam.focal_assumed:
        warnings.append(
            "the camera is too square-on to the net for the goal outline to reveal its lens, so a "
            "typical phone lens was assumed and speed carries a few percent of extra error; pass "
            "your camera's field of view, or move the camera further to one side, to remove the guess"
        )
    elif not known:
        # Solved from the outline itself, which grass hiding the posts' feet
        # throws off: on a synthetic lawn the lens came out wrong enough to
        # read speeds 7-10% high, and the hidden strip cannot be measured.
        what = "Slow motion does not record which lens filmed it" if info.fps_source == "slow-motion" \
            else "The video did not say which lens filmed it"
        warnings.append(
            f"{what}, so it was worked out from the net's outline, which grass over the bottom of the "
            "posts can throw off by 10-15% and the speeds with it. Pick the lens you filmed with in the form"
        )

    report("checking for camera drift", 0.2)
    motion = CameraMotion()
    if cfg.puck.stabilize and info.frame_count > 0:
        sampled = list(range(0, info.frame_count, max(cfg.puck.stabilize_sample_stride, 1)))
        motion = measure(
            path, sampled, drift_threshold_px=cfg.puck.stabilize_drift_threshold_px
        )
        warnings.extend(motion.notes)
        if motion.needed:
            motion = interpolate(motion, info.frame_count)
        else:
            motion = CameraMotion()

    report("building background", 0.22)
    scale = min(1.0, cfg.puck.work_width / float(info.width))
    # The plate must be built from frames that agree on where the scene is.
    n_plate = min(cfg.puck.bg_sample_frames, max(info.frame_count, 1))
    bg_indices = np.linspace(0, max(info.frame_count - 1, 0), n_plate).astype(int)
    work_size = (int(round(info.width * scale)), int(round(info.height * scale))) if scale < 1.0 else None
    # With grey_decode the detector works on grey frames decoded straight at
    # working size (see the config).  Either way the plate is read exactly the
    # way the per-frame pass below reads, so the two never differ by how they
    # were decoded.
    grey = cfg.puck.grey_decode and cfg.puck.method == "median" and not motion.needed
    if not motion.needed:
        # The net is found, and the plate is read afresh: let the full-size
        # samples go first (48 of them are 1.2 GB at 4K).
        samples = None
        small_samples = sample_frames(path, n_plate, info.frame_count, exact=cfg.puck.stabilize,
                                      size=work_size, gray=grey, fast=True)
    else:
        small_samples = []
        for f, idx in zip(samples[: cfg.puck.bg_sample_frames], bg_indices):
            img = motion.compensate(f, int(idx)) if motion.needed else f
            small_samples.append(
                cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else img
            )
        samples = None
    background, noise = (build_background_and_noise(small_samples) if cfg.puck.method == "median"
                         else (None, None))

    goal_width_px = float(np.linalg.norm(net.quad[1] - net.quad[0]))
    # How big a puck looks at the goal plane, in working-resolution pixels.
    puck_px_small = plane.px_per_inch_at(0.0, 24.0) * PUCK_DIAMETER_IN * scale
    detector = PuckDetector(cfg, background, puck_px=puck_px_small, scale=scale, noise=noise)

    # --- pass 2: puck candidates, every frame --------------------------
    report("tracking the puck", 0.3)
    cands_by_frame: dict[int, list[Candidate]] = {}
    # Keep everything for now when clutter is to be sorted out over the
    # whole clip; the per-frame cap is applied after that.
    raw_cap = cfg.puck.raw_candidates_per_frame if cfg.puck.demote_recurring else None
    def one_frame(frame, i: int) -> list[Candidate]:
        if grey:
            return detector.detect(frame, i, limit=raw_cap)
        if not motion.needed:
            return detector.detect(finish(frame, scale=scale), i, limit=raw_cap)
        aligned = motion.compensate(finish(frame), i)
        small = aligned
        if scale < 1.0:
            small = cv2.resize(aligned, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return detector.detect(small, i, limit=raw_cap)

    # Frames are independent against a fixed plate, and OpenCV lets go of
    # Python while it works, so several are shrunk and searched at once
    # while the next ones decode.  The background-subtractor fallback keeps
    # state from frame to frame and so stays one at a time.
    workers = 1 if detector.stateful else max(1, min(cfg.puck.workers or (os.cpu_count() or 1), 8))
    idx = 0
    # Colour frames come as decoded; matching, shrinking and turning them
    # happens on the workers (see video.finish), leaving only the decoding
    # here -- half of a 4K frame's cost used to be spent on this thread.
    frames_in = iter_frames(path, work_size, gray=True) if grey else iter_raw_frames(path)
    if workers == 1:
        for frame in frames_in:
            found = one_frame(frame, idx)
            if found:
                cands_by_frame[idx] = found
            idx += 1
            if progress and info.frame_count and idx % 60 == 0:
                report("tracking the puck", 0.3 + 0.5 * idx / max(info.frame_count, 1))
    else:
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending: deque = deque()

            def collect() -> None:
                i, fut = pending.popleft()
                found = fut.result()
                if found:
                    cands_by_frame[i] = found

            for frame in frames_in:
                pending.append((idx, pool.submit(one_frame, frame, idx)))
                idx += 1
                while len(pending) > 2 * workers:
                    collect()
                if progress and info.frame_count and idx % 60 == 0:
                    report("tracking the puck", 0.3 + 0.5 * idx / max(info.frame_count, 1))
            while pending:
                collect()
    if idx and idx != info.frame_count:
        info.frame_count = idx

    report("assembling shots", 0.85)
    raw_xy = {f: np.array([(c.x, c.y) for c in v], dtype=np.float64) for f, v in cands_by_frame.items()}
    if cfg.puck.demote_recurring:
        n_frames = max(len(cands_by_frame), 1)
        crowd = cfg.track.clutter_radius_frac * goal_width_px if cfg.puck.demote_crowded else None
        cands_by_frame, _, busy = demote_recurring(
            cands_by_frame, puck_px_small / scale, info.fps, cfg, crowd)
        saturated = busy / n_frames
    else:
        saturated = sum(
            1 for v in cands_by_frame.values() if len(v) >= cfg.puck.max_candidates_per_frame
        ) / max(len(cands_by_frame), 1)
    if saturated > cfg.puck.saturated_frame_warn_frac:
        warnings.append(_busy_scene_warning(saturated, small_samples, scale, motion.needed, cfg))
    tracks = build_tracks(cands_by_frame, goal_width_px, cfg, warnings, fps=info.fps)
    tracks = filter_by_speed(tracks, goal_width_px, info.fps, cfg)
    tracks, in_clutter = filter_by_clutter(tracks, raw_xy, goal_width_px, info.fps, cfg)
    window = max(cfg.puck.recurring_min_window_frames, int(round(cfg.puck.recurring_window_s * info.fps)))
    gap = max(2, int(round(0.0125 * info.fps)))
    settling = cfg.track.revisit_check_first_s * info.fps
    in_place = [t for t in tracks if t.start_frame < settling
                and revisits(t, raw_xy, puck_px_small / scale, window, gap) > cfg.track.max_revisit_frac]
    tracks = [t for t in tracks if not any(t is p for p in in_place)]
    near = _near_goal(plane, goal_width_px, cfg)
    tracks = [until_impact(t, info.fps, cfg, near) for t in tracks]

    zones = build_zones(cfg.goal)
    shots: list[Shot] = []
    track_of: dict[int, Track] = {}
    started_at_goal = 0
    slow = 0
    for t in tracks:
        if not approaches_goal(t, plane, cfg):
            started_at_goal += 1
            continue
        s = shot_from_track(len(shots), t, plane, cam, info.fps, cfg, zones)
        if s is not None and _too_slow_for_a_shot(s, cfg):
            slow += 1
            continue
        if s is not None:
            shots.append(s)
            track_of[id(s)] = t
    shots = dedupe_shots(shots, cfg, info.fps)
    # One trail per reported shot, in the same order, so a trail is never
    # drawn for something that was merged away.
    kept_tracks = [track_of[id(s)] for s in shots]

    if slow:
        warnings.append(
            f"not counted as shots: {slow} slow mover(s), under {cfg.shot.min_shot_mph:.0f} mph by the timed "
            "flight -- skating, stickhandling, a puck rolling in. A shot at a net is faster than that."
        )
    if in_clutter:
        warnings.append(
            f"not counted as shots: {in_clutter} short line(s) of flickers in busy background -- leaves "
            "moving in the sun, netting rippling -- that happened to line up. A puck crosses clear "
            "background for most of its flight."
        )
    if in_place:
        warnings.append(
            f"not counted as shots: {len(in_place)} line(s) of things flickering where they stand -- lamps, "
            "posts, edges -- in the first moments of the clip while the camera settled, that happened "
            "to line up. A puck passes any spot once."
        )
    if started_at_goal:
        warnings.append(
            f"not counted as shots: {started_at_goal} bit(s) of movement that started on the goal itself, "
            "such as the posts glinting or the netting moving. A shot has to come from somewhere."
        )

    if not shots:
        if tracks:
            warnings.append(
                f"found {len(tracks)} moving object(s) but none finished at the net; "
                "if shots were taken, the puck may be too small or too blurred to follow"
            )
        else:
            warnings.append("no puck trajectories were found in this clip")

    note = _distance_check(shots, cfg)
    if note:
        warnings.append(note)

    if cfg.speed.shot_distance_ft is None and shots:
        warnings.append(
            "no shooting distance was given, so speed came from the weaker estimators; "
            "supplying it (even roughly) tightens the number considerably"
        )

    if motion.needed:
        warnings.append(
            "positions are reported against the clip's own steady view, since the camera moved; "
            "the marked net follows that view rather than each frame."
        )

    report("done", 1.0)
    return SessionResult(
        video=info,
        net=net,
        shots=shots,
        tracks=kept_tracks,
        camera=cam,
        warnings=warnings,
        config=cfg,
        elapsed_s=time.perf_counter() - started,
    )
