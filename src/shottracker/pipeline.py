"""Video in, session out.

Two passes over the clip.  The first samples frames to find the goal and build
the background plate; the second walks every frame looking for the puck.  The
goal is found before the puck on purpose — its size in pixels is what tells the
puck detector how big a puck should look, and its plane is what turns a pixel
into a spot on the net.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import CameraModel, calibrate_from_homography
from .config import PUCK_DIAMETER_IN, Config
from .geometry import GoalPlane, build_zones, mouth_outline, outer_outline, outer_rect
from .net_detect import NetDetection, detect_net
from .puck_detect import Candidate, PuckDetector, build_background
from .shots import Shot, dedupe_shots, shot_from_track
from .stabilize import CameraMotion, interpolate, measure
from .tracking import Track, build_tracks, filter_by_speed


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    fps_source: str = "container"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration_s": round(self.frame_count / self.fps, 2) if self.fps else None,
            "fps_source": self.fps_source,
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

    source = "container"
    if cfg.fps_override:
        fps, source = float(cfg.fps_override), "override"
    elif fps <= 1.0:
        fps, source = 30.0, "assumed"
    return VideoInfo(path=path, width=width, height=height, fps=fps, frame_count=max(count, 0), fps_source=source)


def sample_frames(path: str, n: int, total: int) -> list[np.ndarray]:
    """Evenly spaced frames, read by seeking so long clips stay cheap."""
    cap = cv2.VideoCapture(path)
    frames: list[np.ndarray] = []
    try:
        if total <= 0:
            while len(frames) < n:
                ok, f = cap.read()
                if not ok:
                    break
                frames.append(f)
            return frames
        for idx in np.linspace(0, max(total - 1, 0), min(n, max(total, 1))).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                frames.append(f)
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

    def report(stage: str, frac: float) -> None:
        if progress:
            progress(stage, frac)

    # --- pass 1: goal outline and background plate ---------------------
    report("sampling", 0.02)
    n_sample = max(cfg.net.sample_frames, cfg.puck.bg_sample_frames)
    samples = sample_frames(path, n_sample, info.frame_count)
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
        warnings.extend(net_notes)

    if net is None:
        return SessionResult(video=info, net=None, warnings=warnings, config=cfg,
                             elapsed_s=time.perf_counter() - started)
    warnings.extend(net.notes)

    plane = GoalPlane(net.quad, cfg.goal)
    cam = calibrate_from_homography(
        plane.H,
        (info.width, info.height),
        outer_rect(cfg.goal),
        plane.image_quad,
        assumed_focal_px=cfg.camera.assumed_focal_frac * info.width,
    )
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
    small_samples = []
    for f, idx in zip(samples[: cfg.puck.bg_sample_frames], bg_indices):
        img = motion.compensate(f, int(idx)) if motion.needed else f
        small_samples.append(
            cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else img
        )
    background = build_background(small_samples) if cfg.puck.method == "median" else None

    goal_width_px = float(np.linalg.norm(net.quad[1] - net.quad[0]))
    # How big a puck looks at the goal plane, in working-resolution pixels.
    puck_px_small = plane.px_per_inch_at(0.0, 24.0) * PUCK_DIAMETER_IN * scale
    detector = PuckDetector(cfg, background, puck_px=puck_px_small, scale=scale)

    # --- pass 2: puck candidates, every frame --------------------------
    report("tracking the puck", 0.3)
    cands_by_frame: dict[int, list[Candidate]] = {}
    cap = cv2.VideoCapture(path)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            aligned = motion.compensate(frame, idx) if motion.needed else frame
            small = aligned
            if scale < 1.0:
                small = cv2.resize(aligned, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            found = detector.detect(small, idx)
            if found:
                cands_by_frame[idx] = found
            idx += 1
            if progress and info.frame_count and idx % 60 == 0:
                report("tracking the puck", 0.3 + 0.5 * idx / max(info.frame_count, 1))
    finally:
        cap.release()
    if idx and idx != info.frame_count:
        info.frame_count = idx

    report("assembling shots", 0.85)
    saturated = sum(
        1 for v in cands_by_frame.values() if len(v) >= cfg.puck.max_candidates_per_frame
    ) / max(len(cands_by_frame), 1)
    if saturated > cfg.puck.saturated_frame_warn_frac:
        warnings.append(
            f"{saturated:.0%} of frames were full of moving objects, which means the background "
            "model is not holding. The usual cause is a camera that was hand-held rather than "
            "propped up; puck detections will be unreliable."
        )
    tracks = build_tracks(cands_by_frame, goal_width_px, cfg, warnings)
    tracks = filter_by_speed(tracks, goal_width_px, info.fps, cfg)

    zones = build_zones(cfg.goal)
    shots: list[Shot] = []
    kept_tracks: list[Track] = []
    for t in tracks:
        s = shot_from_track(len(shots), t, plane, cam, info.fps, cfg, zones)
        if s is not None:
            shots.append(s)
            kept_tracks.append(t)
    shots = dedupe_shots(shots, cfg)

    if not shots:
        if tracks:
            warnings.append(
                f"found {len(tracks)} moving object(s) but none finished at the net; "
                "if shots were taken, the puck may be too small or too blurred to follow"
            )
        else:
            warnings.append("no puck trajectories were found in this clip")

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
