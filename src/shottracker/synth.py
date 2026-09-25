"""Synthetic shooting footage with exact ground truth.

Real hockey video is easy to get and impossible to grade: nobody knows what the
puck was truly doing.  So the pipeline is developed against rendered clips where
the shot speed and the impact point are known to the inch, using a pinhole
camera and a ballistic puck.

World frame: the goal plane is z = 0, x runs image-right along the goal line,
y is up from the ice, and z increases toward the shooter.  Units are inches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import GRAVITY_IN_S2, IN_PER_SEC_TO_MPH, GoalSpec
from .geometry import mouth_outline, outer_outline


@dataclass
class Camera:
    position: np.ndarray
    target: np.ndarray
    width: int = 1280
    height: int = 720
    hfov_deg: float = 62.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.target = np.asarray(self.target, dtype=np.float64)
        fwd = self.target - self.position
        fwd = fwd / np.linalg.norm(fwd)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd])  # world -> camera
        self.fx = (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        self.fy = self.fx
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def project(self, pts_world: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
        cam = (self.R @ (pts - self.position).T).T
        z = np.clip(cam[:, 2], 1e-3, None)
        u = self.fx * cam[:, 0] / z + self.cx
        v = self.fy * cam[:, 1] / z + self.cy
        return np.stack([u, v], axis=1)

    def depth(self, pt_world: np.ndarray) -> float:
        cam = self.R @ (np.asarray(pt_world, dtype=np.float64) - self.position)
        return float(cam[2])


def camera_preset(name: str, width: int = 1280, height: int = 720) -> Camera:
    """Camera placements that match how people actually film themselves shooting.

    "backyard" is the first real slow-motion clip's placement, as the tracker
    measured it: a phone propped on the ground 27 ft out and 12 ft to the side,
    8 in up, filming in portrait past the shooter.  It wants a portrait size
    (see ``BACKYARD_SIZE``).
    """
    presets = {
        # Phone on the boards off to one side: pixel motion tracks the puck.
        "side": (np.array([235.0, 50.0, 190.0]), np.array([0.0, 22.0, 0.0])),
        # A deliberately punishing angle, used to stress the detector.
        "extreme_side": (np.array([320.0, 52.0, 150.0]), np.array([0.0, 22.0, 0.0])),
        # The common case: phone behind and to one side of the shooter.
        "angled": (np.array([170.0, 58.0, 300.0]), np.array([0.0, 22.0, 0.0])),
        # Phone directly behind the shooter, square to the net.
        "head_on": (np.array([14.0, 56.0, 340.0]), np.array([0.0, 22.0, 0.0])),
        # Phone propped on the ground, behind and beside the shooter.
        "backyard": (np.array([144.0, 8.0, 328.0]), np.array([-10.0, 16.0, 0.0])),
    }
    if name not in presets:
        raise ValueError(f"unknown camera preset {name!r}; have {sorted(presets)}")
    pos, tgt = presets[name]
    if name == "backyard":
        # Portrait, with the goal at the ~177 px in 1080 the real clip showed.
        return Camera(pos, tgt, width=width, height=height, hfov_deg=70.0)
    return Camera(pos, tgt, width=width, height=height)


BACKYARD_SIZE = (540, 960)


@dataclass
class SynthShot:
    """One shot, specified by where it starts, where it lands and how long it takes."""

    target_x_in: float
    target_y_in: float
    release_z_in: float = 240.0       # 20 ft out
    release_x_in: float = 0.0
    release_y_in: float = 0.5
    flight_time_s: float = 0.22
    start_time_s: float = 0.5
    gravity: bool = True

    # Filled in by the renderer.
    v0: np.ndarray = field(default=None, repr=False)

    def solve(self) -> None:
        p0 = np.array([self.release_x_in, self.release_y_in, self.release_z_in])
        p1 = np.array([self.target_x_in, self.target_y_in, 0.0])
        t = self.flight_time_s
        v = (p1 - p0) / t
        if self.gravity:
            v = v + np.array([0.0, 0.5 * GRAVITY_IN_S2 * t, 0.0])
        self.v0 = v

    @property
    def release_speed_mph(self) -> float:
        if self.v0 is None:
            self.solve()
        return float(np.linalg.norm(self.v0) * IN_PER_SEC_TO_MPH)

    def position_at(self, dt: float) -> np.ndarray:
        if self.v0 is None:
            self.solve()
        p0 = np.array([self.release_x_in, self.release_y_in, self.release_z_in])
        p = p0 + self.v0 * dt
        if self.gravity:
            p = p - np.array([0.0, 0.5 * GRAVITY_IN_S2 * dt * dt, 0.0])
        return p


# --- scene drawing ----------------------------------------------------------


def _fill_poly_world(img, cam: Camera, pts_world, color) -> None:
    pts = cam.project(np.asarray(pts_world, dtype=np.float64))
    cv2.fillPoly(img, [np.round(pts).astype(np.int32)], color, lineType=cv2.LINE_AA)


def _line_world(img, cam: Camera, a, b, color, thickness=2) -> None:
    p = cam.project(np.array([a, b], dtype=np.float64))
    cv2.line(
        img,
        tuple(np.round(p[0]).astype(int)),
        tuple(np.round(p[1]).astype(int)),
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_scene(
    cam: Camera,
    goal: GoalSpec,
    *,
    ice_markings: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """The static part of the frame: ice, boards, markings and the goal."""
    rng = rng or np.random.default_rng(0)
    img = np.full((cam.height, cam.width, 3), (232, 230, 226), dtype=np.uint8)

    # Boards and a dasher line far behind the net.
    _fill_poly_world(
        img,
        cam,
        [(-900, 0, -420), (900, 0, -420), (900, 42, -420), (-900, 42, -420)],
        (225, 222, 216),
    )
    _fill_poly_world(
        img,
        cam,
        [(-900, 42, -420), (900, 42, -420), (900, 200, -420), (-900, 200, -420)],
        (196, 186, 176),
    )

    if ice_markings:
        # A real rink puts a red goal line right at the posts and a blue line
        # up-ice.  Both are strong red/blue distractors for net detection.
        goal_line = [(-900, 0.01, -1.0), (900, 0.01, -1.0), (900, 0.01, 1.0), (-900, 0.01, 1.0)]
        blue_line = [(-900, 0.01, 299), (900, 0.01, 299), (900, 0.01, 311), (-900, 0.01, 311)]
        _fill_poly_world(img, cam, goal_line, (40, 40, 205))
        _fill_poly_world(img, cam, blue_line, (170, 70, 40))
        # Faceoff dot, up-ice of the shooter.
        cx = cam.project(np.array([[-180.0, 0.02, 420.0]]))[0]
        r = max(2, int(0.06 * cam.fx / max(cam.depth(np.array([-180.0, 0.0, 420.0])), 1.0) * 30))
        cv2.circle(img, tuple(np.round(cx).astype(int)), r, (40, 40, 200), -1, cv2.LINE_AA)

    hw_in = goal.mouth_width_in / 2.0
    y_mouth = goal.mouth_height_in
    depth = -40.0  # the net bag extends behind the goal line

    # Net bag: dark interior with white mesh, drawn before the pipe.
    back = [(-hw_in, y_mouth, depth), (hw_in, y_mouth, depth), (hw_in, 0, depth), (-hw_in, 0, depth)]
    left = [(-hw_in, y_mouth, 0), (-hw_in, y_mouth, depth), (-hw_in, 0, depth), (-hw_in, 0, 0)]
    right = [(hw_in, y_mouth, 0), (hw_in, y_mouth, depth), (hw_in, 0, depth), (hw_in, 0, 0)]
    roof = [(-hw_in, y_mouth, 0), (hw_in, y_mouth, 0), (hw_in, y_mouth, depth), (-hw_in, y_mouth, depth)]
    _fill_poly_world(img, cam, back, (118, 116, 112))
    _fill_poly_world(img, cam, left, (138, 136, 132))
    _fill_poly_world(img, cam, right, (138, 136, 132))
    _fill_poly_world(img, cam, roof, (150, 148, 144))

    mesh = (208, 206, 202)
    for gx in np.arange(-hw_in, hw_in + 0.1, 3.0):
        _line_world(img, cam, (gx, 0, depth), (gx, y_mouth, depth), mesh, 1)
    for gy in np.arange(0, y_mouth + 0.1, 3.0):
        _line_world(img, cam, (-hw_in, gy, depth), (hw_in, gy, depth), mesh, 1)

    # The red pipe, bent where the posts meet the crossbar the way a real frame
    # is.  Drawing square corners here would let the detector off the hook: the
    # corner it has to recover is where the straight sections *would* meet, a
    # point that does not physically exist on the goal.
    red = (36, 34, 196)
    out2d = outer_outline(goal)
    in2d = mouth_outline(goal)
    outer = np.hstack([out2d, np.zeros((len(out2d), 1))])
    inner = np.hstack([in2d, np.zeros((len(in2d), 1))])
    ring = np.vstack([outer, inner[::-1]])
    _fill_poly_world(img, cam, ring, red)

    noise = rng.normal(0.0, 3.0, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _draw_puck(img: np.ndarray, cam: Camera, pos: np.ndarray, *, blur_steps: int, prev: np.ndarray | None) -> None:
    """Draw the puck, smearing it between sub-frame positions like a real shutter."""
    if prev is None:
        positions = [pos]
    else:
        positions = [prev + (pos - prev) * f for f in np.linspace(0.35, 1.0, blur_steps)]
    layer = img.copy()
    for p in positions:
        z = cam.depth(p)
        if z <= 1.0:
            continue
        uv = cam.project(p.reshape(1, 3))[0]
        r = max(1.0, cam.fx * 1.5 / z)
        cv2.ellipse(
            layer,
            (int(round(uv[0])), int(round(uv[1]))),
            (int(round(r)), max(1, int(round(r * 0.55)))),
            0, 0, 360, (26, 24, 24), -1, cv2.LINE_AA,
        )
    cv2.addWeighted(layer, 0.85, img, 0.15, 0, dst=img)


def render_session(
    path: str,
    shots: list[SynthShot],
    *,
    camera: Camera | str = "angled",
    goal: GoalSpec | None = None,
    fps: float = 120.0,
    duration_s: float | None = None,
    seed: int = 0,
    ice_markings: bool = True,
    distractor: bool = False,
    net_sway: bool = False,
    foliage: bool = False,
) -> dict:
    """Render a clip and return its ground truth.

    ``distractor`` adds a dark blob drifting across the scene at stick speed,
    to check that trajectory filtering does not mistake it for a puck.

    ``net_sway`` shakes the netting for a moment after every impact, the way a
    real net keeps moving once the puck has stopped; ``foliage`` puts a bush
    beside the goal whose leaves never keep still.  Both are what a backyard
    adds to a rink, and both made fake shots on real footage.
    """
    goal = goal or GoalSpec()
    if isinstance(camera, str):
        cam = camera_preset(camera, *BACKYARD_SIZE) if camera == "backyard" else camera_preset(camera)
    else:
        cam = camera
    rng = np.random.default_rng(seed)

    for s in shots:
        s.solve()
    if duration_s is None:
        duration_s = max(s.start_time_s + s.flight_time_s for s in shots) + 0.45

    n_frames = int(round(duration_s * fps))
    background = draw_scene(cam, goal, ice_markings=ice_markings, rng=rng)

    # The netting's footprint on screen: the bag behind the mouth.
    hw_in, depth = goal.mouth_width_in / 2.0, -40.0
    bag = cam.project(np.array([(-hw_in, goal.mouth_height_in, depth), (hw_in, goal.mouth_height_in, depth),
                                (hw_in, 0.0, depth), (-hw_in, 0.0, depth)]))
    net_mask = np.zeros(background.shape[:2], np.uint8)
    cv2.fillPoly(net_mask, [np.round(bag).astype(np.int32)], 255)
    net_mask = net_mask.astype(bool)
    impacts = [s.start_time_s + s.flight_time_s for s in shots]

    # A bush just beside the goal: dark green, with red flowers like the real one.
    leaves = []
    if foliage:
        for _ in range(60):
            p = np.array([goal.outer_width_in / 2 + rng.uniform(8, 60), rng.uniform(4, 70), rng.uniform(-60, -10)])
            leaves.append((p, rng.uniform(0, 2 * np.pi), (40, 110, 40) if rng.random() < 0.8 else (60, 40, 200)))

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cam.width, cam.height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open {path} for writing")

    truth_shots = []
    for s in shots:
        truth_shots.append(
            {
                "release_speed_mph": s.release_speed_mph,
                "impact_x_in": s.target_x_in,
                "impact_y_in": s.target_y_in,
                "release_frame": int(round(s.start_time_s * fps)),
                "impact_frame": int(round((s.start_time_s + s.flight_time_s) * fps)),
                "shot_distance_in": float(np.linalg.norm(
                    np.array([s.target_x_in - s.release_x_in, s.target_y_in - s.release_y_in, -s.release_z_in])
                )),
                "release_z_in": s.release_z_in,
            }
        )

    try:
        for i in range(n_frames):
            t = i / fps
            frame = background.copy()
            frame = np.clip(
                frame.astype(np.float32) + rng.normal(0.0, 2.2, frame.shape).astype(np.float32), 0, 255
            ).astype(np.uint8)

            if distractor:
                # Something dark, big-ish and slow: a stick blade sweeping
                # past, and in a long clip sweeping past again every 3.6 s
                # (drifting on for good, it ended up behind the camera).
                dx = -260.0 + 150.0 * (t % 3.6)
                p = np.array([dx, 6.0, 250.0])
                z = cam.depth(p)
                if z > 12.0:
                    uv = cam.project(p.reshape(1, 3))[0]
                    r = int(max(3, cam.fx * 4.0 / z))
                    cv2.ellipse(
                        frame, (int(uv[0]), int(uv[1])), (r * 2, r), 20, 0, 360, (30, 30, 30), -1, cv2.LINE_AA
                    )

            if net_sway:
                # A damped shake of the mesh for a third of a second after each hit.
                for t_hit in impacts:
                    age = t - t_hit
                    if 0.0 < age < 0.35:
                        amp = 3.0 * np.exp(-age / 0.12)
                        dx = int(round(amp * np.sin(2 * np.pi * 9.0 * age)))
                        dy = int(round(0.6 * amp * np.cos(2 * np.pi * 7.0 * age)))
                        shifted = np.roll(np.roll(background, dx, axis=1), dy, axis=0)
                        frame[net_mask] = shifted[net_mask]

            for p, phase, color in leaves:
                sway = np.array([2.5 * np.sin(2 * np.pi * 1.7 * t + phase),
                                 1.5 * np.cos(2 * np.pi * 2.3 * t + phase), 0.0])
                uv = cam.project((p + sway).reshape(1, 3))[0]
                r = max(2, int(cam.fx * 2.5 / max(cam.depth(p), 1.0)))
                cv2.circle(frame, (int(uv[0]), int(uv[1])), r, color, -1, cv2.LINE_AA)

            for s in shots:
                dt = t - s.start_time_s
                if dt < 0 or dt > s.flight_time_s:
                    continue
                prev_dt = max(dt - 1.0 / fps, 0.0)
                prev = s.position_at(prev_dt) if dt > 0 else None
                _draw_puck(frame, cam, s.position_at(dt), blur_steps=5, prev=prev)

            writer.write(frame)
    finally:
        writer.release()

    outer = np.array(
        [
            [-goal.outer_width_in / 2, goal.outer_height_in, 0.0],
            [goal.outer_width_in / 2, goal.outer_height_in, 0.0],
            [goal.outer_width_in / 2, 0.0, 0.0],
            [-goal.outer_width_in / 2, 0.0, 0.0],
        ]
    )
    return {
        "path": path,
        "fps": fps,
        "frames": n_frames,
        "size": [cam.width, cam.height],
        "net_quad": cam.project(outer).tolist(),
        "shots": truth_shots,
    }
