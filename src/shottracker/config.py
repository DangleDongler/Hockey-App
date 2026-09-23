"""Tunables and real-world constants for the shot tracker.

All real-world lengths are in INCHES and all speeds are reported in MPH.
Working in inches keeps the goal-plane homography numerically well-scaled and
matches how hockey equipment is actually specified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# --- Real-world constants (NHL / IIHF regulation) ---------------------------

GOAL_MOUTH_WIDTH_IN = 72.0   # 6 ft between the inside edges of the posts
GOAL_MOUTH_HEIGHT_IN = 48.0  # 4 ft from the ice to the underside of the crossbar
POST_DIAMETER_IN = 2.375     # 2 3/8" goal pipe
# Posts and crossbar are joined by a bend, not a mitre, so the mouth's top
# corners are rounded.  Approximate: it varies between makes, and it is only
# used for drawing and for calling corner hits -- the scale comes from where
# the straight sections of pipe would meet, which the radius does not move.
CORNER_RADIUS_IN = 4.0
PUCK_DIAMETER_IN = 3.0
PUCK_THICKNESS_IN = 1.0

IN_PER_SEC_TO_MPH = 3600.0 / 63360.0  # 1 in/s == 0.0568 mph
GRAVITY_IN_S2 = 386.088               # 9.80665 m/s^2, in inches


@dataclass
class GoalSpec:
    """Physical dimensions of the goal being shot at.

    The tracker reconstructs a plane attached to the goal mouth.  ``mouth_*``
    is the opening a puck can pass through; ``outer_*`` adds the pipe, which is
    what an edge detector actually latches onto.
    """

    mouth_width_in: float = GOAL_MOUTH_WIDTH_IN
    mouth_height_in: float = GOAL_MOUTH_HEIGHT_IN
    post_diameter_in: float = POST_DIAMETER_IN
    corner_radius_in: float = CORNER_RADIUS_IN

    @property
    def outer_width_in(self) -> float:
        # A post on each side of the mouth.
        return self.mouth_width_in + 2.0 * self.post_diameter_in

    @property
    def outer_height_in(self) -> float:
        # Crossbar sits on top of the mouth; the posts stand on the ice.
        return self.mouth_height_in + self.post_diameter_in


@dataclass
class NetDetectConfig:
    """Automatic goal-outline detection."""

    # Frames sampled across the clip to vote on a single static outline.
    sample_frames: int = 24
    work_width: int = 960          # detection resolution; results are rescaled

    # Red goal pipe in HSV.  Red wraps the hue circle so we need two bands.
    hue_low_max: int = 12
    hue_high_min: int = 168
    sat_min: int = 80
    val_min: int = 55

    close_kernel: int = 9          # bridge the pipe across mesh/occlusions
    min_pipe_area_frac: float = 2e-4   # of the working frame area
    min_quad_area_frac: float = 0.004  # a real goal is not a speck

    # Plausibility gates on the recovered quad, in rectified goal units.
    # Outer w/h is ~1.52 head-on and shrinks as the camera swings side-on.
    # The bound is deliberately loose: edge support, not aspect, is what
    # separates a real goal from a red jacket.
    min_aspect: float = 0.35
    max_aspect: float = 4.0
    min_edge_support: float = 0.45  # fraction of fitted edge pixels that agree
    # A goal is a frame around a hole.  Anything that fills its own bounding
    # box is a solid red object -- a jacket, a pad, a paint mark -- not a net.
    max_fill: float = 0.45
    # Frames must agree with the consensus to within this fraction of the
    # quad's diagonal, otherwise they are dropped as outliers.
    consensus_tol_frac: float = 0.06
    min_agreeing_frames: int = 3
    # Below this, say nothing rather than something wrong.  A confidently
    # reported outline that is really a flowering shrub poisons every number
    # downstream, so a refusal is the more useful answer.
    min_confidence: float = 0.35


@dataclass
class PuckDetectConfig:
    """Per-frame puck candidate extraction."""

    work_width: int = 960

    # "median" suits a phone on a tripod: the puck occupies a handful of frames,
    # so a per-pixel median over the clip is a clean, puck-free plate.
    # "mog2" is the fallback for a camera that drifts.
    method: str = "median"
    bg_sample_frames: int = 48
    diff_threshold: float = 20.0     # grey levels away from the background plate
    mog_history: int = 250
    mog_var_threshold: float = 28.0

    # Aligning hand-held frames before differencing sounds like it should help,
    # and on the one piece of real footage measured so far it did not: it raised
    # false candidates by 71% (319 -> 547 per frame). The noise there came from
    # dappled sunlight moving through leaves, which no amount of translation
    # fixes, while the resampling blur pushed more pixels over the threshold.
    # Off until footage exists where it demonstrably helps.
    stabilize: bool = False
    stabilize_sample_stride: int = 20
    stabilize_drift_threshold_px: float = 3.0

    # Size bounds as multiples of the puck's apparent area on the goal plane,
    # so they hold at any camera distance.  Motion blur smears a puck over many
    # times its own area, and it looks larger while it is nearer the camera.
    min_area_mult: float = 0.05
    max_area_mult: float = 45.0
    min_area_px: float = 5.0

    max_streak_aspect: float = 16.0  # blur stretches it; a stick blade is worse
    dark_value_max: int = 110        # vulcanized rubber is near-black
    dark_weight: float = 0.45        # how much darkness contributes to the score

    max_candidates_per_frame: int = 14
    # A frame pinned at the cap means the foreground model is failing (usually
    # a camera that moved).  Past this fraction of frames, say so.
    saturated_frame_warn_frac: float = 0.5


@dataclass
class TrackingConfig:
    """Assembling candidates into shot trajectories."""

    max_frame_gap: int = 3         # tolerate missed detections mid-flight
    # Gating radius = base + velocity-scaled term, in units of the goal width.
    gate_base_frac: float = 0.03
    gate_vel_frac: float = 0.55

    min_track_length: int = 4
    # A shot crosses ground fast; slow blobs are limbs, sticks, shadows.
    min_mean_speed_goalwidths_per_sec: float = 0.45
    # A puck in flight is very nearly a straight line in the image; the only
    # real departure is the arc gravity puts on it.
    max_line_residual_frac: float = 0.04   # of the goal width

    ransac_iterations: int = 60
    # Seed pairs grow with the square of the candidates per frame, so a bad
    # foreground model can make the search explode.  This is the safety valve.
    max_seeds: int = 150_000


@dataclass
class CameraConfig:
    """What we assume about the lens.

    The goal's own outline reveals the focal length, but only when the camera
    is off to one side; a square-on view carries no such information.  This is
    the fallback, expressed as focal length in units of image width.  0.80 is
    about 64 degrees horizontally, typical of phone video.  If you know your
    phone's field of view, setting it removes the guess.
    """

    assumed_focal_frac: float = 0.80

    @staticmethod
    def frac_from_hfov(hfov_deg: float) -> float:
        import math

        return 0.5 / math.tan(math.radians(hfov_deg) / 2.0)


@dataclass
class SpeedConfig:
    """Shot speed estimation."""

    # "time_of_flight" | "goal_plane" | "auto"
    method: str = "auto"
    # Distance from the shooting spot to the goal line, in feet.  Time-of-flight
    # needs this; it is the single number that most improves accuracy.
    shot_distance_ft: float | None = None
    # How well the player knows that distance.  Speed scales directly with it,
    # so this, not detection noise, usually dominates the error bar.
    distance_uncertainty_ft: float = 1.0
    # Lateral offset of the shooter from the centre of the net, in feet
    # (negative = shooter's left).  Used to correct the 3-D flight distance.
    shooter_offset_ft: float = 0.0
    # Height of the puck at release (it starts on the ice).
    release_height_in: float = PUCK_THICKNESS_IN / 2.0

    # Frames before impact used by the goal-plane method.  Close to the plane,
    # the goal-plane scale is the puck's true scale.
    plane_window: int = 6

    # The goal outline is only located to about a pixel, and in a square-on
    # view the camera pose that follows from it is poorly conditioned.  These
    # are the resulting speed errors, as fractions, folded into the error bar.
    pose_uncertainty_frac: float = 0.03
    assumed_lens_uncertainty_frac: float = 0.09

    # Used only to warn, before any analysis, whether the capture rate can
    # resolve a shot at all: flight time = distance / speed.
    nominal_speed_mph: float = 45.0

    # Sanity band.  Outside it we keep the number but flag it.
    plausible_min_mph: float = 10.0
    plausible_max_mph: float = 120.0


@dataclass
class ShotConfig:
    """Turning a trajectory into a scored shot."""

    # A mapped impact within this distance of the pipe centreline is a post.
    post_tolerance_in: float = POST_DIAMETER_IN
    # How far outside the mouth we still attribute the shot to this net.
    miss_margin_in: float = 48.0
    # Below the ice is not a place a puck can arrive.  A small allowance covers
    # measurement error; anything further down is a fragment of a trajectory
    # caught mid-flight, well in front of the goal plane, not an impact.
    below_ice_margin_in: float = 8.0
    # Sub-frame extrapolation past the last detection, in frames.
    impact_extrapolation_frames: float = 0.5
    min_shot_separation_frames: int = 8


@dataclass
class TargetConfig:
    """What the player is aiming at, if anything.

    ``kind`` is a named target (top_left, top_right, bottom_left,
    bottom_right, five_hole), a group scored against whichever member each
    shot was nearest (top_shelf, any_corner, low_corners), or "custom" with an
    aim point in goal inches.
    """

    kind: str | None = None
    # Hit radius around the aim point.  Six inches is roughly the size of the
    # hanging target discs sold for backyard nets.
    radius_in: float = 6.0
    # How far in from the pipe (and the ice) the named targets sit.
    inset_in: float = 6.0
    x_in: float | None = None
    y_in: float | None = None


@dataclass
class Config:
    goal: GoalSpec = field(default_factory=GoalSpec)
    camera: CameraConfig = field(default_factory=CameraConfig)
    net: NetDetectConfig = field(default_factory=NetDetectConfig)
    puck: PuckDetectConfig = field(default_factory=PuckDetectConfig)
    track: TrackingConfig = field(default_factory=TrackingConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    shot: ShotConfig = field(default_factory=ShotConfig)
    target: TargetConfig = field(default_factory=TargetConfig)

    # Override the container's frame rate.  Phone slow-motion clips frequently
    # lie about this, and every speed scales linearly with it.
    fps_override: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
