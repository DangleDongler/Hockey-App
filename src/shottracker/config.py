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
# corners are rounded.  This is the inside of the bend, measured at 5.8" and
# 6.4" on the two top corners of a Bauer steel goal from a square-on photo
# flattened onto the goal plane.  It varies between makes, and it is only used
# for drawing and for calling corner hits -- the scale comes from where the
# straight sections of pipe would meet, which the radius does not move.
CORNER_RADIUS_IN = 6.0
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
    # With the lens known, put back posts' feet hidden in grass (see
    # camera.hidden_feet).
    restore_hidden_feet: bool = True
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

    # Second method: find the goal by its *shape* -- two thin upright bars on
    # the ground -- for nets whose pipe is too dark or faded for the colour
    # method.  The colour band is deliberately loose (shaded maroon pipe sits
    # around H 150-175 and overlaps skin); thinness is what does the rejecting,
    # because legs and shrubs are not twelve times taller than they are wide.
    post_hue_low_max: int = 8
    post_hue_high_min: int = 150
    post_sat_min: int = 45
    post_val_min: int = 25
    # On the orange side of red live wood, brick and fence stain.  Real goal
    # pipe there is strongly coloured (sunlit posts measured S~145); stained
    # fence boards measured S 56-89.  So that side needs real saturation.
    post_low_hue_sat_min: int = 100
    post_min_aspect: float = 6.0          # height / width
    post_max_width_frac: float = 0.04     # of the frame width
    # Of the frame height.  A net across a backyard filmed from beside the
    # shooter stands about 4% of a portrait frame tall, and glare can eat the
    # top of a post, so the floor sits well under that; the pairing checks
    # (level feet, spacing, a crossbar between) keep out lone red uprights.
    post_min_height_frac: float = 0.025
    # Structural support below this counts as the colour method not having
    # really found pipe, and the shape method is tried instead.
    colour_method_trust: float = 0.8
    # "auto" tries colour and falls back to shape; "colour" or "posts" forces one.
    method: str = "auto"
    # Frames must agree with the consensus to within this fraction of the
    # quad's diagonal, otherwise they are dropped as outliers.
    consensus_tol_frac: float = 0.06
    min_agreeing_frames: int = 3
    # Most sampled frames must put the goal in the same place.  A minority
    # agreeing means the goal moved in the frame -- a hand-held camera -- and a
    # single outline would be wrong for the rest of the clip.
    min_agreement_frac: float = 0.5
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
    # ...or this many times the pixel's own usual spread, if that is more.
    # Textured ground near the lens and sunlit leaves shimmer; the puck, a
    # black disc crossing them, clears either threshold easily.  0 disables.
    noise_threshold_k: float = 5.0
    # Decode straight to grey at working size for the per-frame pass (see
    # video.iter_frames): faster, but PyAV's grey differs from OpenCV's by a
    # grey level or so, and at the net -- the puck over the red post -- that
    # moved a synthetic shot's last sighting 5 px and its speed 7%.  Off:
    # full colour frames, shrunk the way the detector was tuned on.
    grey_decode: bool = False
    # Frames searched at once (0: one per CPU core, up to 8).
    workers: int = 0
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
    # Clutter that stays put -- leaves flickering in the wind, rippling
    # netting, a fence in the sun -- turns up at the same spot frame after
    # frame, where a puck passes any one spot once.  Candidates seen at the
    # same place (within a puck's width) in several frames just before *and*
    # just after are ranked below every other candidate before the cap above
    # is applied.  Before and after both, so the puck landing in netting that
    # only moves once it is hit keeps its place.  On a real 240 fps clip in
    # the wind this let through all 60 puck sightings while demoting three
    # quarters of the clutter; without it the puck ranked 7th to 25th of ~90
    # candidates a frame and was capped away.
    demote_recurring: bool = True
    raw_candidates_per_frame: int = 400
    recurring_window_s: float = 0.125
    recurring_min_window_frames: int = 8
    recurring_min_hits: int = 3       # distinct frames, on each side
    # Then, of what is left, candidates in a crowd -- more than the tracker's
    # clutter_max_neighbours others within clutter_radius_frac goal widths,
    # the same test a track's clear sightings must pass -- rank after the
    # ones standing alone.  A bush in the wind is dozens of leaves that each
    # move too far to count as recurring; a puck in flight is mostly alone.
    demote_crowded: bool = True
    # A frame pinned at the cap means the foreground model is failing (usually
    # a camera that moved).  Past this fraction of frames, say so.
    saturated_frame_warn_frac: float = 0.5


@dataclass
class TrackingConfig:
    """Assembling candidates into shot trajectories."""

    max_frame_gap: int = 3         # tolerate missed detections mid-flight
    # ...but not at the ends: up to this many detections cut off from either
    # end by a gap this long (in frames) are dropped.  See _without_stragglers.
    straggler_gap_frames: int = 3
    max_straggler_run: int = 2
    straggler_min_fps: float = 50.0
    # A stray end point further off the track's line than this fraction of
    # the distance the puck would have covered in the gap is not the puck.
    # Measured: 0.01 for a puck reappearing from behind a post, 0.24-0.45 for
    # netting and bounces after the impact.
    straggler_off_line: float = 0.15
    # The end of a flight (see until_impact): a step at 60 fps spacing that
    # turns more than this, or speeds up by more than this factor, from the
    # steps before it -- and only once the puck is moving this many pixels a
    # frame, so that jitter is not taken for a turn.
    impact_turn_deg: float = 20.0
    impact_speed_jump: float = 1.4
    impact_min_step_px: float = 4.0
    impact_stop_ratio: float = 0.4
    # ...and only within this many goal widths of the goal's outline.
    impact_near_goal_frac: float = 0.35
    # Gating radius = base + velocity-scaled term, in units of the goal width.
    gate_base_frac: float = 0.03
    gate_vel_frac: float = 0.55

    min_track_length: int = 4
    # ...and seen for at least this long.  A count of detections alone means
    # 0.13 s at 30 fps but 0.017 s at 240, where leaves flickering beside the
    # goal string together into "shots".  A real shot is on camera for a
    # large part of its flight, a tenth of a second or more.
    min_track_s: float = 0.1
    # A shot crosses ground fast; slow blobs are limbs, sticks, shadows.
    min_mean_speed_goalwidths_per_sec: float = 0.45
    # A puck in flight is very nearly a straight line in the image; the only
    # real departure is the arc gravity puts on it.
    max_line_residual_frac: float = 0.04   # of the goal width

    # Flickering leaves near the sun, rippling netting: where dozens of
    # things move in every frame, four or five of them line up by chance,
    # and the result looks like a short, straight flight.  A real puck spends
    # most of its flight crossing clear background.  So a track must have
    # this much time's worth of sightings (and never fewer than the floor)
    # with no more than ``clutter_max_neighbours`` other candidates within
    # ``clutter_radius_frac`` goal widths, in that frame.  Measured on real
    # clips at 60 fps: every chance line had at most 3 such sightings, every
    # real shot at least 7.
    min_clear_sightings_s: float = 0.075
    min_clear_sightings: int = 3
    clutter_radius_frac: float = 0.5
    clutter_max_neighbours: int = 10
    # In a clip's first moments, while the camera settles, a track whose
    # sightings keep turning up again in place -- more than this fraction of
    # the nearby frames, either side -- is things flickering where they stand,
    # not a puck (see tracking.revisits).  A line of lamps and posts doing
    # that in the first 8 frames of a real clip read as an 89 mph shot: 43%;
    # real shots in flight, 0-8%.  Only then: later on, the recurring-clutter
    # test (which needs frames before as well as after) has it covered, and a
    # real shot's track can start where the puck sat on the stick -- one in a
    # real session scored 31% and was dropped.
    max_revisit_frac: float = 0.2
    revisit_check_first_s: float = 0.5

    ransac_iterations: int = 60
    # Seed pairs grow with the square of the candidates per frame, so a bad
    # foreground model can make the search explode.  This is the safety valve.
    max_seeds: int = 150_000


@dataclass
class CameraConfig:
    """What we know, or assume, about the lens.

    In order of preference: a field of view the player gives; the focal
    length the phone wrote into the file; the one the goal's own outline
    implies; and, last, a typical phone lens.  The outline only reveals the
    focal length when the camera is well off to one side -- on a real clip
    filmed from the ground 26 ft out it came out 40% short, which put the
    camera two feet up and nearly behind the shooter instead of two inches
    off the ground and seven feet to the side.
    """

    # Horizontal field of view of the frame as filmed, when known.
    hfov_deg: float | None = None
    # The lens the player says they filmed with, as a 35 mm-equivalent focal
    # length (see LENS_CHOICES).  Used when the file does not record its own:
    # slow motion never does, and a phone may strip it when uploading.
    focal_35mm: float | None = None
    use_lens_metadata: bool = True
    # The 35 mm-equivalent focal length a phone records is rounded to a whole
    # millimetre: about 4% at the ultra-wide's 13-14 mm.
    lens_metadata_spread: float = 0.04
    assumed_focal_frac: float = 0.80

    # iPhone video, 35 mm-equivalent, as the phone records it for the
    # stabilized video frame (the 0.5x lens measured 14 on an iPhone 17; the
    # others are scaled from the lenses' still-photo figures the same way).
    LENS_CHOICES = {"0.5x": 14.0, "1x": 28.0, "2x": 56.0}

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
    # A shot whose flight starts this fraction of the flight or more behind
    # the stated shooting spot suggests the distance given is short.
    release_check_margin: float = 0.1
    # ...and when the shots in a clip agree on it to within this fraction of
    # the distance given, say so.
    distance_warn_frac: float = 0.1
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
    # Slower than this over a timed flight is not a shot: on a real
    # 2.6-minute session, the shooter skating and stickhandling produced
    # tracks of 5-11 mph lasting one to three seconds.
    min_shot_mph: float = 12.0
    # Below the ice is not a place a puck can arrive.  A small allowance covers
    # measurement error; anything further down is a fragment of a trajectory
    # caught mid-flight, well in front of the goal plane, not an impact.
    below_ice_margin_in: float = 8.0
    # Sub-frame extrapolation past the last detection, in frames.
    impact_extrapolation_frames: float = 0.5
    # Place the mark where the puck reached the goal line, timed from how fast
    # it was closing on it, rather than where it was last seen.
    time_the_crossing: bool = True
    # ...but only when the camera sees the flight from at least this far off
    # its line.  From straight behind the shooter, how fast the puck closes on
    # the goal line barely shows, and the timing is worse than none.
    min_view_angle_for_timing_deg: float = 25.0
    # And a last sighting, after missing frames, that comes more than this
    # long after the crossing timed from the sightings before it was past the
    # goal line -- netting springing back, or a leaf the track picked up -- so
    # it is dropped.  On a 20-shot synthetic session from chest height that
    # took the worst mark from 25 in to 7.  Only with the lens known, and at
    # mark_at_crossing_min_fps and up: at 30 fps the timing itself is too
    # coarse to overrule a sighting.
    mark_at_crossing: bool = True
    mark_at_crossing_after_s: float = 0.03
    mark_at_crossing_min_fps: float = 50.0
    # Two impacts closer together than this are one shot plus its aftermath
    # (a rebound, or the netting still moving).  Seconds, not frames: at
    # 240 fps a count of frames is an eighth of the time it is at 30 fps.
    min_shot_separation_s: float = 0.25
    # A shot comes from somewhere.  Its first sighting has to be at least this
    # far outside the goal's outline, and it has to end nearer the goal than it
    # started, both in goal widths on screen.  Posts glinting and netting that
    # sways after an impact make tracks that start at the goal and never
    # approach it; a puck leaving a stick starts well away.  Small enough to
    # hold for a camera low and straight behind the shooter, where a whole
    # shot covers less than half a goal width.
    min_start_outside_goal: float = 0.25
    min_approach_goal: float = 0.15


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
