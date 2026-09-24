"""What a phone's video file says about its own timing.

OpenCV reports the rate a file *plays* at.  For ordinary video that is the rate
it was filmed at, but a phone's slow motion is often handed over with the slow
motion baked in: a second filmed at 240 fps becomes eight seconds of ordinary
30 fps video.  Every frame is then 1/240 s apart in real life while the file
says 1/30, and every speed would come out eight times too slow.

The sound gives it away.  It is not slowed along with the picture, so a baked
slow-motion clip carries eight seconds of video and about one second of sound.
The ratio, times the playback rate, is the rate it was filmed at.

This reads the MP4/QuickTime box structure directly -- the video and sound
tracks' durations and the video's frame count -- because OpenCV exposes
neither the sound track nor per-track durations.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

# Phones film slow motion at one of these rates.  An implied rate that lands
# near none of them is not trusted.
STANDARD_CAPTURE_FPS = (60.0, 120.0, 240.0, 480.0, 960.0)

# Sound shorter than the picture by less than this is an ordinary clip whose
# audio simply stops early or starts late.
MIN_SLOWDOWN = 1.8
# How close the implied rate must come to a standard one.  A ramp of normal
# speed at the start or end of a slow-motion clip drags the implied rate down
# by more than this, and then the clip is flagged rather than guessed at.
SNAP_TOLERANCE = 0.08
# Anything shorter carries too little sound to time the clip by.
MIN_AUDIO_S = 0.2
# The movie box of a phone clip is a few megabytes at most.
MAX_MOOV_BYTES = 64 * 1024 * 1024

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


@dataclass
class TrackTiming:
    handler: str            # "vide", "soun", "meta", ...
    duration_s: float
    frames: int | None = None


@dataclass
class ContainerTiming:
    tracks: list[TrackTiming] = field(default_factory=list)

    def first(self, handler: str) -> TrackTiming | None:
        return next((t for t in self.tracks if t.handler == handler), None)

    @property
    def video(self) -> TrackTiming | None:
        return self.first("vide")

    @property
    def audio(self) -> TrackTiming | None:
        return self.first("soun")


@dataclass
class SlowMotion:
    """The outcome of checking a clip for baked-in slow motion."""

    capture_fps: float | None = None   # set when the clip is confidently slow motion
    slowdown: float | None = None      # video length / sound length
    note: str | None = None            # explanation for the report, either way


def _boxes(buf: bytes, start: int, end: int):
    off = start
    while off + 8 <= end:
        size, kind = struct.unpack(">I4s", buf[off:off + 8])
        header = 8
        if size == 1:
            if off + 16 > end:
                return
            size = struct.unpack(">Q", buf[off + 8:off + 16])[0]
            header = 16
        elif size == 0:
            size = end - off
        if size < header or off + size > end:
            return
        yield kind, off + header, off + size
        off += size


def _child(buf: bytes, box: tuple[bytes, int, int] | None, kind: bytes):
    if box is None:
        return None
    return next((b for b in _boxes(buf, box[1], box[2]) if b[0] == kind), None)


def _read_moov(path: str) -> bytes | None:
    """Find the movie box, which phones put after the media data."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        file_size = fh.tell()
        off = 0
        while off + 8 <= file_size:
            fh.seek(off)
            head = fh.read(16)
            if len(head) < 8:
                return None
            size, kind = struct.unpack(">I4s", head[:8])
            header = 8
            if size == 1:
                if len(head) < 16:
                    return None
                size = struct.unpack(">Q", head[8:16])[0]
                header = 16
            elif size == 0:
                size = file_size - off
            if size < header:
                return None
            if kind == b"moov":
                if size > MAX_MOOV_BYTES:
                    return None
                fh.seek(off + header)
                return fh.read(size - header)
            off += size
    return None


def _track_timing(buf: bytes, trak: tuple[bytes, int, int]) -> TrackTiming | None:
    mdia = _child(buf, trak, b"mdia")
    hdlr = _child(buf, mdia, b"hdlr")
    mdhd = _child(buf, mdia, b"mdhd")
    if hdlr is None or mdhd is None:
        return None
    handler = buf[hdlr[1] + 8:hdlr[1] + 12].decode("latin-1")

    base = mdhd[1]
    if buf[base] == 1:
        timescale, duration = struct.unpack(">IQ", buf[base + 20:base + 32])
    else:
        timescale, duration = struct.unpack(">II", buf[base + 12:base + 20])
    if not timescale:
        return None

    frames = None
    stts = _child(buf, _child(buf, _child(buf, mdia, b"minf"), b"stbl"), b"stts")
    if stts is not None:
        n = struct.unpack(">I", buf[stts[1] + 4:stts[1] + 8])[0]
        entries = buf[stts[1] + 8:stts[1] + 8 + 8 * n]
        frames = sum(struct.unpack(">I", entries[i:i + 4])[0] for i in range(0, len(entries) - 7, 8))
    return TrackTiming(handler=handler, duration_s=duration / timescale, frames=frames)


def read_timing(path: str) -> ContainerTiming | None:
    """Per-track durations of an MP4/QuickTime file, or None for anything else."""
    try:
        moov = _read_moov(path)
    except OSError:
        return None
    if not moov:
        return None
    timing = ContainerTiming()
    try:
        for kind, start, end in _boxes(moov, 0, len(moov)):
            if kind != b"trak":
                continue
            t = _track_timing(moov, (kind, start, end))
            if t is not None:
                timing.tracks.append(t)
    except (struct.error, IndexError):
        return None
    return timing


def detect_slow_motion(timing: ContainerTiming | None, playback_fps: float) -> SlowMotion:
    """Decide whether a clip is slow motion played back at ``playback_fps``."""
    if timing is None or playback_fps <= 0:
        return SlowMotion()
    video, audio = timing.video, timing.audio
    if video is None or audio is None or audio.duration_s < MIN_AUDIO_S:
        return SlowMotion()

    slowdown = video.duration_s / audio.duration_s
    if slowdown < MIN_SLOWDOWN:
        return SlowMotion(slowdown=slowdown)

    implied = playback_fps * slowdown
    nearest = min(STANDARD_CAPTURE_FPS, key=lambda r: abs(implied / r - 1.0))
    if abs(implied / nearest - 1.0) <= SNAP_TOLERANCE:
        return SlowMotion(
            capture_fps=nearest,
            slowdown=slowdown,
            note=(
                f"this is slow motion played back at {playback_fps:.0f} fps: {video.duration_s:.1f} s of "
                f"video but only {audio.duration_s:.1f} s of sound, so it was filmed at {nearest:.0f} fps. "
                f"Timing uses {nearest:.0f} fps; if that is wrong, set the frame rate by hand."
            ),
        )
    return SlowMotion(
        slowdown=slowdown,
        note=(
            f"this looks like slow motion ({video.duration_s:.1f} s of video but only {audio.duration_s:.1f} s "
            f"of sound), but the numbers don't match a standard filming rate -- usually because part of the "
            f"clip plays at normal speed. Timing assumes {playback_fps:.0f} fps, so speeds may be far too low; "
            "trim the clip to the slow part, or set the frame rate by hand."
        ),
    )
