"""Reading a clip's frames, fast.

OpenCV decodes on one core and converts every frame to full-size colour before
anything can shrink it.  For 4K iPhone video that is 7 frames a second, so a
two-minute clip at 60 fps took over twenty minutes just to read.  PyAV (the
same FFmpeg underneath) decodes on every core and scales during the colour
conversion: 42 frames a second on the same clip, at the size the tracker
works at.  OpenCV stays as the fallback when PyAV is not installed, or cannot
open a file.

Phones store upright video sideways with a flag saying to turn it; OpenCV
applies the flag, so PyAV's frames are turned the same way here, and every
coordinate in the app means the same thing whichever reader produced it.

HDR video (an iPhone films it by default: 10-bit, HLG) is the other catch.
OpenCV turns it into ordinary colour the way the phone shows it, which is
what every threshold in the detector was tuned on -- and that conversion is
where its time goes, 130 ms a 4K frame on one core.  PyAV's plain conversion
is fast but comes out 27 grey levels darker on average, and on a real 4K clip
that lost the shot entirely.  So for HDR one frame is read both ways and the
difference learned as a per-channel tone curve, applied to every PyAV frame:
27 levels apart becomes 5, and the tracker sees what it was tuned on.
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

try:  # pragma: no cover - exercised whichever way the environment is set up
    import av
except ImportError:  # pragma: no cover
    av = None

# Degrees the display matrix asks for -> how to turn the decoded picture.
_TURN = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}


def _turn(frame) -> int:
    r = int(round(getattr(frame, "rotation", 0) or 0)) % 360
    return r if r in _TURN else 0


def _to_bgr(frame, size: tuple[int, int] | None, gray: bool = False) -> np.ndarray:
    """A decoded PyAV frame as upright BGR (or grey), at ``size`` (width, height) if given."""
    turn = _turn(frame)
    fmt = "gray" if gray else "bgr24"
    if size is None:
        img = frame.to_ndarray(format=fmt)
    else:
        w, h = size
        # Scale in the picture's stored orientation, then turn it upright.
        sw, sh = (h, w) if turn in (90, 270) else (w, h)
        img = frame.to_ndarray(format=fmt, width=sw, height=sh, interpolation="AREA")
    return cv2.rotate(img, _TURN[turn]) if turn else img


def _curve(mine: np.ndarray, ref: np.ndarray) -> np.ndarray | None:
    """A per-channel look-up table taking ``mine`` to ``ref`` (same shape)."""
    chans = 1 if mine.ndim == 2 else mine.shape[2]
    lut = np.zeros((256, 1, chans), np.uint8) if chans > 1 else np.zeros(256, np.uint8)
    for ch in range(chans):
        x = (mine if chans == 1 else mine[..., ch]).ravel()
        y = (ref if chans == 1 else ref[..., ch]).ravel()
        count = np.bincount(x, minlength=256)
        mean = np.bincount(x, weights=y, minlength=256) / np.maximum(count, 1)
        seen = np.nonzero(count > 20)[0]
        if len(seen) < 2:
            return None
        vals = np.clip(np.round(np.interp(np.arange(256), seen, mean[seen])), 0, 255)
        if chans == 1:
            lut[:] = vals
        else:
            lut[:, 0, ch] = vals
    return lut


def _is_hdr(stream) -> bool:
    ctx = stream.codec_context
    fmt = getattr(ctx, "pix_fmt", "") or ""
    return getattr(ctx, "color_trc", None) in (16, 18) or "10" in fmt or "12" in fmt


def _tone_curve(path: str, container, stream, size: tuple[int, int] | None = None,
                gray: bool = False) -> np.ndarray | None:
    """Look-up table taking PyAV's plain frames to OpenCV's, from the first frame.

    For grey frames it targets OpenCV's frame turned grey at the same size,
    which is exactly what the rest of the app would have made of it.
    """
    cap = cv2.VideoCapture(path)
    try:
        ok, ref = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    first = next(_decode(container, stream), None)
    if first is None:
        return None
    if size is not None:
        ref = cv2.resize(ref, tuple(size), interpolation=cv2.INTER_AREA)
    if gray:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    mine = _to_bgr(first, size, gray)
    if mine.shape != ref.shape:
        return None
    return _curve(mine, ref)


def _open_av(path: str):
    if av is None:
        return None
    try:
        container = av.open(path)
    except Exception:  # noqa: BLE001 - any failure means: use OpenCV instead
        return None
    if not container.streams.video:
        container.close()
        return None
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    return container, stream


def _decode(container, stream) -> Iterator:
    """Every frame, skipping packets the decoder rejects rather than stopping."""
    for packet in container.demux(stream):
        try:
            yield from packet.decode()
        except av.error.InvalidDataError:  # pragma: no cover - damaged file
            continue


def iter_frames(path: str, size: tuple[int, int] | None = None, gray: bool = False) -> Iterator[np.ndarray]:
    """Every frame in order, upright, at ``size`` (width, height) or full size.

    ``gray`` gives single-channel frames, skipping the colour conversion
    altogether -- the puck detector only ever looks at brightness.
    """
    opened = _open_av(path)
    if opened is not None:
        container, stream = opened
        try:
            # Grey always goes through the curve: PyAV's grey is the video's
            # own luma, a level or so off OpenCV's, and the detector's
            # thresholds were set on OpenCV's.
            lut = None
            if gray or _is_hdr(stream):
                lut = _tone_curve(path, container, stream, size if gray else None, gray)
                container.seek(0, stream=stream)
            for frame in _decode(container, stream):
                img = _to_bgr(frame, size, gray)
                yield cv2.LUT(img, lut) if lut is not None else img
        finally:
            container.close()
        return
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
                img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
            yield cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else img
    finally:
        cap.release()


def keyframes(path: str, n: int, size: tuple[int, int] | None = None,
              gray: bool = False) -> list[np.ndarray] | None:
    """``n`` evenly spread keyframes, decoded alone, or None if there are fewer.

    Keyframes come about once a second in phone video and can be decoded
    without the frames between them, so a long clip is sampled in a few
    seconds instead of minutes of seeking.
    """
    opened = _open_av(path)
    if opened is None:
        return None
    container, stream = opened
    try:
        count = sum(1 for p in container.demux(stream) if p.size and p.is_keyframe)
        if count < n:
            return None
        container.seek(0, stream=stream)
        lut = None
        if gray or _is_hdr(stream):
            lut = _tone_curve(path, container, stream, size if gray else None, gray)
            container.seek(0, stream=stream)
        stream.codec_context.skip_frame = "NONKEY"
        want = set(np.linspace(0, count - 1, n).round().astype(int).tolist())
        out: list[np.ndarray] = []
        for i, frame in enumerate(_decode(container, stream)):
            if i in want:
                img = _to_bgr(frame, size, gray)
                out.append(cv2.LUT(img, lut) if lut is not None else img)
            if len(out) == len(want):
                break
        return out if len(out) >= n // 2 else None
    finally:
        container.close()
