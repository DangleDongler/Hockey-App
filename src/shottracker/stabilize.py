"""Compensating for a camera that was held rather than propped.

The puck detector works by comparing each frame against a plate of the scene
with no puck in it.  That only holds if the scene sits still in the frame.
Hand-held footage drifts — a phone held by a parent wanders tens of pixels over
a minute — and then *everything* differs from the plate, the frame fills with
false candidates, and the real puck is one blob among hundreds.

Hand-held drift is overwhelmingly translation: people sway and re-aim, they do
not roll the camera. So a per-frame (dx, dy), recovered by phase correlation,
is enough to put every frame back on the same footing. Anything the model
cannot explain — a genuine pan, a zoom, someone walking the camera across the
yard — shows up as a weak correlation response and is reported rather than
silently mis-corrected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CameraMotion:
    """Per-frame translation that puts each frame back onto a reference view."""

    shifts: dict[int, tuple[float, float]] = field(default_factory=dict)
    response: dict[int, float] = field(default_factory=dict)
    reference_frame: int = 0
    needed: bool = False          # whether the drift was large enough to matter
    notes: list[str] = field(default_factory=list)

    def shift(self, frame: int) -> tuple[float, float]:
        return self.shifts.get(frame, (0.0, 0.0))

    def compensate(self, img: np.ndarray, frame: int) -> np.ndarray:
        """Warp a frame back onto the reference view."""
        dx, dy = self.shift(frame)
        if dx == 0.0 and dy == 0.0:
            return img
        M = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float64)
        return cv2.warpAffine(
            img, M, (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

    def to_reference(self, pt, frame: int) -> tuple[float, float]:
        """Map a point seen in ``frame`` into reference coordinates."""
        dx, dy = self.shift(frame)
        return (pt[0] - dx, pt[1] - dy)

    def to_frame(self, pt, frame: int) -> tuple[float, float]:
        """Map a reference-coordinate point into where it appears in ``frame``."""
        dx, dy = self.shift(frame)
        return (pt[0] + dx, pt[1] + dy)

    @property
    def max_shift_px(self) -> float:
        if not self.shifts:
            return 0.0
        return float(max(abs(dx) + abs(dy) for dx, dy in self.shifts.values()))


def _prepare(gray: np.ndarray) -> np.ndarray:
    return np.float32(gray)


def estimate_motion(
    reference_gray: np.ndarray,
    frames: dict[int, np.ndarray],
    *,
    scale: float = 1.0,
    min_response: float = 0.12,
) -> dict[int, tuple[tuple[float, float], float]]:
    """Translation of each frame relative to a reference, in full-res pixels.

    ``frames`` maps frame index to a grayscale image at the same size as
    ``reference_gray``; ``scale`` converts those pixels back to full resolution.
    """
    ref = _prepare(reference_gray)
    window = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    out: dict[int, tuple[tuple[float, float], float]] = {}
    for idx, gray in frames.items():
        if gray.shape != reference_gray.shape:
            continue
        (dx, dy), resp = cv2.phaseCorrelate(ref, _prepare(gray), window)
        if resp < min_response:
            out[idx] = ((0.0, 0.0), float(resp))
            continue
        out[idx] = ((float(dx) / scale, float(dy) / scale), float(resp))
    return out


def measure(
    path: str,
    frame_indices: list[int],
    *,
    work_width: int = 320,
    drift_threshold_px: float = 3.0,
    min_response: float = 0.12,
) -> CameraMotion:
    """Sample a clip and decide whether motion compensation is needed."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return CameraMotion()

    grays: dict[int, np.ndarray] = {}
    scale = 1.0
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            scale = min(1.0, work_width / float(frame.shape[1]))
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            grays[int(idx)] = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    finally:
        cap.release()

    if len(grays) < 2:
        return CameraMotion()

    # Use a middle frame as the reference so drift is split either side of it.
    keys = sorted(grays)
    ref_idx = keys[len(keys) // 2]
    est = estimate_motion(grays[ref_idx], grays, scale=scale, min_response=min_response)

    motion = CameraMotion(reference_frame=ref_idx)
    weak = 0
    for idx, ((dx, dy), resp) in est.items():
        motion.shifts[idx] = (dx, dy)
        motion.response[idx] = resp
        if resp < min_response:
            weak += 1

    motion.needed = motion.max_shift_px > drift_threshold_px
    if motion.needed:
        motion.notes.append(
            f"the camera drifted up to {motion.max_shift_px:.0f} px during the clip, so frames were "
            "aligned before looking for the puck. Propping the phone against something steady makes "
            "this unnecessary and works better."
        )
    if weak > len(est) * 0.25:
        motion.notes.append(
            f"{weak} of {len(est)} sampled frames could not be aligned to the rest of the clip. "
            "The camera was probably panned or carried, which this correction cannot undo; "
            "detections will be unreliable."
        )
    return motion


def interpolate(motion: CameraMotion, frame_count: int) -> CameraMotion:
    """Fill in every frame's shift by interpolating between sampled ones."""
    if not motion.shifts:
        return motion
    keys = np.array(sorted(motion.shifts))
    dxs = np.array([motion.shifts[k][0] for k in keys])
    dys = np.array([motion.shifts[k][1] for k in keys])
    dense = CameraMotion(
        reference_frame=motion.reference_frame,
        needed=motion.needed,
        notes=list(motion.notes),
        response=dict(motion.response),
    )
    all_frames = np.arange(max(frame_count, 1))
    dense.shifts = {
        int(f): (float(x), float(y))
        for f, x, y in zip(all_frames, np.interp(all_frames, keys, dxs), np.interp(all_frames, keys, dys))
    }
    return dense
