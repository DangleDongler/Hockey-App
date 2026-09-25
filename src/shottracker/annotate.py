"""A copy of a clip with every shot drawn on it, to keep or to share.

The same picture the web player draws over the video -- the net's outline,
each shot's path from where it was first seen to where it hit, a mark there
with its number, result and speed, the zone it hit lighting up, older shots
fading back -- burned into the frames and written as H.264, which every
phone plays.  It works from a clip's result as the web player gets it, so a
session reloaded from disk can be drawn without being analysed again.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import cv2
import numpy as np

from .video import iter_frames

# BGR, matching the web player's colours.
OUTCOME_BGR = {"on_net": (94, 197, 34), "post": (11, 158, 245), "miss": (68, 68, 239)}
NET_BGR = (248, 189, 56)
DOT_BGR = (68, 68, 239)
INK = (32, 18, 11)
FLASH_S = 0.6       # the zone it hit lights up this long
CALLOUT_S = 1.5     # and the newest shot's label is shown large this long
OLD_ALPHA = 0.4     # shots before the newest are drawn this faint

ZONE_CELL = {
    "top_left": (0, 0), "top_mid": (0, 1), "top_right": (0, 2),
    "mid_left": (1, 0), "mid_mid": (1, 1), "mid_right": (1, 2),
    "low_left": (2, 0), "five_hole": (2, 1), "low_right": (2, 2),
}
FONT = cv2.FONT_HERSHEY_DUPLEX


def _lerp(a, b, f):
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


def _zone_cell(quad, key: str | None):
    if key not in ZONE_CELL:
        return None
    row, col = ZONE_CELL[key]
    tl, tr, br, bl = quad

    def at(fx, fy):
        return _lerp(_lerp(tl, tr, fx), _lerp(bl, br, fx), fy)

    return np.array([at(col / 3, row / 3), at((col + 1) / 3, row / 3),
                     at((col + 1) / 3, (row + 1) / 3), at(col / 3, (row + 1) / 3)])


def _pts(points, k: float) -> np.ndarray:
    return np.round(np.asarray(points, dtype=np.float64) * k * 16).astype(np.int32)


def _polyline(img, points, k, color, width, closed=False):
    if len(points) >= 2:
        cv2.polylines(img, [_pts(points, k)], closed, color, max(1, int(round(width))), cv2.LINE_AA, shift=4)


def _dot(img, xy, k, radius, fill, edge=None, edge_w=0):
    c = tuple(int(v) for v in _pts([xy], k)[0])
    cv2.circle(img, c, int(round(radius * 16)), fill, -1, cv2.LINE_AA, shift=4)
    if edge is not None:
        cv2.circle(img, c, int(round(radius * 16)), edge, max(1, int(round(edge_w))), cv2.LINE_AA, shift=4)


def _pill(img, parts, x, y, height, fill, ink, align="left"):
    """A rounded label of text and drawn ticks/crosses, kept inside the picture.

    ``parts`` is a list of strings and the markers "tick" / "cross", which the
    built-in fonts cannot draw.
    """
    scale = height / 30.0
    thick = max(1, int(round(height / 14)))
    gap = height * 0.25
    widths = []
    for p in parts:
        if p in ("tick", "cross"):
            widths.append(height * 0.55)
        else:
            widths.append(cv2.getTextSize(p, FONT, scale, thick)[0][0])
    w = sum(widths) + gap * (len(parts) - 1) + height * 0.9
    h = height * 1.5
    x0 = x - w / 2 if align == "center" else x - w if align == "right" else x
    x0 = min(max(2, x0), img.shape[1] - w - 2)
    y = min(max(h / 2 + 2, y), img.shape[0] - h / 2 - 2)
    r = h / 2
    p1, p2 = (int(x0 + r), int(y - r)), (int(x0 + w - r), int(y + r))
    cv2.rectangle(img, p1, p2, fill, -1, cv2.LINE_AA)
    cv2.circle(img, (int(x0 + r), int(y)), int(r), fill, -1, cv2.LINE_AA)
    cv2.circle(img, (int(x0 + w - r), int(y)), int(r), fill, -1, cv2.LINE_AA)
    cx = x0 + height * 0.45
    base = y + height * 0.35
    for p, pw in zip(parts, widths):
        if p == "tick":
            a = (cx, y + height * 0.02)
            b = (cx + pw * 0.38, y + height * 0.3)
            c = (cx + pw, y - height * 0.35)
            cv2.polylines(img, [np.round(np.array([a, b, c]) * 16).astype(np.int32)], False, ink,
                          thick + 1, cv2.LINE_AA, shift=4)
        elif p == "cross":
            s = pw * 0.8
            for a, b in (((cx, y - s / 2), (cx + s, y + s / 2)), ((cx, y + s / 2), (cx + s, y - s / 2))):
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), ink, thick + 1, cv2.LINE_AA)
        else:
            cv2.putText(img, p, (int(cx), int(base)), FONT, scale, ink, thick, cv2.LINE_AA)
        cx += pw + gap


def _label_parts(number: int, outcome: str, mph: float | None, full: bool) -> list[str]:
    parts = [f"#{number}"]
    if not full:
        return parts
    parts.append({"on_net": "tick", "miss": "cross"}.get(outcome, outcome))
    if mph is not None:
        parts.append(f"{round(mph)} mph")
    return parts


class Painter:
    """Draws a clip's shots onto its frames, one frame at a time."""

    def __init__(self, clip: dict, k: float, fps: float, numbers: dict[int, int] | None = None):
        self.clip, self.k, self.fps = clip, k, fps
        self.u = max(1.0, k * min(clip["video"]["width"], clip["video"]["height"]) / 540.0)
        tracks = {t["shot_index"]: t for t in clip.get("tracks", [])}
        self.shots = [(s, tracks.get(i), (numbers or {}).get(s["index"], s["index"] + 1))
                      for i, s in enumerate(clip.get("shots", []))]

    def _net(self, img):
        net, goal, k, u = self.clip.get("net"), self.clip.get("goal") or {}, self.k, self.u
        if not net or not net.get("quad"):
            return
        _polyline(img, goal.get("outer_outline") or net["quad"], k, NET_BGR, 2 * u, closed=True)
        mouth = goal.get("mouth_outline") or goal.get("mouth_quad")
        if mouth:
            _polyline(img, mouth, k, NET_BGR, 1 * u, closed=True)
        for xy in net["quad"]:
            _dot(img, xy, k, 3.5 * u, DOT_BGR)
        tl, tr = net["quad"][0], net["quad"][1]
        _pill(img, ["NET"], (tl[0] + tr[0]) / 2 * k, min(tl[1], tr[1]) * k - 12 * u, 11 * u, DOT_BGR,
              (255, 255, 255), "center")

    def draw(self, img: np.ndarray, frame: int) -> np.ndarray:
        self._net(img)
        landed = [s for s, _, _ in self.shots if frame >= s["impact_frame"]]
        latest = max(landed, key=lambda s: s["impact_frame"]) if landed else None
        showing = [(s, t, n) for s, t, n in self.shots if frame >= s.get("first_tracked_frame", s["impact_frame"])]
        old = [x for x in showing if x[0] is not latest and frame >= x[0]["impact_frame"]]
        new = [x for x in showing if x not in old]
        # The shots before the newest, drawn on a copy and blended in faint;
        # then the zone just hit lights up; then the newest, and any puck
        # still in the air, on top at full strength.
        if old:
            faded = img.copy()
            for shot, track, number in old:
                self._shot(faded, frame, shot, track, number, newest=False)
            img = cv2.addWeighted(img, 1 - OLD_ALPHA, faded, OLD_ALPHA, 0)
        if latest is not None:
            age = (frame - latest["impact_frame"]) / self.fps
            quad = (self.clip.get("goal") or {}).get("mouth_quad")
            cell = _zone_cell(quad, latest.get("zone_key")) if quad and age < FLASH_S else None
            if cell is not None:
                a = 0.45 * (1 - age / FLASH_S)
                tint = img.copy()
                cv2.fillPoly(tint, [_pts(cell, self.k)], self._color(latest), cv2.LINE_AA, shift=4)
                img = cv2.addWeighted(img, 1 - a, tint, a, 0)
        for shot, track, number in new:
            self._shot(img, frame, shot, track, number, newest=shot is latest)
        return img

    @staticmethod
    def _color(shot: dict) -> tuple[int, int, int]:
        return OUTCOME_BGR.get(shot.get("outcome"), (184, 163, 148))

    def _shot(self, img, frame: int, shot: dict, track: dict | None, number: int, newest: bool) -> None:
        k, u = self.k, self.u
        done = frame >= shot["impact_frame"]
        color = self._color(shot) if done else (255, 255, 255)
        pts = [p for p, f in zip(track["points"], track["frames"]) if f <= frame] if track else []
        if done:
            pts.append(shot["impact_image"])
        _polyline(img, pts, k, (0, 0, 0), 5 * u)
        _polyline(img, pts, k, color, 2.5 * u)
        if not done:
            if pts:
                c = tuple(int(v) for v in _pts([pts[-1]], k)[0])
                cv2.circle(img, c, int(round(7 * u * 16)), (255, 255, 255), max(1, int(round(2 * u))),
                           cv2.LINE_AA, shift=4)
            return
        age = (frame - shot["impact_frame"]) / self.fps
        x, y = shot["impact_image"]
        _dot(img, (x, y), k, 6 * u, color, INK, 2 * u)
        mph = (shot.get("speed") or {}).get("mph")
        size = (20 if newest and age < CALLOUT_S else 13 if newest else 10) * u
        parts = _label_parts(number, shot.get("outcome", ""), mph, newest)
        right = x * k + 10 * u + size * 6 > img.shape[1]
        _pill(img, parts, x * k + (-10 if right else 10) * u, y * k - 14 * u, size, color, INK,
              "right" if right else "left")


def render(video_path: str, clip: dict, out_path: str, numbers: dict[int, int] | None = None,
           max_width: int = 1080, progress: Callable[[float], None] | None = None) -> str:
    """Write ``out_path``: the clip at up to ``max_width`` across, with its shots drawn in."""
    import av

    v = clip["video"]
    width, height = int(v["width"]), int(v["height"])
    k = min(1.0, max_width / float(width))
    size = (int(round(width * k)) // 2 * 2, int(round(height * k)) // 2 * 2)
    k = size[0] / float(width)
    rate = float(v.get("playback_fps") or v.get("fps") or 30.0)
    painter = Painter(clip, k, rate, numbers)
    total = max(int(v.get("frame_count") or 0), 1)

    out = av.open(out_path, "w")
    try:
        stream = out.add_stream("libx264", rate=Fraction(rate).limit_denominator(1001))
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        stream.options = {"crf": "20", "preset": "veryfast", "movflags": "+faststart"}
        for i, frame in enumerate(iter_frames(video_path, size)):
            drawn = painter.draw(frame, i)
            for packet in stream.encode(av.VideoFrame.from_ndarray(drawn, format="bgr24")):
                out.mux(packet)
            if progress and i % 30 == 0:
                progress(min(1.0, i / total))
        for packet in stream.encode():
            out.mux(packet)
    finally:
        out.close()
    return out_path
