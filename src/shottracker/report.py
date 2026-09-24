"""Session statistics and the shot chart.

The chart is the point of the whole exercise: a picture of the net with a dot
wherever a puck arrived, so a player can see at a glance that every shot they
thought was going top corner actually went centre mass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .geometry import build_zones, mouth_outline

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import SessionResult


def summarize(result: "SessionResult") -> dict[str, Any]:
    shots = result.shots
    spec = result.config.goal
    zones = build_zones(spec)

    total = len(shots)
    on_net = [s for s in shots if s.outcome == "on_net"]
    posts = [s for s in shots if s.outcome == "post"]
    misses = [s for s in shots if s.outcome == "miss"]

    speeds = [s.speed.mph for s in shots if s.speed is not None]
    zone_counts = {z.key: 0 for z in zones}
    for s in on_net:
        if s.zone_key in zone_counts:
            zone_counts[s.zone_key] += 1

    out: dict[str, Any] = {
        "shots": total,
        "on_net": len(on_net),
        "posts": len(posts),
        "misses": len(misses),
        "accuracy_pct": round(100.0 * len(on_net) / total, 1) if total else None,
        "speed_mph": {
            "max": round(max(speeds), 1) if speeds else None,
            "mean": round(float(np.mean(speeds)), 1) if speeds else None,
            "min": round(min(speeds), 1) if speeds else None,
        },
        "zone_counts": zone_counts,
        "zone_labels": {z.key: z.label for z in zones},
    }

    if shots:
        pts = np.array([s.impact_goal_in for s in shots], dtype=float)
        centre = pts.mean(axis=0)
        spread = float(np.sqrt(((pts - centre) ** 2).sum(axis=1).mean()))
        out["grouping"] = {
            "centre_in": [round(float(centre[0]), 1), round(float(centre[1]), 1)],
            "spread_in": round(spread, 1),
            "note": "spread is the RMS distance of impacts from their own centre; smaller is tighter",
        }

        # How close the player is getting to the four corners a goalie hates.
        hw, top = spec.mouth_width_in / 2.0, spec.mouth_height_in
        corners = np.array([[-hw, top], [hw, top], [-hw, 0.0], [hw, 0.0]])
        d = np.linalg.norm(pts[:, None, :] - corners[None], axis=2).min(axis=1)
        out["corner_precision"] = {
            "mean_distance_in": round(float(d.mean()), 1),
            "best_distance_in": round(float(d.min()), 1),
            "note": "distance from each impact to the nearest corner of the net",
        }

    return out


# --- shot chart -------------------------------------------------------------

_OUTCOME_COLOR = {"on_net": "#22c55e", "post": "#f59e0b", "miss": "#ef4444"}


def shot_chart_svg(result: "SessionResult", *, width_px: int = 900, show_zones: bool = True) -> str:
    """A front-on diagram of the net with every impact marked."""
    spec = result.config.goal
    zones = build_zones(spec)

    pad_in = 14.0
    w_in = spec.mouth_width_in + 2 * pad_in
    h_in = spec.mouth_height_in + 2 * pad_in
    s = width_px / w_in
    height_px = int(round(h_in * s))

    def X(x_in: float) -> float:
        return (x_in + spec.mouth_width_in / 2.0 + pad_in) * s

    def Y(y_in: float) -> float:
        return (spec.mouth_height_in + pad_in - y_in) * s

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_px} {height_px}" '
        f'width="{width_px}" height="{height_px}" font-family="system-ui, sans-serif">',
        f'<rect width="{width_px}" height="{height_px}" fill="#f8fafc"/>',
        '<path d="'
        + " ".join(
            f"{'M' if i == 0 else 'L'} {X(px):.1f} {Y(py):.1f}"
            for i, (px, py) in enumerate(mouth_outline(spec))
        )
        + ' Z" fill="#ffffff"/>',
    ]

    # Mesh.
    for gx in np.arange(-spec.mouth_width_in / 2, spec.mouth_width_in / 2 + 0.1, 6.0):
        parts.append(
            f'<line x1="{X(gx):.1f}" y1="{Y(0):.1f}" x2="{X(gx):.1f}" y2="{Y(spec.mouth_height_in):.1f}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
    for gy in np.arange(0, spec.mouth_height_in + 0.1, 6.0):
        parts.append(
            f'<line x1="{X(-spec.mouth_width_in/2):.1f}" y1="{Y(gy):.1f}" '
            f'x2="{X(spec.mouth_width_in/2):.1f}" y2="{Y(gy):.1f}" stroke="#e2e8f0" stroke-width="1"/>'
        )

    if show_zones:
        counts = summarize(result)["zone_counts"]
        for z in zones:
            cx, cy = z.center
            n = counts.get(z.key, 0)
            parts.append(
                f'<rect x="{X(z.x0):.1f}" y="{Y(z.y1):.1f}" width="{(z.x1-z.x0)*s:.1f}" '
                f'height="{(z.y1-z.y0)*s:.1f}" fill="{"#dcfce7" if n else "none"}" '
                f'fill-opacity="{min(0.25 + 0.15*n, 0.8):.2f}" stroke="#cbd5e1" stroke-width="1" '
                f'stroke-dasharray="4 4"/>'
            )
            if n:
                parts.append(
                    f'<text x="{X(cx):.1f}" y="{Y(cy)+5:.1f}" text-anchor="middle" '
                    f'font-size="{max(11, 0.018*width_px):.0f}" fill="#16a34a" font-weight="600">{n}</text>'
                )

    # The pipe, following the bend where the posts meet the crossbar.
    pipe = spec.post_diameter_in * s
    path = " ".join(
        f"{'M' if i == 0 else 'L'} {X(px):.1f} {Y(py):.1f}"
        for i, (px, py) in enumerate(mouth_outline(spec))
    )
    parts.append(
        f'<path d="{path}" fill="none" stroke="#dc2626" stroke-width="{pipe:.1f}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )

    for shot in result.shots:
        x, y = shot.impact_goal_in
        color = _OUTCOME_COLOR.get(shot.outcome, "#64748b")
        r = max(5.0, 0.008 * width_px)
        parts.append(
            f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{r:.1f}" fill="{color}" '
            f'fill-opacity="0.85" stroke="#0f172a" stroke-width="1.2"/>'
        )
        label = f"{shot.index + 1}"
        if shot.speed:
            label += f" · {shot.speed.mph:.0f}"
        parts.append(
            f'<text x="{X(x):.1f}" y="{Y(y)-r-4:.1f}" text-anchor="middle" font-size="'
            f'{max(10, 0.014*width_px):.0f}" fill="#0f172a">{label}</text>'
        )

    targeting = result.targeting()
    if targeting:
        for t in targeting["targets"]:
            parts.append(
                f'<circle cx="{X(t["x_in"]):.1f}" cy="{Y(t["y_in"]):.1f}" r="{t["radius_in"]*s:.1f}" '
                f'fill="#3b82f6" fill-opacity="0.08" stroke="#2563eb" stroke-width="1.5" '
                f'stroke-dasharray="5 4"/>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def format_text_report(result: "SessionResult") -> str:
    """Human-readable summary for the terminal."""
    s = summarize(result)
    lines: list[str] = []
    v = result.video
    rate = (f"{v.fps:.0f} fps (slow motion, plays at {v.playback_fps:.0f})" if v.playback_fps
            else f"{v.fps:.1f} fps ({v.fps_source})")
    lines.append(f"Clip      {v.width}x{v.height} @ {rate}, {v.frame_count} frames")
    if result.net:
        lines.append(f"Net       found by {result.net.method}, confidence {result.net.confidence:.0%} "
                     f"({result.net.frames_used}/{result.net.frames_tried} frames agreed)")
    else:
        lines.append("Net       NOT FOUND")
    if result.camera:
        p = result.camera.position
        lines.append(f"Camera    {p[2]/12:.1f} ft out, {p[0]/12:+.1f} ft across, {p[1]/12:.1f} ft up "
                     f"(reprojection {result.camera.residual_px:.2f} px)")

    lines.append("")
    lines.append(f"Shots     {s['shots']}   on net {s['on_net']}   posts {s['posts']}   missed {s['misses']}")
    if s["accuracy_pct"] is not None:
        lines.append(f"Accuracy  {s['accuracy_pct']:.0f}% on net")
    if s["speed_mph"]["max"] is not None:
        lines.append(f"Speed     max {s['speed_mph']['max']:.0f} mph, average {s['speed_mph']['mean']:.0f} mph")
    if "grouping" in s:
        g = s["grouping"]
        lines.append(f"Grouping  centred {g['centre_in'][0]:+.0f}\", {g['centre_in'][1]:.0f}\" up, "
                     f"spread {g['spread_in']:.0f}\"")
    if "corner_precision" in s:
        c = s["corner_precision"]
        lines.append(f"Corners   nearest-corner distance: best {c['best_distance_in']:.0f}\", "
                     f"average {c['mean_distance_in']:.0f}\"")

    targeting = result.targeting()
    if targeting:
        ts = targeting["summary"]
        lines.append(
            f"Target    {targeting['label']} ({targeting['radius_in']:.0f}\" radius): "
            f"{ts['hits']} of {ts['shots']} on target"
            + (f", {ts['mean_distance_in']:.0f}\" off on average" if ts.get("mean_distance_in") is not None else "")
        )
        if ts.get("bias", {}).get("description"):
            lines.append(f"          {ts['bias']['description']}")

    if result.shots:
        lines.append("")
        lines.append("  #   frame    speed          where")
        for shot in result.shots:
            sp = "-"
            if shot.speed:
                sp = f"{shot.speed.mph:5.1f} mph"
                if shot.speed.uncertainty_mph:
                    sp += f" ±{shot.speed.uncertainty_mph:.1f}"
            where = shot.zone_label or shot.miss_detail or shot.outcome
            if shot.outcome == "post":
                where = f"POST ({shot.miss_detail})"
            elif shot.outcome == "miss":
                where = f"missed - {shot.miss_detail}"
            if targeting and targeting["per_shot"][shot.index]:
                vt = targeting["per_shot"][shot.index]
                where += "   [on target]" if vt["hit"] else f'   [{vt["distance_in"]:.0f}" off {vt["target_label"]}]'
            lines.append(f"  {shot.index+1:<3} {shot.impact_frame:<7} {sp:<14} {where}")

    if result.warnings:
        lines.append("")
        lines.append("Notes")
        for w in result.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)
